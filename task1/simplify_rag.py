"""
SimpleText Task 1: Retrieval-Augmented Simplification.

For each test sentence, retrieves the 3 most similar training pairs
(by TF-IDF cosine similarity) and uses them as few-shot examples.
The model sees exactly how similar sentences were simplified in the
actual Cochrane references.
"""

import asyncio
import ast
import csv
import json
import logging
import os
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import openai

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
CONCURRENT = 20
CHUNK_SIZE = 100
N_EXAMPLES = 3

SYSTEM_PROMPT = """You simplify biomedical sentences for a general audience.
Follow the exact style shown in the examples. Keep as many original words as possible.
Only change what is necessary. Output ONLY the simplified sentence."""


def load_training_pairs(path="cochrane-auto/data/cochraneauto_sents_train.csv"):
    """Load training sentence pairs with references."""
    pairs = []
    with open(path) as f:
        for row in csv.DictReader(f):
            simple = row["simple"].strip()
            if not simple or simple == "[]":
                continue
            try:
                simples = ast.literal_eval(simple)
            except:
                continue
            if not simples:
                continue
            # Take first reference
            pairs.append({
                "complex": row["complex"],
                "simple": simples[0],
            })
    log.info("Loaded %d training pairs", len(pairs))
    return pairs


def build_retriever(pairs):
    """Build TF-IDF index for retrieving similar training examples."""
    corpus = [p["complex"] for p in pairs]
    vectorizer = TfidfVectorizer(
        max_features=20000,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    log.info("Built TF-IDF index: %d docs, %d features", *tfidf_matrix.shape)
    return vectorizer, tfidf_matrix


def retrieve_examples(query, vectorizer, tfidf_matrix, pairs, n=3):
    """Find n most similar training pairs with reasonable compression ratios."""
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_idx = sims.argsort()[::-1]
    results = []
    for i in top_idx:
        p = pairs[i]
        ratio = len(p["simple"].split()) / max(len(p["complex"].split()), 1)
        if 0.5 <= ratio <= 1.3:
            results.append(p)
        if len(results) >= n:
            break
    if not results:
        results = [pairs[i] for i in top_idx[:n]]
    return results


def make_prompt(complex_text, examples, language="en"):
    """Build prompt with retrieved few-shot examples."""
    parts = []
    for ex in examples:
        parts.append(f"Complex: {ex['complex']}")
        parts.append(f"Simple: {ex['simple']}")
        parts.append("")

    lang_note = ""
    if language != "en":
        lang_note = f"\nKeep the response in {language}. Do not translate to English.\n"

    parts.append(f"{lang_note}Complex: {complex_text}")
    parts.append("Simple:")

    return "\n".join(parts)


def postprocess(source, prediction):
    """Clean up model output."""
    if not prediction:
        return source
    prediction = prediction.strip()
    prediction = re.sub(r'^(Simple|Output|Simplified):\s*', '', prediction, flags=re.IGNORECASE)
    prediction = prediction.strip('"\'')

    # Take only first line
    lines = [l.strip() for l in prediction.split('\n') if l.strip()]
    if lines:
        prediction = lines[0]

    # Reject extreme outputs
    src_words = len(source.split())
    pred_words = len(prediction.split())
    if src_words > 0 and pred_words > src_words * 2.0:
        return source
    if pred_words < max(1, src_words * 0.10):
        return source

    return prediction


async def run_task(input_path, output_path, task, model="google/gemma-2-27b-it"):
    with open(input_path) as f:
        data = json.load(f)

    run_id = f"{TEAM_ID}_Task{task.replace('.', '')}_RAG"
    log.info("Task %s: %d items, model=%s, run_id=%s", task, len(data), model, run_id)

    # Load training data and build retriever
    pairs = load_training_pairs()
    vectorizer, tfidf_matrix = build_retriever(pairs)

    # Resume
    done = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        for idx, item in enumerate(existing):
            done[idx] = item
        log.info("  Resuming: %d done", len(done))

    remaining = [(idx, item) for idx, item in enumerate(data) if idx not in done]
    if not remaining:
        log.info("  All done")
        return

    log.info("  %d to process", len(remaining))

    # Setup API client
    base_url = "https://openrouter.ai/api/v1"
    api_key = os.environ["OPENROUTER_API_KEY"]
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENT)

    async def process_one(idx, item):
        lang = item.get("language", "en")

        # Retrieve similar training examples (only for English)
        if lang == "en":
            examples = retrieve_examples(item["complex"], vectorizer, tfidf_matrix, pairs, N_EXAMPLES)
        else:
            # For non-English, use fixed generic examples
            examples = pairs[:N_EXAMPLES]

        prompt = make_prompt(item["complex"], examples, lang)

        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=512,
                    temperature=0.2,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = resp.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate" in err_str.lower():
                    await asyncio.sleep(5)
                else:
                    log.warning("  Error for idx %d: %s", idx, e)
                raw = ""

        prediction = postprocess(item["complex"], raw)
        return idx, {**item, "prediction": prediction, "run_id": run_id}

    all_results = dict(done)

    for start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[start:start + CHUNK_SIZE]
        log.info("  Chunk %d-%d of %d", start, start + len(chunk), len(remaining))

        results = await asyncio.gather(*[process_one(idx, item) for idx, item in chunk])

        for idx, result in results:
            all_results[idx] = result

        # Save incrementally (ordered)
        ordered = [all_results[i] for i in sorted(all_results.keys())]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False)

        # Stats
        chunk_results = [r for _, r in results]
        copies = sum(1 for r in chunk_results if r["complex"].strip() == r["prediction"].strip())
        ratios = [len(r["prediction"].split()) / max(len(r["complex"].split()), 1) for r in chunk_results]
        arr = np.array(ratios)
        log.info("  Compression: mean=%.2f, median=%.2f, copies=%d/%d",
                 arr.mean(), np.median(arr), copies, len(chunk_results))

    log.info("Done: %d predictions saved to %s", len(all_results), output_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1.1", "1.2"], default="1.1")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="google/gemma-2-27b-it")
    parser.add_argument("--concurrent", type=int, default=20)
    parser.add_argument("--examples", type=int, default=3)
    args = parser.parse_args()

    CONCURRENT = args.concurrent
    N_EXAMPLES = args.examples
    asyncio.run(run_task(args.input, args.output, args.task, model=args.model))
