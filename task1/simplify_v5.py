"""
SimpleText Task 1 v5: Multi-candidate generation + SARI-proxy reranking.

Generates N candidates per sentence at varying temperatures, then selects
the best using a reference-free scoring heuristic based on:
- Compression ratio proximity to target
- Word overlap with source (keep score proxy)
- Vocabulary simplicity (average word length reduction)
- Cochrane PLS vocabulary usage (common replacements)
"""

import asyncio
import json
import os
import logging
import openai
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
CONCURRENT = 40
N_CANDIDATES = 5

SYSTEM_PROMPT = (
    "You are a Cochrane plain language summary writer. "
    "You simplify biomedical text by making minimal, targeted changes: "
    "replace medical jargon with everyday words, remove statistical measures "
    "in parentheses (confidence intervals, risk ratios, p-values, I² values, IQR), "
    "but keep important numbers (participant counts, study counts, percentages). "
    "Keep the original sentence structure and wording. Never explain or expand."
)

# Cochrane PLS common vocabulary: words that references tend to use
# These are words that appear frequently in Cochrane plain language summaries
PLS_VOCAB = {
    "people", "patients", "children", "adults", "women", "men", "babies",
    "studies", "trials", "review", "evidence", "results", "found",
    "treatment", "compared", "showed", "reduced", "increased", "improved",
    "difference", "effect", "risk", "may", "probably", "uncertain",
    "low", "moderate", "high", "quality", "certainty", "unclear",
    "side", "effects", "outcomes", "deaths", "pain", "infection",
    "included", "participants", "months", "weeks", "years",
}

SENT_EXAMPLES = """Complex: Computer reminders achieved a median improvement in process adherence of 4.2% (interquartile range (IQR): 0.8% to 18.8%) across all reported process outcomes, 3.3% (IQR: 0.5% to 10.6%) for medication ordering, 3.8% (IQR: 0.5% to 6.6%) for vaccinations, and 3.8% (IQR: 0.4% to 16.3%) for test ordering.
Output: Computer reminders achieved a median improvement of 4.2% across all reported process outcomes, 3.3% for medication ordering, 3.8% for vaccinations, and 3.8% for test ordering.

Complex: TCV compared to control may result in a large reduction in acute typhoid fever (risk ratio (RR) 0.20, 95% confidence interval (CI) 0.12 to 0.32; I2 = 32%; 7 studies, 105,839 participants; low-certainty evidence).
Output: The typhoid conjugate vaccine compared to control may greatly reduce acute typhoid fever (7 studies, 105,839 participants; low-certainty evidence).

Complex: Prophylactic antibiotics did not have an important effect on dyspareunia (difficult or painful sexual intercourse) or breastfeeding at six weeks.
Output: Prophylactic antibiotics did not have a clear effect on pain during sex or breastfeeding at six weeks.

Complex: We included 19 trials (17 RCTs and two cluster-RCTs).
Output: We included 19 trials.

Complex: The 19 trials enrolled 395,650 participants, with ages ranging from six weeks to 60 years.
Output: The 19 trials enrolled 395,650 participants, with ages ranging from six weeks to 60 years.

Complex: There may be little or no difference between the skin closure techniques in terms of incisional hernia and operative time, though the evidence for these two outcomes is very uncertain.
Output: There may be little or no difference between the two techniques in the risk of incisional hernia or in operative time, but the results are very uncertain.

Complex: Taxane-containing regimens appear to improve overall survival, time to progression, and tumour response rate in women with metastatic breast cancer.
Output: Chemotherapy regimens including taxanes improved survival and decreased the progression of metastatic breast cancer."""


def make_prompt(complex_sentence, language="en"):
    if language != "en":
        return (
            f"Simplify this biomedical sentence for a general audience. "
            f"Keep the response in the SAME language as the input ({language}). "
            f"Replace medical/technical terms with simpler words. "
            f"Remove statistical measures in parentheses (CI, RR, p-values, IQR). "
            f"Keep important numbers (participant counts, percentages, ages). "
            f"Keep the sentence structure. "
            f"Output ONLY the simplified sentence.\n\n"
            f"Complex: {complex_sentence}\n\nOutput:"
        )
    return (
        f"Simplify this biomedical sentence for a general audience.\n"
        f"- Remove ONLY statistical measures in parentheses (CI, RR, OR, p-values, IQR, I²)\n"
        f"- Keep important numbers: participant counts, study counts, percentages, ages\n"
        f"- Replace medical/technical terms with common words\n"
        f"- Keep the sentence structure and non-technical words exactly as they are\n"
        f"- Output ONLY the simplified sentence\n\n"
        f"Examples:\n\n"
        f"{SENT_EXAMPLES}\n\n"
        f"Complex: {complex_sentence}\n\nOutput:"
    )


def score_candidate(source, candidate):
    """Reference-free scoring heuristic for candidate selection."""
    if not candidate.strip():
        return -1.0

    src_words = source.lower().split()
    cand_words = candidate.lower().split()
    src_set = set(src_words)
    cand_set = set(cand_words)

    if not cand_words:
        return -1.0

    # 1. Compression ratio score: prefer 0.55-0.85 range
    ratio = len(cand_words) / max(len(src_words), 1)
    if 0.55 <= ratio <= 0.85:
        comp_score = 1.0
    elif ratio < 0.55:
        comp_score = max(0, ratio / 0.55)
    else:
        comp_score = max(0, 1.0 - (ratio - 0.85) / 0.5)

    # 2. Keep score: proportion of source words retained
    kept = src_set & cand_set
    keep_score = len(kept) / max(len(src_set), 1)

    # 3. PLS vocabulary bonus: reward using Cochrane plain language words
    pls_words = cand_set & PLS_VOCAB
    pls_score = len(pls_words) / max(len(cand_set), 1)

    # 4. Vocabulary simplicity: prefer shorter average word length
    avg_src_len = sum(len(w) for w in src_words) / max(len(src_words), 1)
    avg_cand_len = sum(len(w) for w in cand_words) / max(len(cand_words), 1)
    simplicity = max(0, (avg_src_len - avg_cand_len) / avg_src_len) if avg_src_len > 0 else 0

    # 5. Penalty for exact copy
    if candidate.strip() == source.strip():
        return -0.5

    # Weighted combination
    score = (
        comp_score * 0.25 +
        keep_score * 0.40 +
        pls_score * 0.15 +
        simplicity * 0.20
    )
    return score


async def run_task(input_path, output_path, task, model="gpt-4o-mini"):
    with open(input_path) as f:
        data = json.load(f)

    run_id = f"{TEAM_ID}_Task{task.replace('.', '')}_V5"
    log.info(f"Task {task}: {len(data)} items, model={model}, run_id={run_id}, candidates={N_CANDIDATES}")

    # Resume
    done = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        for item in existing:
            if task == "1.1":
                key = (item["pair_id"], str(item.get("para_id")), str(item.get("sent_id")))
            else:
                key = item["pair_id"]
            done[key] = item
        log.info(f"  Resuming: {len(done)} done")

    remaining = []
    for item in data:
        if task == "1.1":
            key = (item["pair_id"], str(item.get("para_id")), str(item.get("sent_id")))
        else:
            key = item["pair_id"]
        if key not in done:
            remaining.append(item)

    if not remaining:
        log.info("  All done")
        return

    log.info(f"  {len(remaining)} to process")

    is_sentence = task == "1.1"
    max_tokens = 300 if is_sentence else 1500

    if model.startswith("gpt-"):
        base_url = "https://api.openai.com/v1"
        api_key = os.environ["OPENAI_API_KEY"]
    elif model.startswith("meta-llama/"):
        base_url = "https://api.together.xyz/v1"
        api_key = os.environ["TOGETHER_API_KEY"]
    else:
        base_url = "https://api.groq.com/openai/v1"
        api_key = os.environ["GROQ_API_KEY"]
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENT)

    temperatures = [0.0, 0.3, 0.5, 0.7, 1.0][:N_CANDIDATES]

    async def generate_one(prompt, temp):
        async with sem:
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temp,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                return None

    async def process_one(item):
        lang = item.get("language", "en")
        prompt = make_prompt(item["complex"], lang)

        # Generate N candidates at different temperatures
        tasks = [generate_one(prompt, t) for t in temperatures]
        candidates = await asyncio.gather(*tasks)
        candidates = [c for c in candidates if c is not None]

        if not candidates:
            return {**item, "prediction": item["complex"], "run_id": run_id}

        # Score and select best
        scored = [(score_candidate(item["complex"], c), c) for c in candidates]
        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]

        # If best is still a copy, try the second-best
        if best.strip() == item["complex"].strip() and len(scored) > 1:
            best = scored[1][1]

        return {**item, "prediction": best, "run_id": run_id}

    chunk_size = 200
    all_results = list(done.values())

    for start in range(0, len(remaining), chunk_size):
        chunk = remaining[start : start + chunk_size]
        log.info(f"  Chunk {start}-{start + len(chunk)} of {len(remaining)}")
        chunk_results = await asyncio.gather(*[process_one(item) for item in chunk])
        all_results.extend(chunk_results)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False)
        log.info(f"  Saved {len(all_results)} total")

        ratios = []
        copies = 0
        for r in chunk_results:
            orig = len(r["complex"].split())
            pred = len(r["prediction"].split())
            if orig > 0:
                ratios.append(pred / orig)
            if r["prediction"].strip() == r["complex"].strip():
                copies += 1
        if ratios:
            avg = sum(ratios) / len(ratios)
            log.info(f"  Chunk compression: {avg:.2f}, copies: {copies}")

    log.info(f"Done: {len(all_results)} predictions")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1.1", "1.2"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--candidates", type=int, default=5)
    args = parser.parse_args()

    N_CANDIDATES = args.candidates
    asyncio.run(run_task(args.input, args.output, args.task, model=args.model))
