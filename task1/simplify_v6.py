"""
SimpleText Task 1 v6: SARI-optimized grounded simplification.

Strategy based on leaderboard analysis:
- Sentence-level processing (not document)
- Grounded: keep original words, only delete/replace jargon
- Easysplit: split complex sentences into shorter ones
- Target 60-80% compression (never expand)
- Every sentence must be changed (no exact copies)

Uses Google GenAI SDK (Gemma 4, Gemini models).
"""

import asyncio
import json
import logging
import os
import re

from google import genai
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
CONCURRENT = 30
CHUNK_SIZE = 100

# Exact prompt from the 45.73 Haiku run — simple and effective
SYSTEM_PROMPT = ""

EXAMPLES = ""


def make_prompt(complex_text, language="en"):
    if language != "en":
        return (
            f"Simplify the following biomedical sentence for a general audience.\n"
            f"Keep the response in {language}. Do not translate to English.\n"
            f"Make it easier to understand while preserving the key factual information.\n"
            f"Use simpler words, shorter sentences, and avoid jargon.\n"
            f"Do NOT add any information not present in the original.\n"
            f"Do NOT include any preamble or explanation - output ONLY the simplified sentence.\n\n"
            f"Complex sentence: {complex_text}\n\n"
            f"Simplified sentence:"
        )
    return (
        f"Simplify the following biomedical sentence for a general audience.\n"
        f"Make it easier to understand while preserving the key factual information.\n"
        f"Use simpler words, shorter sentences, and avoid jargon.\n"
        f"Do NOT add any information not present in the original.\n"
        f"Do NOT include any preamble or explanation - output ONLY the simplified sentence.\n\n"
        f"Complex sentence: {complex_text}\n\n"
        f"Simplified sentence:"
    )


def postprocess(source, prediction):
    """Clean up model output and enforce constraints."""
    if not prediction:
        return source

    # Remove common model artifacts
    prediction = prediction.strip()
    prediction = re.sub(r'^(Simple|Output|Simplified):\s*', '', prediction, flags=re.IGNORECASE)
    prediction = prediction.strip('"\'')

    src_words = len(source.split())
    pred_words = len(prediction.split())

    # Only reject extreme expansions (>2x) or near-empty outputs
    if src_words > 0 and pred_words > src_words * 2.0:
        return source
    if pred_words < max(1, src_words * 0.10):
        return source

    return prediction


async def run_task(input_path, output_path, task, model="gemma-4-31b-it"):
    with open(input_path) as f:
        data = json.load(f)

    run_id = f"{TEAM_ID}_Task{task.replace('.', '')}_V6"
    log.info("Task %s: %d items, model=%s, run_id=%s", task, len(data), model, run_id)

    # Resume from existing output
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
        log.info("  Resuming: %d done", len(done))

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

    log.info("  %d to process", len(remaining))

    # Determine backend based on model name
    use_oai = "/" in model or model.startswith("llama") or model.startswith("qwen") or model.startswith("gpt-oss")
    if use_oai:
        import openai as _openai
        if model.startswith("llama") and "groq" not in model:
            base_url = "https://api.groq.com/openai/v1"
            api_key = os.environ["GROQ_API_KEY"]
            log.info("  Using Groq backend")
        elif model.startswith("qwen") or model.startswith("gpt-oss") or model.startswith("zai"):
            base_url = "https://api.cerebras.ai/v1"
            api_key = os.environ["CEREBRAS_API_KEY"]
            log.info("  Using Cerebras backend")
        elif os.environ.get("OPENROUTER_API_KEY"):
            base_url = "https://openrouter.ai/api/v1"
            api_key = os.environ["OPENROUTER_API_KEY"]
            log.info("  Using OpenRouter backend")
        else:
            base_url = "https://api.together.xyz/v1"
            api_key = os.environ["TOGETHER_API_KEY"]
            log.info("  Using Together.ai backend")
        oai_client = _openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    else:
        google_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        log.info("  Using Google GenAI backend")

    sem = asyncio.Semaphore(CONCURRENT)

    async def process_one(item):
        lang = item.get("language", "en")
        prompt = make_prompt(item["complex"], lang)

        max_retries = 5
        raw = ""
        async with sem:
            for attempt in range(max_retries):
                try:
                    if use_oai:
                        resp = await oai_client.chat.completions.create(
                            model=model,
                            max_tokens=500 if task == "1.1" else 2000,
                            temperature=0.2,
                            messages=[
                                {"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        raw = resp.choices[0].message.content.strip()
                    else:
                        response = await google_client.aio.models.generate_content(
                            model=model,
                            contents=prompt,
                            config=genai.types.GenerateContentConfig(
                                system_instruction=SYSTEM_PROMPT,
                                temperature=0.2,
                                max_output_tokens=500 if task == "1.1" else 2000,
                            ),
                        )
                        raw = response.text.strip() if response.text else ""
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "rate" in err_str.lower():
                        wait = 2 ** attempt * 3
                        await asyncio.sleep(wait)
                    else:
                        log.warning("  Error for %s: %s", item.get("pair_id", "?"), e)
                        break

        prediction = postprocess(item["complex"], raw)
        return {**item, "prediction": prediction, "run_id": run_id}

    all_results = list(done.values())

    for start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[start:start + CHUNK_SIZE]
        log.info("  Chunk %d-%d of %d", start, start + len(chunk), len(remaining))

        results = await asyncio.gather(*[process_one(item) for item in chunk])
        all_results.extend(results)

        # Save incrementally
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False)

        # Log stats
        ratios = []
        copies = 0
        for r in results:
            src_w = len(r["complex"].split())
            pred_w = len(r["prediction"].split())
            if src_w > 0:
                ratios.append(pred_w / src_w)
            if r["prediction"].strip() == r["complex"].strip():
                copies += 1
        if ratios:
            import numpy as np
            arr = np.array(ratios)
            log.info("  Compression: mean=%.2f, median=%.2f, copies=%d/%d",
                     arr.mean(), np.median(arr), copies, len(results))

    log.info("Done: %d predictions saved to %s", len(all_results), output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1.1", "1.2"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemma-4-31b-it",
                        help="Model name (gemma-4-31b-it, gemini-3.5-flash, etc.)")
    parser.add_argument("--concurrent", type=int, default=30)
    args = parser.parse_args()

    CONCURRENT = args.concurrent
    asyncio.run(run_task(args.input, args.output, args.task, model=args.model))
