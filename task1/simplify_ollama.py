"""
SimpleText Task 1: Local Gemma 2 9B via Ollama API.
Aggressive Haiku-style simplification targeting SARI > 46.
"""

import asyncio
import json
import logging
import os
import httpx
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
OLLAMA_URL = "http://localhost:11434/api/generate"
CONCURRENT = 4  # M4 can handle ~4 parallel Gemma 2 9B requests
CHUNK_SIZE = 50

PROMPT_TEMPLATE = """Rewrite this biomedical sentence in plain language. Keep as many original words as possible. Only change what is necessary to make it clearer. Output ONLY the rewritten sentence.

Rules:
- Replace jargon with simpler words (participants→people, mortality→death, adverse events→side effects, RCTs→studies)
- Remove statistical parentheticals like (95% CI ...), (RR ...), (I² = ...) but keep study counts
- Keep the sentence roughly the same length
- Do NOT add explanations or extra information

Examples:
Input: We included 19 trials (17 RCTs and two cluster-RCTs).
Output: We included 19 studies.

Input: The 19 trials enrolled 395,650 participants, with ages ranging from six weeks to 60 years.
Output: The 19 trials included 395,650 people, with ages ranging from six weeks to 60 years.

Input: Low-certainty evidence suggests that calcium supplementation compared to placebo or control may result in little to no difference in body weight (mean difference (MD) -0.15 kg, 95% confidence interval (CI) -0.55 to 0.24; P = 0.45, I² = 46%; 17 studies, 1317 participants; low-certainty evidence).
Output: Low-certainty evidence suggests that calcium supplements compared to placebo may make little to no difference in body weight (17 studies, 1317 people; low-certainty evidence).

Input: Prophylactic antibiotics did not have an important effect on dyspareunia (difficult or painful sexual intercourse) or breastfeeding at six weeks.
Output: Preventive antibiotics did not have a clear effect on pain during sex or breastfeeding at six weeks.

Input: Fewer participants experienced constipation with transdermal fentanyl (28%) than with oral morphine (46%).
Output: Fewer people experienced constipation with fentanyl patches (28%) than with oral morphine (46%).

Input: There were no deaths in the included studies.
Output: There were no deaths in the included studies.

Input: {sentence}
Output:"""


async def generate(client, text, language="en"):
    if language != "en":
        prompt = (
            f"Simplify this biomedical sentence for a general audience. "
            f"Keep the response in {language}. Do not translate to English. "
            f"Make it easier to understand. Remove statistical details. "
            f"Output ONLY the simplified sentence.\n\n"
            f"Input: {text}\nOutput:"
        )
    else:
        prompt = PROMPT_TEMPLATE.format(sentence=text)

    try:
        resp = await client.post(
            OLLAMA_URL,
            json={
                "model": "gemma2:9b",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 512,
                    "top_p": 0.9,
                },
            },
            timeout=120.0,
        )
        resp.raise_for_status()
        result = resp.json()["response"].strip()
        # Clean up common artifacts
        for prefix in ["Output:", "Simplified:", "Simple:"]:
            if result.startswith(prefix):
                result = result[len(prefix):].strip()
        result = result.strip('"\'')
        # Remove multi-line responses (take first line only for sentence-level)
        lines = [l.strip() for l in result.split('\n') if l.strip()]
        if lines:
            result = lines[0]
        return result
    except Exception as e:
        log.warning("Error: %s", e)
        return ""


async def run_task(input_path, output_path, task):
    with open(input_path) as f:
        data = json.load(f)

    run_id = f"{TEAM_ID}_Task{task.replace('.', '')}_gemma2_9b"
    log.info("Task %s: %d items, run_id=%s", task, len(data), run_id)

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

    sem = asyncio.Semaphore(CONCURRENT)

    async def process_one(client, idx, item):
        lang = item.get("language", "en")
        async with sem:
            result = await generate(client, item["complex"], lang)

        if not result or len(result.split()) < max(1, len(item["complex"].split()) * 0.1):
            result = item["complex"]

        return idx, {**item, "prediction": result, "run_id": run_id}

    all_results = dict(done)

    async with httpx.AsyncClient() as client:
        for start in range(0, len(remaining), CHUNK_SIZE):
            chunk = remaining[start:start + CHUNK_SIZE]
            log.info("  Chunk %d-%d of %d", start, start + len(chunk), len(remaining))

            tasks = [process_one(client, idx, item) for idx, item in chunk]
            results = await asyncio.gather(*tasks)

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
            import numpy as np
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
    parser.add_argument("--concurrent", type=int, default=4)
    args = parser.parse_args()

    CONCURRENT = args.concurrent
    asyncio.run(run_task(args.input, args.output, args.task))
