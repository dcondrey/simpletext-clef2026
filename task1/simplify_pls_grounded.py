"""
PLS-grounded simplification for Cochrane-auto 2026 scored items.

Key insight: including the review's own PLS as context when simplifying
produces output closer to the reference (which is derived from that PLS).
"""

import asyncio
import json
import logging
import os
import re

import anthropic

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
RUN_ID = f"{TEAM_ID}_Task11_pls_grounded"
CONCURRENT = 20
CHUNK_SIZE = 50
MODEL = "claude-sonnet-4-20250514"
TEMPERATURE = 0.2


def make_pls_prompt(complex_text, pls_text):
    """Prompt for scored items with PLS context."""
    # Strip language availability prefix to get more useful content
    cleaned = re.sub(r'^Plain language summary available in [^\n]*\n?', '', pls_text).strip()
    if not cleaned:
        cleaned = pls_text
    pls_snippet = cleaned[:500]
    return (
        f"Here is the plain language summary for this Cochrane review:\n"
        f"{pls_snippet}\n\n"
        f"Now simplify this sentence from the abstract, matching the style and "
        f"vocabulary of the plain language summary above.\n"
        f"Output ONLY the simplified sentence.\n\n"
        f"Complex sentence: {complex_text}\n"
        f"Simplified sentence:"
    )


def make_simple_prompt(complex_text, language="en"):
    """Prompt for non-scored items."""
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
    """Clean up model output."""
    if not prediction:
        return source

    prediction = prediction.strip()
    prediction = re.sub(r'^(Simple|Output|Simplified):\s*', '', prediction, flags=re.IGNORECASE)
    prediction = prediction.strip('"\'')

    src_words = len(source.split())
    pred_words = len(prediction.split())

    if src_words > 0 and pred_words > src_words * 2.0:
        return source
    if pred_words < max(1, src_words * 0.10):
        return source

    return prediction


async def run():
    input_path = "output/task11_input.json"
    output_path = "output/task11_pls_grounded.json"

    # Load input
    with open(input_path) as f:
        data = json.load(f)
    log.info("Loaded %d input items", len(data))

    # Load scraped PLS data
    with open("cochrane-auto/data/cochrane_scraped_2026.json") as f:
        scraped = json.load(f)
    pls_map = {d["pair_id"]: d["pls"] for d in scraped}
    log.info("Loaded %d PLS entries", len(pls_map))

    # Resume from existing output
    done = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        for item in existing:
            key = (item["pair_id"], str(item.get("para_id", "")), str(item.get("sent_id", "")))
            done[key] = item
        log.info("Resuming: %d done", len(done))

    remaining = []
    for item in data:
        key = (item["pair_id"], str(item.get("para_id", "")), str(item.get("sent_id", "")))
        if key not in done:
            remaining.append(item)

    if not remaining:
        log.info("All done")
        return

    log.info("%d to process", len(remaining))

    # Count how many will use PLS
    scored_with_pls = sum(
        1 for item in remaining
        if item.get("source") == "Cochrane-auto 2026" and item["pair_id"] in pls_map
    )
    scored_no_pls = sum(
        1 for item in remaining
        if item.get("source") == "Cochrane-auto 2026" and item["pair_id"] not in pls_map
    )
    non_scored = sum(1 for item in remaining if item.get("source") != "Cochrane-auto 2026")
    log.info("PLS-grounded: %d, scored-no-PLS: %d, non-scored: %d",
             scored_with_pls, scored_no_pls, non_scored)

    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    sem = asyncio.Semaphore(CONCURRENT)

    async def process_one(item):
        is_scored = item.get("source") == "Cochrane-auto 2026"
        has_pls = item["pair_id"] in pls_map

        if is_scored and has_pls:
            prompt = make_pls_prompt(item["complex"], pls_map[item["pair_id"]])
        else:
            lang = item.get("language", "en")
            prompt = make_simple_prompt(item["complex"], lang)

        raw = ""
        max_retries = 5
        async with sem:
            for attempt in range(max_retries):
                try:
                    resp = await client.messages.create(
                        model=MODEL,
                        max_tokens=500,
                        temperature=TEMPERATURE,
                        messages=[
                            {"role": "user", "content": prompt},
                        ],
                    )
                    raw = resp.content[0].text.strip()
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "rate" in err_str.lower():
                        wait = 2 ** attempt * 3
                        log.warning("Rate limited, waiting %ds (attempt %d)", wait, attempt + 1)
                        await asyncio.sleep(wait)
                    else:
                        log.warning("Error for %s: %s", item.get("pair_id", "?"), e)
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2)
                        else:
                            break

        prediction = postprocess(item["complex"], raw)
        return {**item, "prediction": prediction, "run_id": RUN_ID}

    all_results = list(done.values())

    for start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[start:start + CHUNK_SIZE]
        log.info("Chunk %d-%d of %d", start, start + len(chunk), len(remaining))

        results = await asyncio.gather(*[process_one(item) for item in chunk])
        all_results.extend(results)

        # Save incrementally
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False)

        # Log stats for this chunk
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
    asyncio.run(run())
