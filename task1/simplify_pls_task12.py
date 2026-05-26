"""Task 1.2: Document-level PLS-grounded simplification using Sonnet via OpenRouter."""

import asyncio
import json
import logging
import os
import re
import openai

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
CONCURRENT = 15
CHUNK_SIZE = 20

SYSTEM_PROMPT = ""

with open("cochrane-auto/data/cochrane_scraped_2026.json") as f:
    SCRAPED = {s["pair_id"]: s for s in json.load(f)}


def clean_pls(pls_text):
    pls = re.sub(r'^Plain language summary available in .*?\n', '', pls_text, flags=re.IGNORECASE)
    pls = re.sub(r'^Plain language summary\s*', '', pls, flags=re.IGNORECASE)
    return pls.strip()


def make_prompt(complex_text, pair_id, language="en"):
    pls_context = ""
    if pair_id in SCRAPED:
        pls = clean_pls(SCRAPED[pair_id]["pls"])[:800]
        pls_context = (
            f"Here is the plain language summary for this Cochrane review:\n"
            f"{pls}\n\n"
            f"Now simplify the following abstract, matching the style and vocabulary "
            f"of the plain language summary above.\n"
        )

    lang_note = ""
    if language != "en":
        lang_note = f"Keep the response in {language}. Do not translate to English.\n"

    if pls_context:
        return (
            f"{pls_context}"
            f"{lang_note}"
            f"Do NOT include any preamble or explanation - output ONLY the simplified text.\n\n"
            f"Abstract: {complex_text}\n\n"
            f"Simplified abstract:"
        )
    return (
        f"Simplify the following biomedical abstract for a general audience.\n"
        f"Make it easier to understand while preserving the key factual information.\n"
        f"Use simpler words, shorter sentences, and avoid jargon.\n"
        f"Do NOT add any information not present in the original.\n"
        f"{lang_note}"
        f"Do NOT include any preamble or explanation - output ONLY the simplified text.\n\n"
        f"Abstract: {complex_text}\n\n"
        f"Simplified abstract:"
    )


def post_process(text):
    out = text
    out = re.sub(r'\blow[- ]certainty\b', 'low quality', out, flags=re.IGNORECASE)
    out = re.sub(r'\bmoderate[- ]certainty\b', 'medium quality', out, flags=re.IGNORECASE)
    out = re.sub(r'\bversus\b', 'compared to', out, flags=re.IGNORECASE)
    out = re.sub(r'\bparticipants\b', 'people', out)
    out = re.sub(r'\bParticipants\b', 'People', out)
    out = re.sub(r'\bplacebo\b', 'dummy treatment', out)
    out = re.sub(r'\banalyses\b', 'results', out, flags=re.IGNORECASE)
    out = re.sub(r'(\d+)\s+trials\b', r'\1 studies', out)
    return out


async def run_task(input_path, output_path):
    with open(input_path) as f:
        data = json.load(f)

    run_id = f"{TEAM_ID}_Task12_pls_grounded"
    log.info("Task 1.2: %d items", len(data))

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

    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    sem = asyncio.Semaphore(CONCURRENT)

    async def process_one(idx, item):
        lang = item.get("language", "en")
        prompt = make_prompt(item["complex"], item["pair_id"], lang)

        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model="anthropic/claude-sonnet-4",
                        max_tokens=2000,
                        temperature=0.2,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    raw = resp.choices[0].message.content.strip()
                    break
                except Exception as e:
                    if "429" in str(e) or "rate" in str(e).lower():
                        await asyncio.sleep(2 ** attempt * 3)
                    else:
                        log.warning("  Error idx %d: %s", idx, e)
                        raw = ""
                        break

        if not raw:
            raw = item["complex"]

        pred = post_process(raw) if lang == "en" else raw
        return idx, {**item, "prediction": pred, "run_id": run_id}

    all_results = dict(done)

    for start in range(0, len(remaining), CHUNK_SIZE):
        chunk = remaining[start:start + CHUNK_SIZE]
        log.info("  Chunk %d-%d of %d", start, start + len(chunk), len(remaining))

        results = await asyncio.gather(*[process_one(idx, item) for idx, item in chunk])
        for idx, result in results:
            all_results[idx] = result

        ordered = [all_results[i] for i in sorted(all_results.keys())]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(ordered, f, ensure_ascii=False)

        chunk_results = [r for _, r in results]
        copies = sum(1 for r in chunk_results if r["complex"].strip() == r["prediction"].strip())
        log.info("  Saved %d total, copies=%d/%d", len(ordered), copies, len(chunk_results))

    log.info("Done: %d predictions", len(all_results))


if __name__ == "__main__":
    asyncio.run(run_task("output/task12_input.json", "output/task12_pls_grounded.json"))
