"""
Opus batch on scored subset only (Cochrane-auto 2026, 3679 items).
Runs two prompts: aggressive simplification + additions-focused.
"""

import anthropic
import json
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
INPUT_PATH = "output/task11_input.json"
BATCH_ID_FILE = "output/opus_batch_id_{variant}.txt"
RESULTS_PATH = "output/task11_opus_{variant}_results.jsonl"
OUTPUT_PATH = "output/task11_opus_{variant}.json"

PROMPT_A = """Simplify the following biomedical sentence for a general audience.
Make it easier to understand while preserving the key factual information.
Use simpler words, shorter sentences, and avoid jargon.
Do NOT add any information not present in the original.
Do NOT include any preamble or explanation - output ONLY the simplified sentence.

Complex sentence: {sentence}

Simplified sentence:"""

PROMPT_B = """Rewrite this for a Cochrane Plain Language Summary. Your audience is someone with no medical training.
Write naturally, as if explaining to a friend. Use everyday words instead of medical jargon.
Remove statistical measures in parentheses but keep the key numbers and findings.
Do NOT include any preamble or explanation - output ONLY the rewritten sentence.

Complex sentence: {sentence}

Simplified sentence:"""


def submit_batch(variant="a"):
    with open(INPUT_PATH) as f:
        data = json.load(f)

    scored = [(i, item) for i, item in enumerate(data) if item.get("source") == "Cochrane-auto 2026"]
    log.info("Scored subset: %d items (variant=%s)", len(scored), variant)

    prompt_template = PROMPT_A if variant == "a" else PROMPT_B

    requests = []
    for idx, item in scored:
        lang = item.get("language", "en")
        prompt = prompt_template.format(sentence=item["complex"])
        if lang != "en":
            prompt = f"Keep the response in {lang}. Do not translate to English.\n\n" + prompt

        requests.append({
            "custom_id": f"s{idx}",
            "params": {
                "model": "claude-opus-4-20250514",
                "max_tokens": 512,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            }
        })

    log.info("Submitting %d requests...", len(requests))
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)

    batch_file = BATCH_ID_FILE.format(variant=variant)
    with open(batch_file, "w") as f:
        f.write(batch.id)

    log.info("Batch ID: %s (saved to %s)", batch.id, batch_file)
    log.info("Status: %s", batch.processing_status)


def check_batch(variant="a"):
    batch_file = BATCH_ID_FILE.format(variant=variant)
    with open(batch_file) as f:
        batch_id = f.read().strip()

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)
    log.info("Batch %s: %s — %s", variant, batch.processing_status, batch.request_counts)


def download_batch(variant="a"):
    batch_file = BATCH_ID_FILE.format(variant=variant)
    with open(batch_file) as f:
        batch_id = f.read().strip()

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        log.info("Not done: %s — %s", batch.processing_status, batch.request_counts)
        return

    results_file = RESULTS_PATH.format(variant=variant)
    with open(results_file, "w") as out:
        for result in client.messages.batches.results(batch_id):
            out.write(json.dumps(result.model_dump(), default=str) + "\n")

    # Parse results
    with open(INPUT_PATH) as f:
        input_data = json.load(f)

    result_by_idx = {}
    succeeded = failed = 0
    with open(results_file) as f:
        for line in f:
            r = json.loads(line)
            idx = int(r["custom_id"][1:])
            if r["result"]["type"] == "succeeded":
                content = r["result"]["message"].get("content", [])
                if content and content[0].get("text"):
                    result_by_idx[idx] = content[0]["text"].strip()
                succeeded += 1
            else:
                failed += 1

    # Build output for scored subset only
    output = []
    for i, item in enumerate(input_data):
        if item.get("source") == "Cochrane-auto 2026":
            pred = result_by_idx.get(i, item["complex"])
            output.append({**item, "prediction": pred,
                          "run_id": f"{TEAM_ID}_Task11_opus_{variant}"})

    output_file = OUTPUT_PATH.format(variant=variant)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)

    log.info("Done: %d succeeded, %d failed → %s", succeeded, failed, output_file)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.info("Usage: batch_opus.py [submit|check|download] [a|b]")
        sys.exit(1)

    cmd = sys.argv[1]
    variant = sys.argv[2] if len(sys.argv) > 2 else "a"

    if cmd == "submit":
        submit_batch(variant)
    elif cmd == "check":
        check_batch(variant)
    elif cmd == "download":
        download_batch(variant)
