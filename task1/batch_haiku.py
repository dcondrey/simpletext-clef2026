"""
Submit Haiku batch using the EXACT prompt from simplify.py (our best 45.73 score).

Usage:
    uv run python batch_haiku.py submit
    uv run python batch_haiku.py check
    uv run python batch_haiku.py download
"""

import anthropic
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
INPUT_PATH = "output/task11_input.json"
BATCH_ID_FILE = "output/haiku_batch_id.txt"
RESULTS_PATH = "output/task11_haiku_batch_results.jsonl"
OUTPUT_PATH = "output/task11_haiku_batch.json"

PROMPT_TEMPLATE = """\
Simplify the following biomedical sentence for a general audience.
Make it easier to understand while preserving the key factual information.
Use simpler words, shorter sentences, and avoid jargon.
Do NOT add any information not present in the original.
Do NOT include any preamble or explanation - output ONLY the simplified sentence.

Complex sentence: {sentence}

Simplified sentence:"""


def submit_batch():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    log.info("Building %d batch requests...", len(data))

    requests = []
    for idx, item in enumerate(data):
        prompt = PROMPT_TEMPLATE.format(sentence=item["complex"])
        requests.append({
            "custom_id": f"s{idx}",
            "params": {
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 300,
                "messages": [{"role": "user", "content": prompt}],
            }
        })

    log.info("Submitting batch of %d requests...", len(requests))
    client = anthropic.Anthropic()
    batch = client.messages.batches.create(requests=requests)

    log.info("Batch ID: %s", batch.id)
    log.info("Status: %s", batch.processing_status)

    with open(BATCH_ID_FILE, "w") as f:
        f.write(batch.id)

    log.info("Batch ID saved to %s", BATCH_ID_FILE)


def check_batch():
    with open(BATCH_ID_FILE) as f:
        batch_id = f.read().strip()

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)

    log.info("Batch ID: %s", batch.id)
    log.info("Status: %s", batch.processing_status)
    log.info("Counts: %s", batch.request_counts)

    if batch.processing_status == "ended":
        log.info("Batch complete! Run 'download' to get results.")


def download_batch():
    with open(BATCH_ID_FILE) as f:
        batch_id = f.read().strip()

    client = anthropic.Anthropic()
    batch = client.messages.batches.retrieve(batch_id)

    if batch.processing_status != "ended":
        log.info("Batch not done yet. Status: %s", batch.processing_status)
        log.info("Counts: %s", batch.request_counts)
        return

    log.info("Downloading results...")

    with open(RESULTS_PATH, "w") as out:
        for result in client.messages.batches.results(batch_id):
            out.write(json.dumps(result.model_dump(), default=str) + "\n")

    log.info("Results saved to %s", RESULTS_PATH)

    with open(INPUT_PATH) as f:
        input_data = json.load(f)

    result_by_idx = {}
    succeeded = 0
    failed = 0

    with open(RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            idx = int(r["custom_id"][1:])
            if r["result"]["type"] == "succeeded":
                result_by_idx[idx] = r["result"]["message"]["content"][0]["text"].strip()
                succeeded += 1
            else:
                failed += 1

    results = []
    for idx, item in enumerate(input_data):
        prediction = result_by_idx.get(idx, item["complex"])
        results.append({
            **item,
            "prediction": prediction,
            "run_id": f"{TEAM_ID}_task11_haiku_batch",
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    log.info("Done: %d succeeded, %d failed, %d total", succeeded, failed, len(results))
    log.info("Output: %s", OUTPUT_PATH)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.info("Usage: batch_haiku.py [submit|check|download]")
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "submit":
        submit_batch()
    elif cmd == "check":
        check_batch()
    elif cmd == "download":
        download_batch()
    else:
        log.error("Unknown command: %s", cmd)
