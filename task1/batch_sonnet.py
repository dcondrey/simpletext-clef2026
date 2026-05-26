"""
Generate Sonnet simplifications via Anthropic Message Batches API.

Usage:
    uv run python batch_sonnet.py submit   # Submit batch
    uv run python batch_sonnet.py check    # Check status
    uv run python batch_sonnet.py download # Download results
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
BATCH_ID_FILE = "output/sonnet_batch_id.txt"
RESULTS_PATH = "output/task11_sonnet_batch_results.jsonl"
OUTPUT_PATH = "output/task11_v8_sonnet.json"

SYSTEM_PROMPT = """\
You are a Cochrane plain language summary writer. Rewrite the sentence in simpler English.

Rules:
- Keep the same meaning and most of the same words
- Replace medical jargon with common words (e.g. participants→people, mortality→death, adverse events→side effects, RCTs→studies)
- Remove statistical details in parentheses like (95% CI ...), (RR ...), (I² = ...) but keep study counts and participant numbers
- Keep the sentence roughly the same length — do not over-compress or expand
- Do NOT add explanations or new information
- Do NOT unnecessarily rephrase words that are already simple
- Output ONLY the simplified sentence"""

EXAMPLES = [
    ("We included 19 trials (17 RCTs and two cluster-RCTs).",
     "We included 19 trials."),
    ("The 19 trials enrolled 395,650 participants, with ages ranging from six weeks to 60 years.",
     "The 19 trials included 395,650 people, with ages ranging from six weeks to 60 years."),
    ("Low-certainty evidence suggests that calcium supplementation compared to placebo or control may result in little to no difference in body weight (mean difference (MD) -0.15 kg, 95% confidence interval (CI) -0.55 to 0.24; P = 0.45, I² = 46%; 17 studies, 1317 participants; low-certainty evidence).",
     "Low-certainty evidence suggests that calcium supplements compared to placebo may make little to no difference in body weight (17 studies, 1317 people; low-certainty evidence)."),
    ("Prophylactic antibiotics did not have an important effect on dyspareunia (difficult or painful sexual intercourse) or breastfeeding at six weeks.",
     "Preventive antibiotics did not have a clear effect on pain during sex or breastfeeding at six weeks."),
    ("There were no deaths in the included studies.",
     "There were no deaths in the included studies."),
    ("Fewer participants experienced constipation with transdermal fentanyl (28%) than with oral morphine (46%).",
     "Fewer people experienced constipation with transdermal fentanyl (28%) than with oral morphine (46%)."),
]


def build_messages(complex_text, language="en"):
    messages = []
    for src, tgt in EXAMPLES:
        messages.append({"role": "user", "content": f"Simplify: {src}"})
        messages.append({"role": "assistant", "content": tgt})

    lang_note = ""
    if language != "en":
        lang_note = f" (Keep the response in {language}. Do not translate to English.)"

    messages.append({"role": "user", "content": f"Simplify{lang_note}: {complex_text}"})
    return messages


def submit_batch():
    with open(INPUT_PATH) as f:
        data = json.load(f)

    log.info("Building %d batch requests...", len(data))

    requests = []
    for idx, item in enumerate(data):
        custom_id = f"s{idx}"
        messages = build_messages(item["complex"], item.get("language", "en"))

        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": "claude-sonnet-4-20250514",
                "max_tokens": 1024,
                "temperature": 0.2,
                "system": SYSTEM_PROMPT,
                "messages": messages,
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

    # Stream results to JSONL
    with open(RESULTS_PATH, "w") as out:
        for result in client.messages.batches.results(batch_id):
            out.write(json.dumps(result.model_dump(), default=str) + "\n")

    log.info("Results saved to %s", RESULTS_PATH)

    # Parse into submission format
    with open(INPUT_PATH) as f:
        input_data = json.load(f)

    # Index results by custom_id (s0, s1, s2, ...)
    result_by_idx = {}
    succeeded = 0
    failed = 0

    with open(RESULTS_PATH) as f:
        for line in f:
            r = json.loads(line)
            idx = int(r["custom_id"][1:])  # "s123" -> 123
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
            "run_id": f"{TEAM_ID}_Task11_V8_sonnet",
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)

    log.info("Done: %d succeeded, %d failed, %d total", succeeded, failed, len(results))
    log.info("Output: %s", OUTPUT_PATH)

    # Stats
    import numpy as np
    copies = sum(1 for r in results if r["complex"].strip() == r["prediction"].strip())
    ratios = [len(r["prediction"].split()) / max(len(r["complex"].split()), 1) for r in results]
    ra = np.array(ratios)
    log.info("Copies: %d (%.1f%%), Compression: mean=%.3f median=%.3f",
             copies, copies / len(results) * 100, ra.mean(), np.median(ra))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.info("Usage: batch_sonnet.py [submit|check|download]")
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
