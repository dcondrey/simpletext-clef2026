"""
SimpleText Task 1: Local model simplification via Ollama.
No 3rd party API required.
"""

import json
import os
import logging
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"
WORKERS = 4


def ollama_generate(prompt, max_tokens=512):
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0.3},
    }, timeout=120)
    resp.raise_for_status()
    text = resp.json()["response"].strip()
    # Qwen3 may include <think> blocks - strip them
    if "<think>" in text:
        import re
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def simplify_sentence(complex_sentence, language="en"):
    if language != "en":
        prompt = (
            f"/no_think\nSimplify this biomedical sentence for a general audience. "
            f"Keep the SAME language as the input. Use simpler words. Preserve key facts. "
            f"Output ONLY the simplified sentence.\n\n"
            f"Complex: {complex_sentence}\n\nSimplified:"
        )
    else:
        prompt = (
            f"/no_think\nSimplify this biomedical sentence for a general audience. "
            f"Use simpler words and shorter sentences. Avoid jargon. "
            f"Preserve key factual information. Do NOT add new information. "
            f"Output ONLY the simplified sentence.\n\n"
            f"Complex: {complex_sentence}\n\nSimplified:"
        )
    return ollama_generate(prompt, max_tokens=300)


def simplify_document(complex_document, language="en"):
    if language != "en":
        prompt = (
            f"/no_think\nSimplify this biomedical abstract for a general audience. "
            f"Keep the SAME language as the input. "
            f"Rewrite as a plain language summary using simpler words. "
            f"You may merge, split, or reorder sentences. Do NOT add new information. "
            f"Output ONLY the simplified text.\n\n"
            f"Complex:\n{complex_document}\n\nSimplified:"
        )
    else:
        prompt = (
            f"/no_think\nSimplify this biomedical abstract for a general audience. "
            f"Rewrite as a plain language summary. Use simpler words and shorter sentences. "
            f"Avoid technical jargon. You may merge, split, or reorder sentences. "
            f"Do NOT add new information. Output ONLY the simplified text.\n\n"
            f"Complex:\n{complex_document}\n\nSimplified:"
        )
    return ollama_generate(prompt, max_tokens=1000)


def process_item(item, task, run_id):
    lang = item.get("language", "en")
    for attempt in range(3):
        try:
            if task == "1.1":
                prediction = simplify_sentence(item["complex"], lang)
            else:
                prediction = simplify_document(item["complex"], lang)
            return {**item, "prediction": prediction, "run_id": run_id}
        except Exception as e:
            if attempt < 2:
                time.sleep(2)
            else:
                log.warning(f"Failed: {e}")
                return {**item, "prediction": item["complex"], "run_id": run_id}


def run_task(input_path, output_path, task):
    with open(input_path) as f:
        data = json.load(f)
    log.info(f"Task {task}: {len(data)} items")

    run_id = f"{TEAM_ID}_task{task.replace('.', '')}_qwen3"

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

    all_results = list(done.values())
    save_every = 200

    with ThreadPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(process_item, item, task, run_id): i
                   for i, item in enumerate(remaining)}

        for i, future in enumerate(tqdm(as_completed(futures), total=len(futures),
                                        desc=f"Task {task}")):
            result = future.result()
            all_results.append(result)

            if (i + 1) % save_every == 0:
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, ensure_ascii=False)
                log.info(f"  Saved {len(all_results)} predictions")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False)
    log.info(f"Done: {len(all_results)} predictions in {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1.1", "1.2"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    MODEL = args.model
    WORKERS = args.workers
    run_task(args.input, args.output, args.task)
