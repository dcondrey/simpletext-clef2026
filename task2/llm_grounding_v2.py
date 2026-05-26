"""LLM grounding classifier v2 with precision-focused biomedical prompt.

Asks the LLM to identify specific unsupported claims with evidence,
distinguishing between hallucination types.

Usage:
    uv run python llm_grounding_v2.py --split dev --evaluate
    uv run python llm_grounding_v2.py --split test
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path("cache/llm_grounding_v2")

SYSTEM_PROMPT = """You are verifying whether a simplified sentence is fully supported by its source abstract.

The source is a Cochrane systematic review abstract. The sentence is from a simplified version of that abstract. Your job is to determine whether the sentence introduces any consequential information not supported by the source.

CHECK FOR THESE SPECIFIC ERROR TYPES (in order of severity):

1. FABRICATED NUMBERS: The sentence contains a specific number, percentage, confidence interval, p-value, odds ratio, risk ratio, or sample size that does not appear in the source in any form.
   Example: Source says "OR 1.52" but sentence says "OR 2.31"

2. WRONG DIRECTION: The sentence reverses the direction of an effect. Source says risk increased, sentence says decreased. Source says no significant difference, sentence says significant improvement.

3. UNSUPPORTED CERTAINTY: Source uses hedged language (may, might, suggests, limited evidence) but sentence states the claim as definitive fact.
   Example: Source says "may reduce symptoms" but sentence says "reduces symptoms"

4. UNSUPPORTED SCOPE: Source specifies a population, timeframe, or condition that the sentence drops or changes.
   Example: Source says "in adults over 65 with moderate symptoms" but sentence says "in patients"

5. CONFUSED ENTITIES: The sentence attributes a finding to the wrong intervention, comparison, or outcome.

6. INSERTED CONTENT: The sentence contains an entire claim, explanation, or elaboration with no basis in the source at all. This includes instructional text, meta-commentary, or prompt leakage.

IMPORTANT RULES:
- Paraphrasing is ALLOWED. The sentence does not need to match the source verbatim.
- Simplification is ALLOWED. Removing technical detail without changing meaning is fine.
- Rounding is ALLOWED. "23.7%" simplified to "about 24%" is fine.
- Only flag errors that would mislead a reader about the medical evidence.

RESPOND WITH EXACTLY ONE LINE:
- If the sentence is fully supported: SUPPORTED
- If not, state the error type and quote the specific unsupported claim:
  FABRICATED_NUMBER: "[quoted claim]" not found in source
  WRONG_DIRECTION: "[quoted claim]" contradicts source which says "[source quote]"
  UNSUPPORTED_CERTAINTY: "[quoted claim]" but source says "[hedged version]"
  UNSUPPORTED_SCOPE: "[quoted claim]" but source specifies "[qualifier]"
  CONFUSED_ENTITY: "[quoted claim]" but source attributes this to "[correct entity]"
  INSERTED_CONTENT: "[quoted claim]" has no basis in source"""

USER_TEMPLATE = """SOURCE ABSTRACT:
{source}

SENTENCE TO VERIFY:
{sentence}"""


def parse_response(text):
    if not text:
        return {"label": "SUPPORTED", "error_type": None, "evidence": None}
    text = text.strip()
    if text.startswith("SUPPORTED"):
        return {"label": "SUPPORTED", "error_type": None, "evidence": None}
    for error_type in ["FABRICATED_NUMBER", "WRONG_DIRECTION", "UNSUPPORTED_CERTAINTY",
                        "UNSUPPORTED_SCOPE", "CONFUSED_ENTITY", "INSERTED_CONTENT"]:
        if text.startswith(error_type):
            return {"label": "HALLUCINATED", "error_type": error_type, "evidence": text}
    if any(w in text.upper() for w in ["FABRICAT", "WRONG", "UNSUPPORT", "CONFUS", "INSERT", "NOT FOUND", "CONTRADICTS"]):
        return {"label": "HALLUCINATED", "error_type": "UNKNOWN", "evidence": text}
    return {"label": "SUPPORTED", "error_type": None, "evidence": text}


async def call_openai(client, source, sentence, semaphore, model="gpt-4o"):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    async with semaphore:
        for attempt in range(2):
            try:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "max_tokens": 150,
                        "temperature": 0.0,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": USER_TEMPLATE.format(
                                source=source[:4000], sentence=sentence)},
                        ],
                    },
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"]
                    return parse_response(text)
                elif resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt * 2)
                else:
                    return None
            except Exception:
                await asyncio.sleep(2 ** attempt)
    return None


async def run_split(split_name, model="gpt-4o"):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{split_name}_{model}.json"

    docs = json.load(open(f"{split_name}_data.json"))
    log.info("Loaded %d docs from %s_data.json", len(docs), split_name)

    cached = {}
    if cache_path.exists():
        cached = json.load(open(cache_path))
        log.info("Loaded %d cached results", len(cached))

    total_sents = sum(len(d["sentences"]) for d in docs)
    to_process = []
    for d in docs:
        for i, sent in enumerate(d["sentences"]):
            key = f"{d['id']}_{i}"
            if key not in cached:
                to_process.append((d["id"], i, d["source"], sent, key))

    log.info("Total: %d, cached: %d, to process: %d", total_sents, len(cached), len(to_process))

    if not to_process:
        return cached

    semaphore = asyncio.Semaphore(15 if "mini" in model else 8)

    async with httpx.AsyncClient(timeout=90) as client:
        batch_size = 25
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i + batch_size]
            tasks = [call_openai(client, src, sent, semaphore, model)
                     for _, _, src, sent, _ in batch]
            results = await asyncio.gather(*tasks)

            for (_, _, _, _, key), result in zip(batch, results):
                if result is not None:
                    cached[key] = result

            if (i + batch_size) % 250 == 0 or i + batch_size >= len(to_process):
                with open(cache_path, "w") as f:
                    json.dump(cached, f)
                done = min(i + batch_size, len(to_process))
                halluc = sum(1 for v in cached.values() if v.get("label") == "HALLUCINATED")
                log.info("  %d/%d done, %d hallucinated (%.1f%%)",
                         done, len(to_process), halluc, halluc / max(len(cached), 1) * 100)

    with open(cache_path, "w") as f:
        json.dump(cached, f)
    return cached


def evaluate(split_name, model="gpt-4o"):
    cache_path = CACHE_DIR / f"{split_name}_{model}.json"
    if not cache_path.exists():
        log.info("No cache for %s_%s", split_name, model)
        return

    cached = json.load(open(cache_path))
    docs = json.load(open(f"{split_name}_data.json"))

    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    from collections import Counter
    import numpy as np

    preds, gt = [], []
    error_types = Counter()
    for d in docs:
        for i, label in enumerate(d.get("labels", ["None"] * len(d["sentences"]))):
            key = f"{d['id']}_{i}"
            entry = cached.get(key, {"label": "SUPPORTED"})
            pred = 1 if entry.get("label") == "HALLUCINATED" else 0
            true = 0 if label == "None" else 1
            preds.append(pred)
            gt.append(true)
            if pred == 1:
                error_types[entry.get("error_type", "UNKNOWN")] += 1

    preds = np.array(preds)
    gt = np.array(gt)

    tn, fp, fn, tp = confusion_matrix(gt, preds).ravel()
    log.info("=== %s %s evaluation ===", split_name, model)
    log.info("TP=%d FP=%d FN=%d TN=%d", tp, fp, fn, tn)
    log.info("Precision: %.3f", tp / max(tp + fp, 1))
    log.info("Recall:    %.3f", tp / max(tp + fn, 1))
    log.info("F1:        %.3f", f1_score(gt, preds))
    log.info("Halluc rate: %.1f%% (true: %.1f%%)", preds.mean() * 100, gt.mean() * 100)
    log.info("Error types: %s", dict(error_types.most_common()))

    doc_ids = [d["id"] for d in docs for _ in d["sentences"]]
    gt_doc = {d["id"]: int(any(l != "None" for l in d.get("labels", []))) for d in docs}

    import pandas as pd
    df = pd.DataFrame({"doc_id": doc_ids[:len(preds)], "pred": preds})
    dp = {did: int(g["pred"].max() > 0) for did, g in df.groupby("doc_id")}
    y_t = [gt_doc[d] for d in gt_doc]
    y_p = [dp.get(d, 0) for d in gt_doc]
    log.info("Doc F1: %.4f", f1_score(y_t, y_p))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    parser.add_argument("--model", default="gpt-4o")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()

    asyncio.run(run_split(args.split, args.model))
    if args.evaluate:
        evaluate(args.split, args.model)


if __name__ == "__main__":
    main()
