"""Local SARI evaluation using available test references."""

import ast
import csv
import json
import logging
import sys

from easse.sari import corpus_sari

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_references(path="cochrane-auto/data/cochraneauto_sents_test.csv"):
    refs = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            s = row.get("simple", "").strip()
            if not s or s == "[]":
                continue
            try:
                simples = ast.literal_eval(s)
            except Exception:
                continue
            if not simples:
                continue
            key = (row["pair_id"], str(row.get("para_id", "")), str(row.get("sent_id", "")))
            refs[key] = simples
    return refs


def evaluate(predictions_path):
    with open(predictions_path) as f:
        preds = json.load(f)

    refs = load_references()
    log.info("Loaded %d references", len(refs))

    sources = []
    predictions = []
    references_list = []

    for item in preds:
        key = (item["pair_id"], str(item.get("para_id", "")), str(item.get("sent_id", "")))
        if key in refs:
            sources.append(item["complex"])
            predictions.append(item["prediction"])
            references_list.append(refs[key])

    log.info("Matched %d items with references", len(sources))

    if not sources:
        log.info("No matches found")
        return

    max_refs = max(len(r) for r in references_list)
    refs_transposed = []
    for i in range(max_refs):
        refs_transposed.append([r[i] if i < len(r) else "" for r in references_list])

    sari = corpus_sari(sources, predictions, refs_transposed)
    log.info("SARI: %.4f (on %d items)", sari, len(sources))

    copies = sum(1 for s, p in zip(sources, predictions) if s.strip() == p.strip())
    log.info("Copies: %d/%d (%.1f%%)", copies, len(sources), copies / len(sources) * 100)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.info("Usage: eval_local.py <predictions.json>")
        sys.exit(1)
    evaluate(sys.argv[1])
