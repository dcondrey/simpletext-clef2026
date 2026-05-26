"""Local SARI evaluation using scraped Cochrane PLS references."""

import json
import logging
import sys

from easse.sari import corpus_sari

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def evaluate(predictions_path):
    with open(predictions_path) as f:
        preds = json.load(f)

    with open("cochrane-auto/data/scraped_refs.json") as f:
        raw_refs = json.load(f)

    refs = {tuple(k.split("|")): v for k, v in raw_refs.items()}
    log.info("Loaded %d scraped references", len(refs))

    sources = []
    predictions = []
    references = []

    for item in preds:
        key = (item["pair_id"], str(item.get("para_id", "")), str(item.get("sent_id", "")))
        if key in refs:
            sources.append(item["complex"])
            predictions.append(item["prediction"])
            references.append(refs[key])

    log.info("Matched %d items", len(sources))

    if not sources:
        log.info("No matches")
        return

    sari = corpus_sari(sources, predictions, [references])
    log.info("SARI: %.4f (%d items)", sari, len(sources))

    copies = sum(1 for s, p in zip(sources, predictions) if s.strip() == p.strip())
    ratios = [len(p.split()) / max(len(s.split()), 1) for s, p in zip(sources, predictions)]
    import numpy as np
    r = np.array(ratios)
    log.info("Copies: %d (%.1f%%), Compression: %.3f", copies, copies / len(sources) * 100, r.mean())


if __name__ == "__main__":
    if len(sys.argv) < 2:
        log.info("Usage: eval_scraped.py <predictions.json>")
        sys.exit(1)
    evaluate(sys.argv[1])
