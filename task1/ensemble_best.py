"""Best ensemble strategy for SimpleText Task 1.1.

Strategy: 3-model consensus (sonnet, opus_haiku, haikuv2) with source-divergence
tiebreak + statistical sentence routing + regex post-processing.

Local SARI: 41.33 (vs 40.57 for best single model, sonnet)
"""

import json
import re
import sys
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_FILES = {
    "sonnet": "output/task11_sonnet_or.json",
    "opus_haiku": "output/task11_opus_haiku.json",
    "haikuv2": "output/writerslogic_Task11_HaikuV2.json",
}

TOP3 = ["sonnet", "opus_haiku", "haikuv2"]

PP_REGEXES = [
    (r"\blow-certainty\b", "low quality"),
    (r"\bversus\b", "compared to"),
    (r"\bparticipants\b", "people"),
    (r"\bplacebo\b", "dummy treatment"),
    (r"\banalyses\b", "results"),
    (r"\btrials\b", "studies"),
    (r"\bmoderate-certainty\b", "medium quality"),
]


def apply_pp(text):
    for pat, rep in PP_REGEXES:
        text = re.sub(pat, rep, text)
    return text


def ngram_overlap(a, b, ng=2):
    def ngrams(t, n):
        t = t.lower()
        return set(t[i : i + n] for i in range(len(t) - n + 1))

    ga, gb = ngrams(a, ng), ngrams(b, ng)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def has_stats(text):
    return "CI" in text or "confidence interval" in text.lower() or "%" in text


def pick_best(source, candidates):
    """Pick best candidate using consensus + source-divergence tiebreak.

    1. Compute pairwise bigram overlap between all candidates (consensus score).
    2. Among candidates within 95% of max consensus, pick the one most
       different from the source (highest source divergence = more simplification).
    3. For long (>=25 words) statistical sentences, always use sonnet.
    """
    if len(source.split()) >= 25 and has_stats(source):
        return candidates["sonnet"]

    scores = {}
    for na in TOP3:
        scores[na] = sum(
            ngram_overlap(candidates[na], candidates[nb], 2)
            for nb in TOP3
            if nb != na
        )

    max_score = max(scores.values())
    close = [na for na in TOP3 if scores[na] >= max_score * 0.95]

    if len(close) > 1:
        best = min(close, key=lambda nm: ngram_overlap(source, candidates[nm], 2))
    else:
        best = close[0]

    return candidates[best]


def main():
    models = {}
    for name, path in MODEL_FILES.items():
        with open(path) as f:
            models[name] = json.load(f)
        log.info("Loaded %s: %d items", name, len(models[name]))

    indexed = {}
    for name, preds in models.items():
        d = {}
        for item in preds:
            key = (
                item["pair_id"],
                str(item.get("para_id", "")),
                str(item.get("sent_id", "")),
            )
            d[key] = item
        indexed[name] = d

    base = models["sonnet"]
    output = []

    for item in base:
        key = (
            item["pair_id"],
            str(item.get("para_id", "")),
            str(item.get("sent_id", "")),
        )
        src = item["complex"]

        all_have = all(key in indexed[n] for n in TOP3)
        if not all_have:
            pred = item["prediction"]
        else:
            candidates = {nm: indexed[nm][key]["prediction"] for nm in TOP3}
            pred = pick_best(src, candidates)

        pred = apply_pp(pred)
        output.append({**item, "prediction": pred, "run_id": "writerslogic_Task11_ensemble_v3"})

    out_path = "output/task11_ensemble_v3.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    log.info("Written %d items to %s", len(output), out_path)


if __name__ == "__main__":
    main()
