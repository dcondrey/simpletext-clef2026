"""Generate v7 submissions with expanded features and optimized thresholds."""

import json
import logging
import os
import pickle
import subprocess

import numpy as np
import pandas as pd

from feature_extraction import (
    LABELS, build_dataset,
    compute_embedding_features,
)
from train_v7 import add_llm_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = "models_v7"
SUBMISSION_DIR = "submissions_v7"


def make_binary_labels(sent_probs, doc_thresh, sent_thresh):
    doc_positive = np.max(sent_probs) > doc_thresh
    if doc_positive:
        labels = ["Overgeneration" if p > sent_thresh else "None" for p in sent_probs]
        if all(l == "None" for l in labels):
            labels[int(np.argmax(sent_probs))] = "Overgeneration"
        return labels
    return ["None"] * len(sent_probs)


def make_multiclass_labels(bin_labels, mc_probs):
    labels = []
    for j, bl in enumerate(bin_labels):
        if bl != "None":
            pred_cls = int(np.argmax(mc_probs[j]))
            if pred_cls == 0:
                pred_cls = int(np.argmax(mc_probs[j, 1:])) + 1
            labels.append(LABELS[pred_cls])
        else:
            labels.append("None")
    return labels


def save_submission(results, filename):
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    json_path = os.path.join(SUBMISSION_DIR, filename)
    zip_path = json_path.replace(".json", ".zip")
    with open(json_path, "w") as f:
        json.dump(results, f)
    subprocess.run(["zip", "-j", zip_path, json_path], check=True, capture_output=True)
    total_sents = sum(len(r["labels"]) for r in results)
    non_none = sum(1 for r in results for l in r["labels"] if l != "None")
    pos_docs = sum(1 for r in results if any(l != "None" for l in r["labels"]))
    log.info("  %s: %d/%d non-None sents, %d/%d pos docs",
             filename, non_none, total_sents, pos_docs, len(results))


def main():
    log.info("Loading artifacts...")
    with open(os.path.join(MODEL_DIR, "artifacts.pkl"), "rb") as f:
        art = pickle.load(f)

    feature_cols = art["feature_cols"]
    doc_thresh = art["doc_thresh"]
    sent_thresh = art["sent_thresh"]
    log.info("Features: %d, doc_thresh: %.3f, sent_thresh: %.3f",
             len(feature_cols), doc_thresh, sent_thresh)

    log.info("Loading test data...")
    test_docs = json.load(open("test_data.json"))
    log.info("  %d docs, %d sents", len(test_docs), sum(len(d["sentences"]) for d in test_docs))

    cache_path = os.path.join(MODEL_DIR, "cache", "test_df.pkl")
    if os.path.exists(cache_path):
        log.info("Loading cached test features...")
        test_df = pd.read_pickle(cache_path)
    else:
        import gc
        from sentence_transformers import SentenceTransformer
        log.info("Computing test embeddings...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embed_feats = compute_embedding_features(test_docs, embed_model)
        del embed_model
        gc.collect()

        log.info("Building test features...")
        test_df = build_dataset(test_docs, embed_feats)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        test_df.to_pickle(cache_path)
        log.info("Cached to %s", cache_path)

    test_df = add_llm_features(test_df, test_docs, "test")

    missing = [c for c in feature_cols if c not in test_df.columns]
    if missing:
        log.info("Zero-filling %d missing features: %s", len(missing), missing)
        for c in missing:
            test_df[c] = 0.0

    X_test = test_df[feature_cols].values
    bin_probs = np.mean([m.predict(X_test) for m in art["bin_models"]], axis=0)
    mc_probs = np.mean([m.predict(X_test) for m in art["mc_models"]], axis=0)

    for run_name, d_t, s_t in [
        ("v7_opt", doc_thresh, sent_thresh),
        ("v7_t35", 0.35, sent_thresh),
        ("v7_t40", 0.40, sent_thresh),
        ("v7_t45", 0.45, sent_thresh),
    ]:
        log.info("Generating %s (doc=%.2f, sent=%.2f)...", run_name, d_t, s_t)
        r21, r22 = [], []
        idx = 0
        for doc in test_docs:
            n = len(doc["sentences"])
            s_bin = bin_probs[idx:idx + n]
            s_mc = mc_probs[idx:idx + n]
            idx += n

            bl = make_binary_labels(s_bin, d_t, s_t)
            ml = make_multiclass_labels(bl, s_mc)

            r21.append({"id": doc["id"], "labels": bl,
                        "run_id": f"writerslogic_Task21b_{run_name}"})
            r22.append({"id": doc["id"], "labels": ml,
                        "run_id": f"writerslogic_Task22b_{run_name}"})

        save_submission(r21, f"writerslogic_Task21b_{run_name}.json")
        save_submission(r22, f"writerslogic_Task22b_{run_name}.json")

    log.info("All submissions saved to %s/", SUBMISSION_DIR)


if __name__ == "__main__":
    main()
