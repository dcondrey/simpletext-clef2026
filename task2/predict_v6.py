"""Generate v6 submissions with optimized thresholds.

Key change from v5: doc-level threshold lowered from 0.74 to 0.43 (CV-optimal).
Also generates multiple threshold variants for A/B testing on CodaBench.
"""

import json
import logging
import os
import pickle
import subprocess

import numpy as np
import pandas as pd

from feature_extraction import LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = "models_v5"
SUBMISSION_DIR = "submissions_v6"


def make_binary_labels(sent_probs, doc_thresh, sent_thresh):
    doc_positive = np.max(sent_probs) > doc_thresh
    if doc_positive:
        labels = ["Overgeneration" if p > sent_thresh else "None" for p in sent_probs]
        if all(l == "None" for l in labels):
            labels[int(np.argmax(sent_probs))] = "Overgeneration"
        return labels
    return ["None"] * len(sent_probs)


def make_multiclass_labels(bin_labels, mc_probs, offsets):
    mc_adj = mc_probs.copy()
    for cls in range(len(LABELS)):
        mc_adj[:, cls] += offsets[cls]

    labels = []
    for j, bl in enumerate(bin_labels):
        if bl != "None":
            pred_cls = int(np.argmax(mc_adj[j]))
            if pred_cls == 0:
                pred_cls = int(np.argmax(mc_adj[j, 1:])) + 1
            labels.append(LABELS[pred_cls])
        else:
            labels.append("None")
    return labels


def generate_submission(test_docs, bin_probs, mc_probs, offsets,
                        doc_thresh, sent_thresh, run_name):
    results_21 = []
    results_22 = []
    idx = 0
    for doc in test_docs:
        n = len(doc["sentences"])
        s_bin = bin_probs[idx:idx + n]
        s_mc = mc_probs[idx:idx + n]
        idx += n

        bin_labels = make_binary_labels(s_bin, doc_thresh, sent_thresh)
        mc_labels = make_multiclass_labels(bin_labels, s_mc, offsets)

        results_21.append({
            "id": doc["id"],
            "labels": bin_labels,
            "run_id": f"writerslogic_Task21b_{run_name}",
        })
        results_22.append({
            "id": doc["id"],
            "labels": mc_labels,
            "run_id": f"writerslogic_Task22b_{run_name}",
        })

    return results_21, results_22


def save_submission(results, filename):
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    json_path = os.path.join(SUBMISSION_DIR, filename)
    zip_path = json_path.replace(".json", ".zip")

    with open(json_path, "w") as f:
        json.dump(results, f)
    subprocess.run(["zip", "-j", zip_path, json_path], check=True,
                   capture_output=True)

    total_sents = sum(len(r["labels"]) for r in results)
    non_none = sum(1 for r in results for l in r["labels"] if l != "None")
    pos_docs = sum(1 for r in results if any(l != "None" for l in r["labels"]))
    log.info("  %s: %d/%d non-None sents, %d/%d pos docs",
             filename, non_none, total_sents, pos_docs, len(results))


def main():
    log.info("Loading artifacts...")
    with open(os.path.join(MODEL_DIR, "artifacts.pkl"), "rb") as f:
        art = pickle.load(f)

    log.info("Loading cached test features...")
    test_df = pd.read_pickle(os.path.join(MODEL_DIR, "cache", "test_df.pkl"))

    log.info("Loading test data...")
    with open("test_data.json") as f:
        test_docs = json.load(f)

    feature_cols = art["feature_cols"]
    X_test = test_df[feature_cols].values

    # LGB-only ensemble (best on dev)
    bin_probs = np.mean([m.predict(X_test) for m in art["bin_models"]], axis=0)
    mc_probs = np.mean([m.predict(X_test) for m in art["mc_models"]], axis=0)
    offsets = art["perclass_offsets"]
    sent_thresh = 0.62  # CV-optimal sentence threshold

    # Generate submissions at multiple doc thresholds
    variants = [
        ("v6_t40", 0.40),
        ("v6_t43", 0.43),  # CV-optimal on dev
        ("v6_t45", 0.45),
        ("v6_t48", 0.48),
        ("v6_t50", 0.50),
    ]

    for run_name, doc_thresh in variants:
        log.info("Generating %s (doc_thresh=%.2f, sent_thresh=%.2f)...",
                 run_name, doc_thresh, sent_thresh)
        r21, r22 = generate_submission(
            test_docs, bin_probs, mc_probs, offsets, doc_thresh, sent_thresh, run_name)
        save_submission(r21, f"writerslogic_Task21b_{run_name}.json")
        save_submission(r22, f"writerslogic_Task22b_{run_name}.json")

    log.info("All submissions saved to %s/", SUBMISSION_DIR)


if __name__ == "__main__":
    main()
