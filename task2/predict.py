"""
Inference and submission generation for SimpleText Task 2 v5.

Loads trained models, extracts features from test data, applies two-threshold
document aggregation, and generates submission files for both Task 2.1 and 2.2.

Produces four submission files:
- writerslogic_Task21b_v4_strong.json (full ensemble, binary)
- writerslogic_Task22b_v4_strong.json (full ensemble, multi-class)
- writerslogic_Task21b_v4_fast.json (light model, binary)
- writerslogic_Task22b_v4_fast.json (light model, multi-class)
"""

import json
import logging
import os
import pickle
import subprocess

import numpy as np
import pandas as pd
from tqdm import tqdm

from feature_extraction import (
    LABELS, build_dataset,
    compute_embedding_features, compute_nli_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

MODEL_DIR = "models_v5"
SUBMISSION_DIR = "submissions_v5"


def load_artifacts():
    """Load all trained model artifacts."""
    path = os.path.join(MODEL_DIR, "artifacts.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_strong(test_df, artifacts):
    """Run full ensemble inference with meta-learner and calibration.

    Returns calibrated sentence-level binary probabilities and multiclass probabilities.
    """
    import xgboost as xgb

    feature_cols = artifacts["feature_cols"]
    X_test = test_df[feature_cols].values

    # LightGBM binary ensemble (5 seeds)
    bin_probs = np.mean([m.predict(X_test) for m in artifacts["bin_models"]], axis=0)

    # LightGBM multiclass ensemble (5 seeds)
    mc_probs = np.mean([m.predict(X_test) for m in artifacts["mc_models"]], axis=0)

    # XGBoost binary
    xgb_probs = artifacts["xgb_model"].predict(xgb.DMatrix(X_test))

    # Blend binary
    weights = artifacts["blend_weights"]
    blended_bin = weights["lgb"] * bin_probs + weights["xgb"] * xgb_probs

    # Meta-learner
    from train import build_meta_features
    meta_X = build_meta_features(blended_bin, mc_probs, test_df)
    meta_probs = artifacts["meta_model"].predict_proba(meta_X)[:, 1]

    # Apply calibrator if available, otherwise use raw meta probs
    calibrator = artifacts.get("calibrator")
    if calibrator is not None:
        calibrated = calibrator.predict(meta_probs)
    else:
        calibrated = meta_probs

    # Apply per-class offsets to multiclass
    offsets = artifacts["perclass_offsets"]
    mc_adjusted = mc_probs.copy()
    for cls in range(len(LABELS)):
        mc_adjusted[:, cls] += offsets[cls]

    return calibrated, mc_adjusted


def predict_light(test_df, artifacts):
    """Run light model inference (single LGB binary + multiclass)."""
    feature_cols = artifacts["feature_cols"]
    X_test = test_df[feature_cols].values

    bin_probs = artifacts["light_bin_model"].predict(X_test)
    mc_probs = artifacts["light_mc_model"].predict(X_test)

    return bin_probs, mc_probs


def aggregate_document_predictions(doc_probs, thresh_high, thresh_soft=None):
    """Apply two-threshold document aggregation.

    A document is positive if:
      max(sent_probs) > thresh_high  OR  mean(top_3_probs) > thresh_soft
    """
    max_prob = np.max(doc_probs)
    if thresh_soft is not None:
        sorted_probs = np.sort(doc_probs)[::-1]
        top3_mean = np.mean(sorted_probs[:3]) if len(sorted_probs) >= 3 else max_prob
        return (max_prob > thresh_high) or (top3_mean > thresh_soft)
    return max_prob > thresh_high


def make_binary_labels(sent_probs, doc_thresh, sent_thresh):
    """Given sentence probabilities and thresholds, produce binary labels for one doc."""
    doc_positive = np.max(sent_probs) > doc_thresh
    if doc_positive:
        labels = ["Overgeneration" if p > sent_thresh else "None" for p in sent_probs]
        if all(l == "None" for l in labels):
            labels[int(np.argmax(sent_probs))] = "Overgeneration"
        return labels
    return ["None"] * len(sent_probs)


def make_multiclass_labels(bin_labels, mc_probs):
    """Given binary labels and multiclass probs, assign categories."""
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


def generate_submissions(test_docs, test_df, artifacts):
    """Generate submission files: raw ensemble + meta ensemble + light."""
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    import xgboost as xgb
    feature_cols = artifacts["feature_cols"]
    X_test = test_df[feature_cols].values

    # Raw LGB ensemble (no meta-learner — simpler, may generalize better)
    log.info("Running raw LGB ensemble...")
    raw_bin_probs = np.mean([m.predict(X_test) for m in artifacts["bin_models"]], axis=0)
    raw_mc_probs = np.mean([m.predict(X_test) for m in artifacts["mc_models"]], axis=0)

    # Full meta-learner ensemble
    log.info("Running full meta ensemble...")
    strong_bin_probs, strong_mc_probs = predict_strong(test_df, artifacts)

    # Light
    log.info("Running light model...")
    light_bin_probs, light_mc_probs = predict_light(test_df, artifacts)

    raw_thresh = artifacts["raw_thresh"]
    thresh_high = artifacts["thresh_high"]
    thresh_soft = artifacts["thresh_soft"]
    light_thresh = artifacts["light_thresh"]
    sent_thresh = artifacts["sent_thresh"]

    # Apply per-class offsets to all multiclass probs
    offsets = artifacts["perclass_offsets"]
    for mc in [raw_mc_probs, strong_mc_probs, light_mc_probs]:
        for cls in range(len(LABELS)):
            mc[:, cls] += offsets[cls]

    # Build all submission variants
    all_submissions = {}
    variants = [
        ("raw",    raw_bin_probs,    raw_mc_probs,    raw_thresh, sent_thresh),
        ("strong", strong_bin_probs, strong_mc_probs, thresh_high, sent_thresh),
        ("fast",   light_bin_probs,  light_mc_probs,  light_thresh, light_thresh),
    ]

    for vname, bin_probs, mc_probs, doc_th, sent_th in variants:
        results_21 = []
        results_22 = []
        idx = 0
        for doc in test_docs:
            n = len(doc["sentences"])
            s_bin = bin_probs[idx:idx + n]
            s_mc = mc_probs[idx:idx + n]
            idx += n

            bin_labels = make_binary_labels(s_bin, doc_th, sent_th)
            mc_labels = make_multiclass_labels(bin_labels, s_mc)

            results_21.append({
                "id": doc["id"],
                "labels": bin_labels,
                "run_id": f"writerslogic_Task21b_v5_{vname}",
            })
            results_22.append({
                "id": doc["id"],
                "labels": mc_labels,
                "run_id": f"writerslogic_Task22b_v5_{vname}",
            })

        all_submissions[f"writerslogic_Task21b_v5_{vname}.json"] = results_21
        all_submissions[f"writerslogic_Task22b_v5_{vname}.json"] = results_22

    # Save and zip all submissions
    for filename, results in all_submissions.items():
        json_path = os.path.join(SUBMISSION_DIR, filename)
        zip_path = json_path.replace(".json", ".zip")

        with open(json_path, "w") as f:
            json.dump(results, f)
        subprocess.run(["zip", "-j", zip_path, json_path], check=True)

        total_sents = sum(len(r["labels"]) for r in results)
        non_none = sum(1 for r in results for l in r["labels"] if l != "None")
        pos_docs = sum(1 for r in results if any(l != "None" for l in r["labels"]))
        log.info("  %s: %d/%d non-None sentences, %d/%d positive docs",
                 filename, non_none, total_sents, pos_docs, len(results))

    log.info("\nAll submissions saved to %s/", SUBMISSION_DIR)
    return all_submissions


def run_inference(use_embed=True, use_nli=True):
    """Full inference pipeline."""
    log.info("Loading model artifacts...")
    artifacts = load_artifacts()
    log.info("  Dev doc F1 (raw): %.4f", artifacts.get("dev_raw_doc_f1", 0))
    log.info("  Dev doc F1 (meta): %.4f", artifacts.get("dev_meta_doc_f1", 0))
    log.info("  Thresholds: raw=%.3f, high=%.3f, soft=%s",
             artifacts.get("raw_thresh", 0),
             artifacts["thresh_high"],
             f"{artifacts['thresh_soft']:.3f}" if artifacts["thresh_soft"] else "N/A")

    import gc

    # Load test data
    log.info("Loading test data...")
    with open("test_data.json") as f:
        test_docs = json.load(f)
    log.info("  %d documents, %d sentences",
             len(test_docs), sum(len(d["sentences"]) for d in test_docs))

    # Check for cached test features
    cache_path = os.path.join(MODEL_DIR, "cache", "test_df.pkl")
    if os.path.exists(cache_path):
        log.info("Loading cached test features...")
        import pandas as pd
        test_df = pd.read_pickle(cache_path)
    else:
        # Compute embedding features (load, compute, release)
        test_embed_feats = None
        if use_embed:
            from sentence_transformers import SentenceTransformer
            log.info("Loading embedding model...")
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("Computing embedding features...")
            test_embed_feats = compute_embedding_features(test_docs, embed_model)
            del embed_model
            gc.collect()

        # Compute NLI features (load after embeddings freed)
        test_nli_feats = None
        if use_nli:
            try:
                from sentence_transformers import CrossEncoder
                log.info("Loading NLI model...")
                nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
                log.info("Computing NLI features...")
                test_nli_feats = compute_nli_features(test_docs, nli_model)
                del nli_model
                gc.collect()
            except Exception as e:
                log.warning("Could not load NLI model: %s", e)

        log.info("Building feature DataFrame...")
        test_df = build_dataset(test_docs, test_embed_feats, test_nli_feats)

        # Cache
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        test_df.to_pickle(cache_path)
        log.info("Cached test features to %s", cache_path)

    # Verify feature alignment
    expected_cols = artifacts["feature_cols"]
    missing = [c for c in expected_cols if c not in test_df.columns]
    if missing:
        log.warning("Missing features (will zero-fill): %s", missing)
        for c in missing:
            test_df[c] = 0.0

    # Generate submissions
    generate_submissions(test_docs, test_df, artifacts)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()

    run_inference(use_embed=not args.no_embed, use_nli=not args.no_nli)
