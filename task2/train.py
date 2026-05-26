"""
Training pipeline for SimpleText Task 2 v5.

Trains:
- Full 5-model ensemble (LightGBM binary x5 seeds + LightGBM multiclass x5 seeds + XGBoost binary)
- Lighter 2-model version (LightGBM binary + multiclass, single seed)
- Meta-stacker (LogisticRegression on OOF predictions)
- Isotonic calibration on dev set
- Two-threshold document aggregation parameters

Saves all artifacts to models_v5/ directory.
"""

import json
import logging
import math
import os
import pickle
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

from feature_extraction import (
    LABELS, LABEL2IDX, build_dataset,
    compute_embedding_features, compute_nli_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]
MODEL_DIR = "models_v5"


# ============================================================================
# LightGBM parameters
# ============================================================================

LGB_PARAMS_BIN = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.03, "num_leaves": 127, "max_depth": -1,
    "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "n_jobs": -1,
}

LGB_PARAMS_MC = {
    "objective": "multiclass", "num_class": len(LABELS), "metric": "multi_logloss",
    "learning_rate": 0.03, "num_leaves": 127, "max_depth": -1,
    "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.7,
    "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "n_jobs": -1,
}


# ============================================================================
# Model training functions
# ============================================================================

def train_lgb_binary(X_train, y_train, X_val, y_val, feature_cols, seed=42):
    """Train a single LightGBM binary model."""
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    params = {**LGB_PARAMS_BIN, "seed": seed, "scale_pos_weight": scale_pos_weight}
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)

    model = lgb.train(params, train_data, num_boost_round=4000, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
    return model


def train_lgb_multiclass(X_train, y_train, X_val, y_val, feature_cols, seed=42):
    """Train a single LightGBM multiclass model with sqrt-balanced weights."""
    class_counts = Counter(y_train)
    max_count = max(class_counts.values())
    class_weights = {c: math.sqrt(max_count / count) for c, count in class_counts.items()}
    sample_weights = np.array([class_weights[y] for y in y_train])

    params = {**LGB_PARAMS_MC, "seed": seed}
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                             feature_name=feature_cols)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)

    model = lgb.train(params, train_data, num_boost_round=4000, valid_sets=[val_data],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
    return model


def train_xgb_binary(X_train, y_train, X_val, y_val, seed=42):
    """Train XGBoost binary model for ensemble diversity."""
    import xgboost as xgb

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic", "eval_metric": "logloss",
        "eta": 0.03, "max_depth": 8, "subsample": 0.8, "colsample_bytree": 0.7,
        "min_child_weight": 20, "scale_pos_weight": scale_pos_weight,
        "seed": seed, "verbosity": 0,
    }

    model = xgb.train(params, dtrain, num_boost_round=3000,
                      evals=[(dval, "val")],
                      early_stopping_rounds=150, verbose_eval=500)
    return model


# ============================================================================
# Cross-validation for OOF predictions
# ============================================================================

def generate_oof_predictions(train_df, feature_cols, n_folds=5):
    """Generate out-of-fold predictions using GroupKFold (grouped by doc_id)."""
    X = train_df[feature_cols].values
    y_multi = train_df["label"].values
    y_bin = (y_multi != 0).astype(int)
    groups = train_df["doc_id"].values

    gkf = GroupKFold(n_splits=n_folds)

    oof_bin_probs = np.zeros(len(X))
    oof_mc_probs = np.zeros((len(X), len(LABELS)))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_multi, groups)):
        log.info("  OOF Fold %d/%d...", fold + 1, n_folds)
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr_bin, y_val_bin = y_bin[train_idx], y_bin[val_idx]
        y_tr_mc, y_val_mc = y_multi[train_idx], y_multi[val_idx]

        # Binary
        bin_model = train_lgb_binary(X_tr, y_tr_bin, X_val, y_val_bin, feature_cols,
                                     seed=SEEDS[fold % len(SEEDS)])
        oof_bin_probs[val_idx] = bin_model.predict(X_val)

        # Multiclass
        mc_model = train_lgb_multiclass(X_tr, y_tr_mc, X_val, y_val_mc, feature_cols,
                                        seed=SEEDS[fold % len(SEEDS)])
        oof_mc_probs[val_idx] = mc_model.predict(X_val)

    log.info("  OOF Binary F1 (thresh=0.5): %.4f", f1_score(y_bin, (oof_bin_probs > 0.5).astype(int)))
    log.info("  OOF MC Macro F1: %.4f", f1_score(y_multi, oof_mc_probs.argmax(axis=1), average="macro"))

    return oof_bin_probs, oof_mc_probs


# ============================================================================
# Meta-learner
# ============================================================================

def build_meta_features(bin_probs, mc_probs, df):
    """Build meta-learner input from base model predictions + key features."""
    return np.column_stack([
        bin_probs.reshape(-1, 1),
        mc_probs,
        df["novel_token_ratio"].values,
        df["max_source_sent_overlap"].values,
        df["sent_position_rel"].values,
        df["high_novel_streak"].values,
        df["tail_novel_mean"].values,
        df["prompt_pattern_count"].values,
    ])


def train_meta_learner(meta_X, y_bin):
    """Train logistic regression meta-learner for binary classification."""
    model = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs",
                               class_weight="balanced")
    model.fit(meta_X, y_bin)
    preds = model.predict(meta_X)
    log.info("  Meta-learner train F1: %.4f", f1_score(y_bin, preds))
    return model


# ============================================================================
# Calibration
# ============================================================================

def calibrate_on_dev(train_probs, train_labels, dev_probs, dev_labels):
    """Fit isotonic regression calibration on dev set probabilities."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(dev_probs, dev_labels)

    calibrated_dev = iso.predict(dev_probs)
    log.info("  Calibration: dev probs range [%.3f, %.3f] → [%.3f, %.3f]",
             dev_probs.min(), dev_probs.max(), calibrated_dev.min(), calibrated_dev.max())
    return iso


# ============================================================================
# Two-threshold document aggregation optimization
# ============================================================================

def optimize_two_threshold(sent_probs, y_sent_bin, doc_ids):
    """Optimize two-threshold document aggregation:
    A document is positive if:
      max(sent_probs) > thresh_high  OR  mean_top_k(sent_probs) > thresh_soft

    Returns (thresh_high, thresh_soft, best_k).
    """
    unique_docs = np.unique(doc_ids)

    # Pre-compute per-document max and top-k means
    doc_info = {}
    for doc_id in unique_docs:
        mask = doc_ids == doc_id
        probs = sent_probs[mask]
        true_labels = y_sent_bin[mask]
        doc_true = int(any(true_labels == 1))
        sorted_probs = np.sort(probs)[::-1]
        doc_info[doc_id] = {
            "max": sorted_probs[0],
            "top3_mean": np.mean(sorted_probs[:3]) if len(sorted_probs) >= 3 else sorted_probs[0],
            "top2_mean": np.mean(sorted_probs[:2]) if len(sorted_probs) >= 2 else sorted_probs[0],
            "true": doc_true,
        }

    doc_trues = np.array([doc_info[d]["true"] for d in unique_docs])
    doc_maxes = np.array([doc_info[d]["max"] for d in unique_docs])
    doc_top3 = np.array([doc_info[d]["top3_mean"] for d in unique_docs])

    n_pos = doc_trues.sum()

    def fast_f1(preds, trues, n_pos):
        tp = (preds & trues).sum()
        fp = (preds & ~trues).sum()
        fn = n_pos - tp
        if tp == 0:
            return 0.0
        return 2 * tp / (2 * tp + fp + fn)

    trues_bool = doc_trues.astype(bool)

    best_f1 = 0
    best_thresh_high = 0.5
    best_thresh_soft = 0.5

    for th in np.arange(0.15, 0.75, 0.01):
        mask_high = doc_maxes > th
        for ts in np.arange(0.10, 0.60, 0.01):
            preds = mask_high | (doc_top3 > ts)
            f1 = fast_f1(preds, trues_bool, n_pos)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh_high = th
                best_thresh_soft = ts

    best_single_f1 = 0
    best_single_thresh = 0.5
    for th in np.arange(0.15, 0.75, 0.005):
        preds = doc_maxes > th
        f1 = fast_f1(preds, trues_bool, n_pos)
        if f1 > best_single_f1:
            best_single_f1 = f1
            best_single_thresh = th

    log.info("  Single threshold: %.3f → doc F1=%.4f", best_single_thresh, best_single_f1)
    log.info("  Two-threshold: high=%.3f, soft=%.3f → doc F1=%.4f",
             best_thresh_high, best_thresh_soft, best_f1)

    if best_f1 > best_single_f1:
        return best_thresh_high, best_thresh_soft, best_f1
    else:
        return best_single_thresh, None, best_single_f1


# ============================================================================
# Per-class threshold optimization for multiclass
# ============================================================================

def optimize_perclass_offsets(probs, y_true):
    """Greedy per-class offset optimization to maximize macro F1."""
    n_classes = probs.shape[1]
    offsets = np.zeros(n_classes)
    best_macro_f1 = f1_score(y_true, probs.argmax(axis=1), average="macro")
    log.info("  Baseline argmax macro F1: %.4f", best_macro_f1)

    for cls in range(1, n_classes):
        best_offset = 0.0
        for offset in np.arange(-0.15, 0.30, 0.005):
            adjusted = probs.copy()
            adjusted[:, cls] += offset
            for c2 in range(1, cls):
                adjusted[:, c2] += offsets[c2]
            preds = adjusted.argmax(axis=1)
            mf1 = f1_score(y_true, preds, average="macro")
            if mf1 > best_macro_f1:
                best_macro_f1 = mf1
                best_offset = offset
        offsets[cls] = best_offset
        if best_offset != 0:
            log.info("    Class %d (%s): offset=%.3f", cls, LABELS[cls], best_offset)

    adjusted = probs.copy()
    for cls in range(n_classes):
        adjusted[:, cls] += offsets[cls]
    final_f1 = f1_score(y_true, adjusted.argmax(axis=1), average="macro")
    log.info("  Per-class tuned macro F1: %.4f", final_f1)

    return offsets


# ============================================================================
# Main training pipeline
# ============================================================================

def run_training(use_embed=True, use_nli=True, use_cache=True):
    """Run full training pipeline. Returns all model artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    cache_dir = os.path.join(MODEL_DIR, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    train_cache = os.path.join(cache_dir, "train_df.pkl")
    dev_cache = os.path.join(cache_dir, "dev_df.pkl")

    if use_cache and os.path.exists(train_cache) and os.path.exists(dev_cache):
        log.info("Loading cached feature DataFrames...")
        train_df = pd.read_pickle(train_cache)
        dev_df = pd.read_pickle(dev_cache)
    else:
        # Load data
        log.info("Loading data...")
        with open("train_data.json") as f:
            train_docs = json.load(f)
        with open("dev_data.json") as f:
            dev_docs = json.load(f)

        # Compute embedding features (load model, compute, then release to free RAM)
        import gc
        train_embed_feats, dev_embed_feats = None, None
        if use_embed:
            from sentence_transformers import SentenceTransformer
            log.info("Loading embedding model (all-MiniLM-L6-v2)...")
            embed_model = SentenceTransformer("all-MiniLM-L6-v2")
            log.info("Computing embedding features (train)...")
            train_embed_feats = compute_embedding_features(train_docs, embed_model)
            log.info("Computing embedding features (dev)...")
            dev_embed_feats = compute_embedding_features(dev_docs, embed_model)
            del embed_model
            gc.collect()

        # Compute NLI features (load model after embeddings are freed)
        train_nli_feats, dev_nli_feats = None, None
        if use_nli:
            try:
                from sentence_transformers import CrossEncoder
                log.info("Loading NLI cross-encoder...")
                nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
                log.info("Computing NLI features (train)...")
                train_nli_feats = compute_nli_features(train_docs, nli_model)
                log.info("Computing NLI features (dev)...")
                dev_nli_feats = compute_nli_features(dev_docs, nli_model)
                del nli_model
                gc.collect()
            except Exception as e:
                log.warning("Could not load NLI model: %s", e)
                use_nli = False

        # Build DataFrames
        log.info("Building feature DataFrames...")
        train_df = build_dataset(train_docs, train_embed_feats, train_nli_feats)
        dev_df = build_dataset(dev_docs, dev_embed_feats, dev_nli_feats)

        # Cache to disk
        log.info("Caching feature DataFrames to %s/...", cache_dir)
        train_df.to_pickle(train_cache)
        dev_df.to_pickle(dev_cache)

    feature_cols = [c for c in train_df.columns if c not in ["doc_id", "label"]]
    log.info("Feature count: %d", len(feature_cols))
    log.info("Train shape: %s, Dev shape: %s", train_df.shape, dev_df.shape)

    X_train = train_df[feature_cols].values
    y_train_multi = train_df["label"].values
    y_train_bin = (y_train_multi != 0).astype(int)

    X_dev = dev_df[feature_cols].values
    y_dev_multi = dev_df["label"].values
    y_dev_bin = (y_dev_multi != 0).astype(int)

    # ========================================================================
    # 1. Generate OOF predictions on train for meta-stacker
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("  GENERATING OOF PREDICTIONS (5-fold)")
    log.info("=" * 60)

    oof_bin_probs, oof_mc_probs = generate_oof_predictions(train_df, feature_cols)

    # ========================================================================
    # 2. Train meta-learner on OOF
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("  TRAINING META-LEARNER")
    log.info("=" * 60)

    meta_X_train = build_meta_features(oof_bin_probs, oof_mc_probs, train_df)
    meta_model = train_meta_learner(meta_X_train, y_train_bin)

    # ========================================================================
    # 3. Train final models on train+dev combined (more data = better)
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("  TRAINING FINAL MODELS (on train+dev combined)")
    log.info("=" * 60)

    full_df = pd.concat([train_df, dev_df], ignore_index=True)
    X_full = full_df[feature_cols].values
    y_full_multi = full_df["label"].values
    y_full_bin = (y_full_multi != 0).astype(int)

    # Use last 10% of combined data as validation for early stopping
    n_val = len(dev_df)
    X_train_final = X_full[:-n_val]
    y_train_bin_final = y_full_bin[:-n_val]
    y_train_mc_final = y_full_multi[:-n_val]
    X_val_final = X_full[-n_val:]
    y_val_bin_final = y_full_bin[-n_val:]
    y_val_mc_final = y_full_multi[-n_val:]

    import xgboost as xgb

    log.info("\nTraining full ensemble (5 seeds binary + 5 seeds multiclass + XGBoost)...")
    bin_models = []
    for seed in SEEDS:
        log.info("  Binary seed=%d", seed)
        m = train_lgb_binary(X_train_final, y_train_bin_final,
                             X_val_final, y_val_bin_final, feature_cols, seed=seed)
        bin_models.append(m)

    mc_models = []
    for seed in SEEDS:
        log.info("  Multiclass seed=%d", seed)
        m = train_lgb_multiclass(X_train_final, y_train_mc_final,
                                 X_val_final, y_val_mc_final, feature_cols, seed=seed)
        mc_models.append(m)

    log.info("  XGBoost binary...")
    xgb_model = train_xgb_binary(X_train_final, y_train_bin_final,
                                 X_val_final, y_val_bin_final, seed=42)

    light_bin_model = bin_models[0]
    light_mc_model = mc_models[0]

    # ========================================================================
    # 4. Optimize thresholds on OOF predictions (21k docs, robust)
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("  THRESHOLD OPTIMIZATION ON OOF (train set)")
    log.info("=" * 60)

    train_doc_ids = train_df["doc_id"].values
    unique_train_docs = np.unique(train_doc_ids)

    # --- Raw ensemble threshold (no meta-learner, simpler = more robust) ---
    log.info("\nOptimizing raw ensemble threshold on OOF...")
    oof_doc_maxes = np.array([oof_bin_probs[train_doc_ids == d].max() for d in unique_train_docs])
    oof_doc_trues = np.array([int(any(y_train_bin[train_doc_ids == d] == 1)) for d in unique_train_docs])
    n_pos = oof_doc_trues.sum()
    trues_bool = oof_doc_trues.astype(bool)

    def fast_f1_arr(preds_bool, trues_bool, n_pos):
        tp = (preds_bool & trues_bool).sum()
        fp = (preds_bool & ~trues_bool).sum()
        fn = n_pos - tp
        if tp == 0:
            return 0.0
        return 2 * tp / (2 * tp + fp + fn)

    best_raw_f1 = 0
    best_raw_thresh = 0.5
    for th in np.arange(0.20, 0.80, 0.005):
        preds = oof_doc_maxes > th
        f1 = fast_f1_arr(preds, trues_bool, n_pos)
        if f1 > best_raw_f1:
            best_raw_f1 = f1
            best_raw_thresh = th
    log.info("  Raw ensemble: thresh=%.3f → OOF doc F1=%.4f", best_raw_thresh, best_raw_f1)

    # --- Meta-learner threshold ---
    log.info("\nOptimizing meta-learner threshold on OOF...")
    oof_meta_probs = meta_model.predict_proba(meta_X_train)[:, 1]
    thresh_high, thresh_soft, oof_meta_doc_f1 = optimize_two_threshold(
        oof_meta_probs, y_train_bin, train_doc_ids)
    log.info("  Meta two-threshold: OOF doc F1=%.4f", oof_meta_doc_f1)

    # Sentence threshold
    best_sent_f1 = 0
    best_sent_thresh = 0.5
    for th in np.arange(0.20, 0.80, 0.005):
        sf1 = f1_score(y_train_bin, (oof_bin_probs > th).astype(int))
        if sf1 > best_sent_f1:
            best_sent_f1 = sf1
            best_sent_thresh = th
    log.info("  OOF sentence F1: %.4f (thresh=%.3f)", best_sent_f1, best_sent_thresh)

    # --- Per-class offsets (cap rare classes to avoid overfitting) ---
    log.info("\nOptimizing per-class offsets on OOF...")
    perclass_offsets = optimize_perclass_offsets(oof_mc_probs, y_train_multi)
    # Cap GROUNDED_OVERGENERATION offset (only 100 train examples, easy to overfit)
    if abs(perclass_offsets[5]) > 0.10:
        log.info("  Capping GROUNDED_OVERGENERATION offset from %.3f to 0.10", perclass_offsets[5])
        perclass_offsets[5] = min(perclass_offsets[5], 0.10)

    # ========================================================================
    # 5. Validate on dev (no tuning, just reporting)
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("  DEV VALIDATION (no tuning)")
    log.info("=" * 60)

    dev_bin_probs_full = np.mean([m.predict(X_dev) for m in bin_models], axis=0)
    dev_mc_probs_full = np.mean([m.predict(X_dev) for m in mc_models], axis=0)
    dev_xgb_probs = xgb_model.predict(xgb.DMatrix(X_dev))
    dev_bin_probs_blended = 0.7 * dev_bin_probs_full + 0.3 * dev_xgb_probs

    dev_doc_ids = dev_df["doc_id"].values
    unique_dev_docs = np.unique(dev_doc_ids)

    # Raw ensemble doc F1 on dev
    dev_doc_maxes_raw = np.array([dev_bin_probs_full[dev_doc_ids == d].max() for d in unique_dev_docs])
    dev_doc_trues = np.array([int(any(y_dev_bin[dev_doc_ids == d] == 1)) for d in unique_dev_docs])
    dev_raw_preds = (dev_doc_maxes_raw > best_raw_thresh).astype(int)
    dev_raw_doc_f1 = f1_score(dev_doc_trues, dev_raw_preds)
    log.info("  Dev doc F1 (raw ensemble, thresh=%.3f): %.4f", best_raw_thresh, dev_raw_doc_f1)

    # Meta doc F1 on dev
    dev_meta_X = build_meta_features(dev_bin_probs_blended, dev_mc_probs_full, dev_df)
    dev_meta_probs = meta_model.predict_proba(dev_meta_X)[:, 1]
    dev_meta_maxes = np.array([dev_meta_probs[dev_doc_ids == d].max() for d in unique_dev_docs])
    dev_meta_top3 = np.array([
        np.mean(np.sort(dev_meta_probs[dev_doc_ids == d])[::-1][:3])
        if sum(dev_doc_ids == d) >= 3
        else dev_meta_probs[dev_doc_ids == d].max()
        for d in unique_dev_docs
    ])
    if thresh_soft is not None:
        dev_meta_preds = ((dev_meta_maxes > thresh_high) | (dev_meta_top3 > thresh_soft)).astype(int)
    else:
        dev_meta_preds = (dev_meta_maxes > thresh_high).astype(int)
    dev_meta_doc_f1 = f1_score(dev_doc_trues, dev_meta_preds)
    log.info("  Dev doc F1 (meta two-thresh): %.4f", dev_meta_doc_f1)

    dev_sent_f1 = f1_score(y_dev_bin, (dev_bin_probs_full > best_sent_thresh).astype(int))
    log.info("  Dev sent F1 (thresh=%.3f): %.4f", best_sent_thresh, dev_sent_f1)

    # Multiclass on dev
    adjusted_mc = dev_mc_probs_full.copy()
    for cls in range(len(LABELS)):
        adjusted_mc[:, cls] += perclass_offsets[cls]
    dev_mc_preds = adjusted_mc.argmax(axis=1)
    log.info("  Dev MC Macro F1: %.4f", f1_score(y_dev_multi, dev_mc_preds, average="macro"))
    log.info("\n%s", classification_report(y_dev_multi, dev_mc_preds, target_names=LABELS, zero_division=0))

    # ========================================================================
    # 6. Light ensemble
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("  LIGHT ENSEMBLE EVALUATION")
    log.info("=" * 60)

    light_doc_maxes = np.array([oof_bin_probs[train_doc_ids == d].max() for d in unique_train_docs])
    best_light_doc_f1 = 0
    best_light_thresh = 0.5
    for th in np.arange(0.20, 0.80, 0.005):
        preds = light_doc_maxes > th
        f1 = fast_f1_arr(preds, trues_bool, n_pos)
        if f1 > best_light_doc_f1:
            best_light_doc_f1 = f1
            best_light_thresh = th
    log.info("  Light model OOF doc F1: %.4f (thresh=%.3f)", best_light_doc_f1, best_light_thresh)

    # ========================================================================
    # 7. Save all artifacts
    # ========================================================================
    log.info("\nSaving model artifacts to %s/...", MODEL_DIR)

    artifacts = {
        "feature_cols": feature_cols,
        "bin_models": bin_models,
        "mc_models": mc_models,
        "xgb_model": xgb_model,
        "meta_model": meta_model,
        "calibrator": None,
        "thresh_high": thresh_high,
        "thresh_soft": thresh_soft,
        "raw_thresh": best_raw_thresh,
        "sent_thresh": best_sent_thresh,
        "perclass_offsets": perclass_offsets,
        "light_bin_model": light_bin_model,
        "light_mc_model": light_mc_model,
        "light_thresh": best_light_thresh,
        "dev_doc_f1": max(dev_raw_doc_f1, dev_meta_doc_f1),
        "dev_raw_doc_f1": dev_raw_doc_f1,
        "dev_meta_doc_f1": dev_meta_doc_f1,
        "dev_sent_f1": dev_sent_f1,
        "light_doc_f1": best_light_doc_f1,
        "blend_weights": {"lgb": 0.7, "xgb": 0.3},
    }

    with open(os.path.join(MODEL_DIR, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)

    # Save feature importance
    importance = bin_models[0].feature_importance(importance_type="gain")
    top_feats = sorted(zip(feature_cols, importance), key=lambda x: -x[1])[:30]
    log.info("\nTop 30 features (binary, gain):")
    for name, imp in top_feats:
        log.info("  %-45s %10.0f", name, imp)

    log.info("\n" + "=" * 60)
    log.info("  TRAINING COMPLETE")
    log.info("=" * 60)
    log.info("  Raw ensemble dev doc F1: %.4f (thresh=%.3f)", dev_raw_doc_f1, best_raw_thresh)
    log.info("  Meta ensemble dev doc F1: %.4f (high=%.3f, soft=%s)",
             dev_meta_doc_f1, thresh_high, f"{thresh_soft:.3f}" if thresh_soft else "N/A")
    log.info("  Light model OOF doc F1: %.4f (thresh=%.3f)", best_light_doc_f1, best_light_thresh)
    log.info("  Sent thresh: %.3f", best_sent_thresh)

    return artifacts


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--no-embed", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    run_training(use_embed=not args.no_embed, use_nli=not args.no_nli,
                 use_cache=not args.no_cache)
