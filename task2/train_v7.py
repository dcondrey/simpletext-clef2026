"""Train v7 model with expanded feature set (109 features).

Uses v3-style simple aggregation (no meta-learner, no XGBoost blend).
5-seed LightGBM binary + multiclass ensembles.
CV-based threshold optimization.
"""

import json
import logging
import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

from feature_extraction import (
    LABELS, LABEL2IDX, build_dataset,
    compute_embedding_features, compute_nli_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SEEDS = [42, 123, 456, 789, 1024]
MODEL_DIR = "models_v7"
CACHE_DIR = os.path.join(MODEL_DIR, "cache")

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


def load_or_build_features(split_name, docs, use_embed=True, use_nli=True):
    cache_path = os.path.join(CACHE_DIR, f"{split_name}_df.pkl")
    if os.path.exists(cache_path):
        log.info("Loading cached %s features...", split_name)
        return pd.read_pickle(cache_path)

    import gc

    embed_feats = None
    if use_embed:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        log.info("Computing %s embedding features...", split_name)
        embed_feats = compute_embedding_features(docs, embed_model)
        del embed_model
        gc.collect()

    nli_feats = None
    if use_nli:
        try:
            from sentence_transformers import CrossEncoder
            log.info("Loading NLI model (deberta-v3-base)...")
            nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
            log.info("Computing %s NLI features...", split_name)
            nli_feats = compute_nli_features(docs, nli_model)
            del nli_model
            gc.collect()
        except Exception as e:
            log.warning("NLI failed: %s", e)

    log.info("Building %s feature DataFrame...", split_name)
    df = build_dataset(docs, embed_feats, nli_feats)

    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_pickle(cache_path)
    log.info("Cached %s features to %s (%d rows, %d cols)", split_name, cache_path, len(df), len(df.columns))
    return df


def add_llm_features(df, docs, split_name):
    llm_path = os.path.join("cache/llm_grounding", f"{split_name}.json")
    if not os.path.exists(llm_path):
        log.info("No LLM grounding cache for %s, skipping", split_name)
        df["llm_gpt4o_halluc"] = 0.0
        df["llm_deepseek_halluc"] = 0.0
        df["llm_agreement"] = 0.0
        return df

    llm_cache = json.load(open(llm_path))
    log.info("Loaded %d LLM grounding results for %s", len(llm_cache), split_name)

    gpt_scores = []
    ds_scores = []
    for d in docs:
        for i in range(len(d["sentences"])):
            key = f"{d['id']}_{i}"
            entry = llm_cache.get(key, {})
            gpt_scores.append(entry.get("gpt4o_mini", 0.0) or 0.0)
            ds_scores.append(entry.get("deepseek", 0.0) or 0.0)

    df["llm_gpt4o_halluc"] = gpt_scores[:len(df)]
    df["llm_deepseek_halluc"] = ds_scores[:len(df)]
    df["llm_agreement"] = (np.array(gpt_scores[:len(df)]) + np.array(ds_scores[:len(df)])) / 2.0
    return df


def train_binary_ensemble(X_train, y_train, X_val, y_val, feature_cols):
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    models = []
    for seed in SEEDS:
        params = {**LGB_PARAMS_BIN, "seed": seed, "scale_pos_weight": scale_pos_weight}
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=4000, valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
        models.append(model)
        log.info("  Seed %d: %d rounds", seed, model.best_iteration)

    val_probs = np.mean([m.predict(X_val) for m in models], axis=0)
    return models, val_probs


def train_multiclass_ensemble(X_train, y_train, X_val, y_val, feature_cols):
    class_counts = np.bincount(y_train, minlength=len(LABELS))
    weights = np.sqrt(class_counts.max() / np.maximum(class_counts, 1))

    sample_weights_train = np.array([weights[int(y)] for y in y_train])

    models = []
    for seed in SEEDS:
        params = {**LGB_PARAMS_MC, "seed": seed}
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=4000, valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
        models.append(model)
        log.info("  Seed %d: %d rounds", seed, model.best_iteration)

    val_probs = np.mean([m.predict(X_val) for m in models], axis=0)
    return models, val_probs


def optimize_thresholds(bin_probs, y_true, doc_ids):
    y_bin = (y_true > 0).astype(int)

    best_sent_f1, best_sent_t = 0, 0.5
    best_doc_f1, best_doc_t = 0, 0.5

    for t in np.arange(0.10, 0.90, 0.005):
        preds = (bin_probs > t).astype(int)
        sf1 = f1_score(y_bin, preds)
        if sf1 > best_sent_f1:
            best_sent_f1, best_sent_t = sf1, t

        doc_true, doc_pred = {}, {}
        for doc_id, pred, true in zip(doc_ids, preds, y_bin):
            doc_true.setdefault(doc_id, 0)
            doc_pred.setdefault(doc_id, 0)
            if true: doc_true[doc_id] = 1
            if pred: doc_pred[doc_id] = 1
        df1 = f1_score(list(doc_true.values()), list(doc_pred.values()))
        if df1 > best_doc_f1:
            best_doc_f1, best_doc_t = df1, t

    log.info("  Best sentence threshold: %.3f (F1=%.4f)", best_sent_t, best_sent_f1)
    log.info("  Best doc threshold: %.3f (F1=%.4f)", best_doc_t, best_doc_f1)
    return best_sent_t, best_doc_t, best_sent_f1, best_doc_f1


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-nli", action="store_true")
    parser.add_argument("--no-embed", action="store_true")
    args = parser.parse_args()

    log.info("Loading data...")
    train_docs = json.load(open("train_data.json"))
    dev_docs = json.load(open("dev_data.json"))

    train_df = load_or_build_features("train", train_docs, not args.no_embed, not args.no_nli)
    dev_df = load_or_build_features("dev", dev_docs, not args.no_embed, not args.no_nli)

    train_df = add_llm_features(train_df, train_docs, "train")
    dev_df = add_llm_features(dev_df, dev_docs, "dev")

    feature_cols = [c for c in train_df.columns if c not in ["doc_id", "label"]]
    log.info("Features: %d", len(feature_cols))
    log.info("Train: %s, Dev: %s", train_df.shape, dev_df.shape)

    X_train = train_df[feature_cols].values
    X_dev = dev_df[feature_cols].values
    y_train = train_df["label"].values
    y_dev = dev_df["label"].values
    y_train_bin = (y_train > 0).astype(int)
    y_dev_bin = (y_dev > 0).astype(int)

    log.info("\n=== Training Binary Ensemble ===")
    bin_models, dev_bin_probs = train_binary_ensemble(X_train, y_train_bin, X_dev, y_dev_bin, feature_cols)

    log.info("\n=== Training Multiclass Ensemble ===")
    mc_models, dev_mc_probs = train_multiclass_ensemble(X_train, y_train, X_dev, y_dev, feature_cols)

    log.info("\n=== Optimizing Thresholds ===")
    doc_ids = dev_df["doc_id"].values
    sent_t, doc_t, sent_f1, doc_f1 = optimize_thresholds(dev_bin_probs, y_dev, doc_ids)

    log.info("\n=== Feature Importance (top 20) ===")
    imp = bin_models[0].feature_importance(importance_type="gain")
    for name, score in sorted(zip(feature_cols, imp), key=lambda x: -x[1])[:20]:
        log.info("  %-45s %10.0f", name, score)

    log.info("\n=== Multiclass Dev Report ===")
    mc_preds = dev_mc_probs.argmax(axis=1)
    log.info("\n%s", classification_report(y_dev, mc_preds, target_names=LABELS, digits=3))

    os.makedirs(MODEL_DIR, exist_ok=True)
    artifacts = {
        "bin_models": bin_models,
        "mc_models": mc_models,
        "feature_cols": feature_cols,
        "sent_thresh": sent_t,
        "doc_thresh": doc_t,
        "dev_sent_f1": sent_f1,
        "dev_doc_f1": doc_f1,
    }
    with open(os.path.join(MODEL_DIR, "artifacts.pkl"), "wb") as f:
        pickle.dump(artifacts, f)
    log.info("Saved artifacts to %s/artifacts.pkl", MODEL_DIR)
    log.info("Dev sent F1: %.4f, Dev doc F1: %.4f", sent_f1, doc_f1)


if __name__ == "__main__":
    main()
