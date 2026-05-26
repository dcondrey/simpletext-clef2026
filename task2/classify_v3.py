"""
SimpleText Task 2 v3: Improved hallucination identification & classification.

Improvements over v2:
- NLI cross-encoder scoring (source vs sentence entailment)
- Contextual window features (surrounding sentences)
- Document-level aggregated features
- Multi-seed LightGBM ensemble (5 seeds)
- Separate optimized binary model for identification (not just thresholding)
- Better class imbalance handling (sqrt-balanced weights + focal-like boosting)
- Additional text quality features (perplexity proxy, coherence)
- Generates both Task 2.1 (identification) and Task 2.2 (classification) submissions

Usage:
    uv run python classify_v3.py
    uv run python classify_v3.py --no-nli   # skip NLI (faster, CPU-only)
"""

import json
import logging
import os
import re
import subprocess
import sys
import zlib
import math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, classification_report
import lightgbm as lgb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LABELS = ["None", "LEAKED_INSTRUCTIONS", "UNGROUNDED_INJECTION",
          "GENERATION_FAILURE", "REPETITIVE_CONTENT", "GROUNDED_OVERGENERATION"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}

SEEDS = [42, 123, 456, 789, 1024]


def load_data(path):
    with open(path) as f:
        return json.load(f)


# ============================================================================
# Feature engineering
# ============================================================================

def compression_ratio(text):
    if not text or len(text) < 5:
        return 1.0
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / max(len(raw), 1)


def char_entropy(text):
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def extract_features(doc):
    """Extract features for all sentences in a document."""
    source = doc["source"]
    sentences = doc["sentences"]
    source_lower = source.lower()
    source_tokens = source_lower.split()
    source_set = set(source_tokens)
    n_sents = len(sentences)

    source_bigrams = set(zip(source_tokens[:-1], source_tokens[1:])) if len(source_tokens) > 1 else set()
    source_trigrams = set(zip(source_tokens[:-2], source_tokens[1:-1], source_tokens[2:])) if len(source_tokens) > 2 else set()

    # Document-level stats
    all_sent_lens = [len(s.split()) for s in sentences]
    doc_mean_len = np.mean(all_sent_lens) if all_sent_lens else 0
    doc_std_len = np.std(all_sent_lens) if len(all_sent_lens) > 1 else 0
    doc_total_words = sum(all_sent_lens)

    features_list = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        sent_tokens = sent_lower.split()
        sent_set = set(sent_tokens)
        f = {}

        # --- Basic text features ---
        f["sent_len_chars"] = len(sent)
        f["sent_len_words"] = len(sent_tokens)
        f["sent_position"] = i
        f["sent_position_rel"] = i / max(n_sents - 1, 1)
        f["is_first_sent"] = int(i == 0)
        f["is_last_sent"] = int(i == n_sents - 1)
        f["is_last_3"] = int(i >= n_sents - 3)
        f["is_last_5"] = int(i >= n_sents - 5)
        f["is_first_3"] = int(i < 3)
        f["n_sentences_in_doc"] = n_sents

        # --- Source alignment features ---
        if source_set:
            overlap = sent_set & source_set
            f["token_overlap_ratio"] = len(overlap) / max(len(sent_set), 1)
            f["novel_token_ratio"] = 1.0 - f["token_overlap_ratio"]
            f["novel_token_count"] = len(sent_set - source_set)
            f["source_coverage"] = len(overlap) / max(len(source_set), 1)
        else:
            f["token_overlap_ratio"] = 0
            f["novel_token_ratio"] = 1
            f["novel_token_count"] = len(sent_set)
            f["source_coverage"] = 0

        sent_bigrams = set(zip(sent_tokens[:-1], sent_tokens[1:])) if len(sent_tokens) > 1 else set()
        f["bigram_overlap_with_source"] = len(sent_bigrams & source_bigrams) / max(len(sent_bigrams), 1) if sent_bigrams else 0

        sent_trigrams = set(zip(sent_tokens[:-2], sent_tokens[1:-1], sent_tokens[2:])) if len(sent_tokens) > 2 else set()
        f["trigram_overlap_with_source"] = len(sent_trigrams & source_trigrams) / max(len(sent_trigrams), 1) if sent_trigrams else 0

        f["source_len_words"] = len(source_set)
        f["sent_to_source_ratio"] = len(sent_tokens) / max(len(source_tokens), 1)

        # Longest contiguous source match
        longest_match = 0
        for start in range(len(sent_tokens)):
            for end in range(start + 1, min(start + 25, len(sent_tokens) + 1)):
                substr = " ".join(sent_tokens[start:end])
                if substr in source_lower:
                    longest_match = max(longest_match, end - start)
                else:
                    break
        f["longest_source_match"] = longest_match
        f["source_match_ratio"] = longest_match / max(len(sent_tokens), 1)

        # --- Prompt leakage / instruction patterns ---
        prompt_patterns = [
            r"\bsimplif", r"\bhere\s+is", r"\bhere\s+are", r"\bI\s+(have|will|can|did)",
            r"\blet\s+me", r"\bplease\s+note", r"\bnote\s*:", r"\bin\s+summary",
            r"\boverall\b", r"\bin\s+conclusion", r"\bkey\s+(takeaway|point|finding)",
            r"\bimportant(ly)?\b", r"\bremember\b", r"\bhope\s+this\s+helps",
            r"\bInput\s*:", r"\bOutput\s*:", r"\bSource\s*:", r"\bTarget\s*:",
            r"\bStep\s+\d", r"\bAssistant\s*:", r"\bUser\s*:",
            r"^\d+\.\s+", r"^-\s+", r"^\*\s+",
            r"\bin\s+simple\s+terms", r"\bto\s+put\s+it\s+simply",
            r"\bthis\s+means\b", r"\bbasically\b",
            r"\bsure[,!]", r"\bof\s+course",
            r"\bI'm\s+happy\s+to", r"\bI'd\s+be\s+happy",
            r"\bAs\s+an\s+AI", r"\bAs\s+a\s+language\s+model",
            r"\bI\s+understand\b", r"\bGreat\s+question",
            r"\bAbsolutely[,!]", r"\bCertainly[,!]",
        ]
        f["prompt_pattern_count"] = sum(1 for p in prompt_patterns if re.search(p, sent, re.IGNORECASE))
        f["has_prompt_marker"] = int(f["prompt_pattern_count"] > 0)

        # Strong instruction leak patterns
        strong_leak_patterns = [
            r"\bInput\s*:", r"\bOutput\s*:", r"\bSource\s*:", r"\bTarget\s*:",
            r"\bAssistant\s*:", r"\bUser\s*:", r"\bSystem\s*:",
            r"\bInstruction\s*:", r"\bPrompt\s*:", r"\bTask\s*:",
            r"\bAs\s+an\s+AI\b", r"\bAs\s+a\s+language\s+model\b",
            r"\bI'm\s+an\s+AI\b", r"\bI\s+am\s+an?\s+AI\b",
        ]
        f["strong_leak_count"] = sum(1 for p in strong_leak_patterns if re.search(p, sent, re.IGNORECASE))

        # --- Structural features ---
        f["starts_with_number"] = int(bool(re.match(r"^\d+[\.\)]\s", sent)))
        f["starts_with_bullet"] = int(bool(re.match(r"^[-\*•]\s", sent)))
        f["has_colon"] = int(":" in sent)
        f["has_quotes"] = int("'" in sent or '"' in sent)
        f["n_special_chars"] = sum(1 for c in sent if c in "[]{}|\\<>@#$%^&*")
        f["exclamation_count"] = sent.count("!")
        f["question_count"] = sent.count("?")
        f["uppercase_ratio"] = sum(1 for c in sent if c.isupper()) / max(len(sent), 1)
        f["comma_count"] = sent.count(",")
        f["starts_with_lowercase"] = int(sent[0].islower() if sent else False)
        f["ends_with_period"] = int(sent.rstrip().endswith("."))
        f["ends_with_colon"] = int(sent.rstrip().endswith(":"))
        f["has_parenthetical"] = int(bool(re.search(r"\(.*?\)", sent)))
        f["contains_bracket_content"] = int(bool(re.search(r"\[.*?\]", sent)))

        # --- Text quality features ---
        f["avg_word_len"] = np.mean([len(w) for w in sent_tokens]) if sent_tokens else 0
        f["max_word_len"] = max((len(w) for w in sent_tokens), default=0)
        f["digit_ratio"] = sum(1 for c in sent if c.isdigit()) / max(len(sent), 1)
        f["compression_ratio"] = compression_ratio(sent)
        f["char_entropy"] = char_entropy(sent)

        # Vocabulary richness
        f["type_token_ratio"] = len(sent_set) / max(len(sent_tokens), 1)

        # --- Repetition features ---
        if i > 0:
            prev = sentences[i - 1].lower()
            prev_tokens = set(prev.split())
            f["overlap_with_prev"] = len(sent_set & prev_tokens) / max(len(sent_set), 1)
            f["exact_match_prev"] = int(sent_lower == prev)
        else:
            f["overlap_with_prev"] = 0
            f["exact_match_prev"] = 0

        if i > 1:
            f["overlap_with_prev2"] = len(sent_set & set(sentences[i - 2].lower().split())) / max(len(sent_set), 1)
        else:
            f["overlap_with_prev2"] = 0

        # Check for repetition with ANY earlier sentence
        f["max_overlap_with_any_prev"] = 0
        f["exact_match_any_prev"] = 0
        for j in range(max(0, i - 10), i):
            prev_j_set = set(sentences[j].lower().split())
            ov = len(sent_set & prev_j_set) / max(len(sent_set), 1)
            f["max_overlap_with_any_prev"] = max(f["max_overlap_with_any_prev"], ov)
            if sent_lower == sentences[j].lower():
                f["exact_match_any_prev"] = 1

        if len(sent_tokens) > 2:
            bg = list(zip(sent_tokens[:-1], sent_tokens[1:]))
            f["bigram_repetition"] = 1.0 - len(set(bg)) / max(len(bg), 1)
        else:
            f["bigram_repetition"] = 0

        if len(sent_tokens) > 3:
            tg = list(zip(sent_tokens[:-2], sent_tokens[1:-1], sent_tokens[2:]))
            f["trigram_repetition"] = 1.0 - len(set(tg)) / max(len(tg), 1)
        else:
            f["trigram_repetition"] = 0

        rep_chars = sum(1 for j in range(1, len(sent)) if sent[j] == sent[j - 1])
        f["repeated_char_ratio"] = rep_chars / max(len(sent), 1)

        # --- Document-level context features ---
        f["sent_len_vs_doc_mean"] = len(sent_tokens) / max(doc_mean_len, 1)
        f["sent_len_zscore"] = (len(sent_tokens) - doc_mean_len) / max(doc_std_len, 1) if doc_std_len > 0 else 0
        f["sent_word_frac_of_doc"] = len(sent_tokens) / max(doc_total_words, 1)

        # Remaining sentences after this one
        f["remaining_sents"] = n_sents - i - 1
        f["remaining_frac"] = (n_sents - i - 1) / max(n_sents, 1)

        # --- Contextual window features ---
        # Average novel token ratio in a window around this sentence
        window = 3
        window_novel = []
        for j in range(max(0, i - window), min(n_sents, i + window + 1)):
            if j == i:
                continue
            wj_tokens = set(sentences[j].lower().split())
            wj_novel = len(wj_tokens - source_set) / max(len(wj_tokens), 1) if source_set else 1.0
            window_novel.append(wj_novel)
        f["window_novel_mean"] = np.mean(window_novel) if window_novel else 0
        f["novel_vs_window"] = f["novel_token_ratio"] - f["window_novel_mean"]

        features_list.append(f)

    return features_list


def compute_embedding_sims(docs, embed_model):
    """Compute embedding similarity between source and each sentence."""
    log.info("Computing embedding similarities...")
    all_sources = []
    all_sents = []
    doc_sent_map = []
    for doc in docs:
        src_idx = len(all_sources)
        all_sources.append(doc["source"][:1500])
        for sent in doc["sentences"]:
            all_sents.append(sent)
            doc_sent_map.append(src_idx)

    log.info("  Encoding %d sources...", len(all_sources))
    src_embs = embed_model.encode(all_sources, batch_size=128, show_progress_bar=True,
                                  normalize_embeddings=True)
    log.info("  Encoding %d sentences...", len(all_sents))
    sent_embs = embed_model.encode(all_sents, batch_size=256, show_progress_bar=True,
                                   normalize_embeddings=True)

    sims = []
    for i in range(len(all_sents)):
        cos = float(np.dot(src_embs[doc_sent_map[i]], sent_embs[i]))
        sims.append(cos)

    log.info("  Mean sim: %.3f, Std: %.3f", np.mean(sims), np.std(sims))
    return sims


def compute_nli_scores(docs, nli_model):
    """Compute NLI entailment scores: P(source entails sentence)."""
    log.info("Computing NLI entailment scores...")
    pairs = []
    for doc in docs:
        src = doc["source"][:1500]
        for sent in doc["sentences"]:
            pairs.append((src, sent))

    log.info("  Scoring %d pairs...", len(pairs))
    # Cross-encoder predict in batches
    batch_size = 128
    all_scores = []
    for start in tqdm(range(0, len(pairs), batch_size), desc="NLI scoring"):
        batch = pairs[start:start + batch_size]
        scores = nli_model.predict(batch)
        if isinstance(scores, np.ndarray) and scores.ndim == 2:
            # Multi-class NLI: [contradiction, neutral, entailment]
            entailment_scores = scores[:, -1]  # entailment probability
            contradiction_scores = scores[:, 0]  # contradiction probability
            all_scores.extend(zip(entailment_scores, contradiction_scores))
        else:
            all_scores.extend([(s, 0.0) for s in scores])

    entail = [s[0] for s in all_scores]
    contra = [s[1] for s in all_scores]
    log.info("  Mean entailment: %.3f, Mean contradiction: %.3f", np.mean(entail), np.mean(contra))
    return entail, contra


def build_dataset(docs, embedding_sims=None, nli_entail=None, nli_contra=None):
    all_features = []
    all_labels = []
    all_doc_ids = []
    sim_idx = 0

    for doc in tqdm(docs, desc="Extracting features"):
        features = extract_features(doc)
        for i, f in enumerate(features):
            if embedding_sims is not None:
                f["embedding_sim_to_source"] = embedding_sims[sim_idx]
            if nli_entail is not None:
                f["nli_entailment"] = nli_entail[sim_idx]
            if nli_contra is not None:
                f["nli_contradiction"] = nli_contra[sim_idx]
            if embedding_sims is not None or nli_entail is not None:
                sim_idx += 1
            all_features.append(f)
            if "labels" in doc:
                all_labels.append(LABEL2IDX.get(doc["labels"][i], 0))
            all_doc_ids.append(doc["id"])

    df = pd.DataFrame(all_features)
    df["doc_id"] = all_doc_ids
    if all_labels:
        df["label"] = all_labels
    return df


def train_multiclass_ensemble(train_df, dev_df, feature_cols, seeds=None):
    """Train multi-seed LightGBM ensemble for 6-class classification."""
    if seeds is None:
        seeds = SEEDS

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_dev = dev_df[feature_cols].values
    y_dev = dev_df["label"].values

    # Sqrt-balanced class weights (less extreme than inverse frequency)
    class_counts = Counter(y_train)
    total = len(y_train)
    max_count = max(class_counts.values())
    class_weights = {c: math.sqrt(max_count / count) for c, count in class_counts.items()}
    sample_weights = np.array([class_weights[y] for y in y_train])

    params = {
        "objective": "multiclass", "num_class": len(LABELS), "metric": "multi_logloss",
        "learning_rate": 0.03, "num_leaves": 127, "max_depth": 10,
        "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.05, "reg_lambda": 0.5, "verbose": -1, "n_jobs": -1,
    }

    models = []
    for seed in seeds:
        log.info("  Training multiclass seed=%d...", seed)
        params["seed"] = seed
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                                 feature_name=feature_cols)
        dev_data = lgb.Dataset(X_dev, label=y_dev, reference=train_data,
                               feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=4000, valid_sets=[dev_data],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
        models.append(model)

    # Ensemble prediction
    dev_probs = np.mean([m.predict(X_dev) for m in models], axis=0)
    dev_preds = dev_probs.argmax(axis=1)
    macro_f1 = f1_score(y_dev, dev_preds, average="macro")
    log.info("\n  Multi-class Macro F1: %.4f", macro_f1)
    log.info("\n%s", classification_report(y_dev, dev_preds, target_names=LABELS, zero_division=0))

    return models, dev_probs


def train_binary_ensemble(train_df, dev_df, feature_cols, seeds=None):
    """Train multi-seed LightGBM ensemble for binary identification."""
    if seeds is None:
        seeds = SEEDS

    X_train = train_df[feature_cols].values
    y_train_multi = train_df["label"].values
    y_train = (y_train_multi != 0).astype(int)  # binary: 0=None, 1=Overgeneration

    X_dev = dev_df[feature_cols].values
    y_dev_multi = dev_df["label"].values
    y_dev = (y_dev_multi != 0).astype(int)

    # Balance weights
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    params = {
        "objective": "binary", "metric": "binary_logloss",
        "learning_rate": 0.03, "num_leaves": 127, "max_depth": 10,
        "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.05, "reg_lambda": 0.5, "verbose": -1, "n_jobs": -1,
        "scale_pos_weight": scale_pos_weight,
    }

    models = []
    for seed in seeds:
        log.info("  Training binary seed=%d...", seed)
        params["seed"] = seed
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        dev_data = lgb.Dataset(X_dev, label=y_dev, reference=train_data,
                               feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=4000, valid_sets=[dev_data],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
        models.append(model)

    dev_probs = np.mean([m.predict(X_dev) for m in models], axis=0)
    return models, dev_probs, y_dev


def optimize_binary_threshold(probs, y_true, dev_df):
    """Optimize threshold for sentence-level and document-level F1."""
    best_sent_f1, best_sent_thresh = 0, 0.5
    best_doc_f1, best_doc_thresh = 0, 0.5

    for thresh in np.arange(0.10, 0.90, 0.005):
        preds = (probs > thresh).astype(int)

        # Sentence-level F1
        sent_f1 = f1_score(y_true, preds, average="binary")
        if sent_f1 > best_sent_f1:
            best_sent_f1 = sent_f1
            best_sent_thresh = thresh

        # Document-level F1
        doc_true, doc_pred = [], []
        for doc_id in dev_df["doc_id"].unique():
            mask = dev_df["doc_id"] == doc_id
            doc_true.append(int(any(y_true[mask.values] == 1)))
            doc_pred.append(int(any(preds[mask.values] == 1)))
        doc_f1 = f1_score(doc_true, doc_pred, average="binary")
        if doc_f1 > best_doc_f1:
            best_doc_f1 = doc_f1
            best_doc_thresh = thresh

    log.info("  Best sentence threshold: %.3f → F1=%.4f", best_sent_thresh, best_sent_f1)
    log.info("  Best document threshold: %.3f → F1=%.4f", best_doc_thresh, best_doc_f1)
    return best_sent_thresh, best_doc_thresh


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-nli", action="store_true", help="Skip NLI cross-encoder")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding similarity")
    args = parser.parse_args()

    use_nli = not args.no_nli
    use_embed = not args.no_embed

    # Load embedding model
    embed_model = None
    if use_embed:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model...")
        embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load NLI cross-encoder
    nli_model = None
    if use_nli:
        try:
            from sentence_transformers import CrossEncoder
            log.info("Loading NLI cross-encoder...")
            nli_model = CrossEncoder("cross-encoder/nli-deberta-v3-base")
        except Exception as e:
            log.warning("Could not load NLI model: %s. Continuing without NLI.", e)
            use_nli = False

    # Load data
    log.info("Loading data...")
    train_docs = load_data("train_data.json")
    dev_docs = load_data("dev_data.json")

    # Compute embeddings
    train_sims, dev_sims = None, None
    if use_embed and embed_model:
        train_sims = compute_embedding_sims(train_docs, embed_model)
        dev_sims = compute_embedding_sims(dev_docs, embed_model)

    # Compute NLI scores
    train_entail, train_contra = None, None
    dev_entail, dev_contra = None, None
    if use_nli and nli_model:
        train_entail, train_contra = compute_nli_scores(train_docs, nli_model)
        dev_entail, dev_contra = compute_nli_scores(dev_docs, nli_model)

    # Build feature DataFrames
    log.info("Building features...")
    train_df = build_dataset(train_docs, train_sims, train_entail, train_contra)
    dev_df = build_dataset(dev_docs, dev_sims, dev_entail, dev_contra)

    feature_cols = [c for c in train_df.columns if c not in ["doc_id", "label"]]
    log.info("Features: %d", len(feature_cols))
    log.info("Train: %s, Dev: %s", train_df.shape, dev_df.shape)

    # ========================================================================
    # Task 2.2: Multi-class classification
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("  TASK 2.2: Multi-class Classification")
    log.info("=" * 60)

    mc_models, dev_mc_probs = train_multiclass_ensemble(train_df, dev_df, feature_cols)

    # ========================================================================
    # Task 2.1: Binary identification
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("  TASK 2.1: Binary Identification")
    log.info("=" * 60)

    bin_models, dev_bin_probs, dev_bin_y = train_binary_ensemble(train_df, dev_df, feature_cols)
    sent_thresh, doc_thresh = optimize_binary_threshold(dev_bin_probs, dev_bin_y, dev_df)

    # Also compute doc-level F1 from the multi-class model
    mc_preds = dev_mc_probs.argmax(axis=1)
    doc_y_mc, doc_p_mc = [], []
    for doc_id in dev_df["doc_id"].unique():
        mask = dev_df["doc_id"] == doc_id
        true_l = dev_df.loc[mask, "label"].values
        pred_l = mc_preds[mask.values]
        doc_y_mc.append(int(any(l != 0 for l in true_l)))
        doc_p_mc.append(int(any(l != 0 for l in pred_l)))
    log.info("  Doc F1 (from multi-class): %.4f", f1_score(doc_y_mc, doc_p_mc))

    # ========================================================================
    # Feature importance
    # ========================================================================
    importance = mc_models[0].feature_importance(importance_type="gain")
    top_features = sorted(zip(feature_cols, importance), key=lambda x: -x[1])[:20]
    log.info("\nTop 20 features (multi-class, gain):")
    for name, imp in top_features:
        log.info("  %-35s %10.0f", name, imp)

    # ========================================================================
    # Generate test predictions
    # ========================================================================
    if not os.path.exists("test_data.json"):
        log.info("\nNo test_data.json found. Skipping test prediction.")
        return

    log.info("\n" + "=" * 60)
    log.info("  Generating Test Submissions")
    log.info("=" * 60)

    test_docs = load_data("test_data.json")
    log.info("Test: %d docs, %d sentences",
             len(test_docs), sum(len(d["sentences"]) for d in test_docs))

    test_sims = None
    if use_embed and embed_model:
        test_sims = compute_embedding_sims(test_docs, embed_model)

    test_entail, test_contra = None, None
    if use_nli and nli_model:
        test_entail, test_contra = compute_nli_scores(test_docs, nli_model)

    test_df = build_dataset(test_docs, test_sims, test_entail, test_contra)

    # Multi-class predictions
    test_mc_probs = np.mean([m.predict(test_df[feature_cols].values) for m in mc_models], axis=0)
    # Binary predictions
    test_bin_probs = np.mean([m.predict(test_df[feature_cols].values) for m in bin_models], axis=0)

    # Build submissions
    mc_results = []
    bin_results_sent = []
    bin_results_doc = []
    idx = 0
    for doc in test_docs:
        n = len(doc["sentences"])
        dp_mc = test_mc_probs[idx:idx + n]
        dp_bin = test_bin_probs[idx:idx + n]
        idx += n

        # Task 2.2: Multi-class labels per sentence
        multi_labels = [LABELS[p] for p in dp_mc.argmax(axis=1)]
        mc_results.append({
            "id": doc["id"],
            "labels": multi_labels,
            "run_id": "writerslogic_Task22_v3",
        })

        # Task 2.1: Binary identification per sentence (using sentence threshold)
        binary_labels = ["Overgeneration" if p > sent_thresh else "None" for p in dp_bin]
        bin_results_sent.append({
            "id": doc["id"],
            "labels": binary_labels,
            "run_id": "writerslogic_Task21_v3_sent",
        })

        # Task 2.1: Binary identification (using doc threshold for higher doc-level F1)
        binary_labels_doc = ["Overgeneration" if p > doc_thresh else "None" for p in dp_bin]
        bin_results_doc.append({
            "id": doc["id"],
            "labels": binary_labels_doc,
            "run_id": "writerslogic_Task21_v3_doc",
        })

    # Save and zip submissions
    os.makedirs("submissions_v3", exist_ok=True)

    submissions = [
        ("submissions_v3/writerslogic_Task22_v3.json", mc_results, "Task 2.2 (classification)"),
        ("submissions_v3/writerslogic_Task21_v3_sent.json", bin_results_sent, "Task 2.1 (identification, sentence-opt)"),
        ("submissions_v3/writerslogic_Task21_v3_doc.json", bin_results_doc, "Task 2.1 (identification, doc-opt)"),
    ]

    for path, results, desc in submissions:
        with open(path, "w") as f:
            json.dump(results, f)
        zip_path = path.replace(".json", ".zip")
        subprocess.run(["zip", "-j", zip_path, path], check=True)
        # Count predictions
        total_sents = sum(len(r["labels"]) for r in results)
        non_none = sum(1 for r in results for l in r["labels"] if l != "None")
        log.info("  %s: %s (%d/%d non-None)", desc, zip_path, non_none, total_sents)

    log.info("\nAll submissions ready in submissions_v3/")


if __name__ == "__main__":
    main()
