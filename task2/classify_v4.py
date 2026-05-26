"""
SimpleText Task 2 v4: All CPU-feasible improvements over v3.

Improvements over v3:
- Upgraded embedding model (all-mpnet-base-v2)
- Upgraded NLI model (cross-encoder/nli-MiniLM2-L6-H768)
- Sequential/transition features (CRF-like: captures "overgeneration continues once started")
- Stacking meta-learner (second-stage LogisticRegression on LightGBM probabilities)
- Joint threshold optimization (sentence + document F1)
- Per-class threshold tuning for Task 2.2 macro F1
- 5-fold cross-validation with full retrain for test

Usage:
    uv run python classify_v4.py
    uv run python classify_v4.py --no-nli   # skip NLI (faster)
    uv run python classify_v4.py --no-embed  # skip embeddings (faster)
"""

import json
import logging
import math
import os
import re
import subprocess
import sys
import zlib
from collections import Counter

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold
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
# Feature engineering (same as v3 + sequential features)
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

    # Pre-compute per-sentence token sets and novel ratios for sequential features
    sent_token_sets = []
    sent_novel_ratios = []
    for sent in sentences:
        st = set(sent.lower().split())
        sent_token_sets.append(st)
        nr = len(st - source_set) / max(len(st), 1) if source_set else 1.0
        sent_novel_ratios.append(nr)

    features_list = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        sent_tokens = sent_lower.split()
        sent_set = sent_token_sets[i]
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
        f["starts_with_bullet"] = int(bool(re.match(r"^[-\*\u2022]\s", sent)))
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
            prev_j_set = sent_token_sets[j]
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
        window = 3
        window_novel = []
        for j in range(max(0, i - window), min(n_sents, i + window + 1)):
            if j == i:
                continue
            window_novel.append(sent_novel_ratios[j])
        f["window_novel_mean"] = np.mean(window_novel) if window_novel else 0
        f["novel_vs_window"] = f["novel_token_ratio"] - f["window_novel_mean"]

        # ====================================================================
        # NEW v4: Sequential / transition features
        # ====================================================================

        # Novel token ratio of previous sentences (signals transition into overgeneration)
        if i > 0:
            f["prev_novel_ratio"] = sent_novel_ratios[i - 1]
            f["novel_ratio_delta"] = sent_novel_ratios[i] - sent_novel_ratios[i - 1]
        else:
            f["prev_novel_ratio"] = 0
            f["novel_ratio_delta"] = 0

        if i > 1:
            f["prev2_novel_ratio"] = sent_novel_ratios[i - 2]
            f["novel_ratio_accel"] = (sent_novel_ratios[i] - sent_novel_ratios[i - 1]) - \
                                     (sent_novel_ratios[i - 1] - sent_novel_ratios[i - 2])
        else:
            f["prev2_novel_ratio"] = 0
            f["novel_ratio_accel"] = 0

        # Cumulative novel ratio from this point to end of doc (tail divergence)
        tail_novels = sent_novel_ratios[i:]
        f["tail_novel_mean"] = np.mean(tail_novels)
        f["tail_novel_max"] = max(tail_novels)
        f["tail_novel_min"] = min(tail_novels)

        # Running average of novel ratio up to this point
        head_novels = sent_novel_ratios[:i + 1]
        f["head_novel_mean"] = np.mean(head_novels)

        # Max novel ratio in previous sentences (has overgeneration already appeared?)
        if i > 0:
            f["max_prev_novel_ratio"] = max(sent_novel_ratios[:i])
            f["mean_prev_novel_ratio"] = np.mean(sent_novel_ratios[:i])
        else:
            f["max_prev_novel_ratio"] = 0
            f["mean_prev_novel_ratio"] = 0

        # Ratio of remaining sentences with high novelty (>0.5)
        if i < n_sents - 1:
            remaining_novels = sent_novel_ratios[i + 1:]
            f["frac_remaining_high_novel"] = sum(1 for x in remaining_novels if x > 0.5) / len(remaining_novels)
        else:
            f["frac_remaining_high_novel"] = 0

        # Previous sentence prompt pattern count (transition signal)
        if i > 0:
            prev_prompt_count = sum(1 for p in prompt_patterns if re.search(p, sentences[i - 1], re.IGNORECASE))
            f["prev_prompt_pattern_count"] = prev_prompt_count
        else:
            f["prev_prompt_pattern_count"] = 0

        # Previous sentence length ratio (sudden length change = signal)
        if i > 0:
            prev_len = len(sentences[i - 1].split())
            f["len_ratio_vs_prev"] = len(sent_tokens) / max(prev_len, 1)
        else:
            f["len_ratio_vs_prev"] = 1.0

        # Consecutive high-novelty streak ending at this sentence
        streak = 0
        for j in range(i, -1, -1):
            if sent_novel_ratios[j] > 0.3:
                streak += 1
            else:
                break
        f["high_novel_streak"] = streak

        # Position relative to first high-novelty sentence
        first_high = n_sents  # sentinel
        for j in range(n_sents):
            if sent_novel_ratios[j] > 0.5:
                first_high = j
                break
        f["dist_from_first_high_novel"] = i - first_high  # negative = before, positive = after

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
    batch_size = 128
    all_scores = []
    for start in tqdm(range(0, len(pairs), batch_size), desc="NLI scoring"):
        batch = pairs[start:start + batch_size]
        scores = nli_model.predict(batch)
        if isinstance(scores, np.ndarray) and scores.ndim == 2:
            entailment_scores = scores[:, -1]
            contradiction_scores = scores[:, 0]
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


# ============================================================================
# LightGBM training
# ============================================================================

LGB_PARAMS_MC = {
    "objective": "multiclass", "num_class": len(LABELS), "metric": "multi_logloss",
    "learning_rate": 0.03, "num_leaves": 127, "max_depth": 10,
    "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.05, "reg_lambda": 0.5, "verbose": -1, "n_jobs": -1,
}

LGB_PARAMS_BIN = {
    "objective": "binary", "metric": "binary_logloss",
    "learning_rate": 0.03, "num_leaves": 127, "max_depth": 10,
    "min_child_samples": 10, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.05, "reg_lambda": 0.5, "verbose": -1, "n_jobs": -1,
}


def train_multiclass_models(X_train, y_train, X_val, y_val, feature_cols, seeds=None):
    """Train multi-seed LightGBM multiclass ensemble."""
    if seeds is None:
        seeds = SEEDS

    class_counts = Counter(y_train)
    max_count = max(class_counts.values())
    class_weights = {c: math.sqrt(max_count / count) for c, count in class_counts.items()}
    sample_weights = np.array([class_weights[y] for y in y_train])

    models = []
    for seed in seeds:
        params = {**LGB_PARAMS_MC, "seed": seed}
        train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights,
                                 feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data,
                               feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=4000, valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
        models.append(model)

    return models


def train_binary_models(X_train, y_train, X_val, y_val, feature_cols, seeds=None):
    """Train multi-seed LightGBM binary ensemble."""
    if seeds is None:
        seeds = SEEDS

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    models = []
    for seed in seeds:
        params = {**LGB_PARAMS_BIN, "seed": seed, "scale_pos_weight": scale_pos_weight}
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data,
                               feature_name=feature_cols)
        model = lgb.train(params, train_data, num_boost_round=4000, valid_sets=[val_data],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
        models.append(model)

    return models


# ============================================================================
# Cross-validation with stacking
# ============================================================================

def cross_validate_with_stacking(full_df, feature_cols, n_folds=5):
    """5-fold GroupKFold CV (grouped by doc_id) producing OOF predictions for stacking."""
    X = full_df[feature_cols].values
    y_multi = full_df["label"].values
    y_bin = (y_multi != 0).astype(int)
    groups = full_df["doc_id"].values

    gkf = GroupKFold(n_splits=n_folds)

    oof_mc_probs = np.zeros((len(X), len(LABELS)))
    oof_bin_probs = np.zeros(len(X))

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y_multi, groups)):
        log.info("\n--- Fold %d/%d ---", fold + 1, n_folds)
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr_mc, y_val_mc = y_multi[train_idx], y_multi[val_idx]
        y_tr_bin, y_val_bin = y_bin[train_idx], y_bin[val_idx]

        # Train multiclass (single seed per fold for speed)
        mc_models = train_multiclass_models(X_tr, y_tr_mc, X_val, y_val_mc, feature_cols,
                                            seeds=[SEEDS[fold % len(SEEDS)]])
        oof_mc_probs[val_idx] = np.mean([m.predict(X_val) for m in mc_models], axis=0)

        # Train binary
        bin_models = train_binary_models(X_tr, y_tr_bin, X_val, y_val_bin, feature_cols,
                                         seeds=[SEEDS[fold % len(SEEDS)]])
        oof_bin_probs[val_idx] = np.mean([m.predict(X_val) for m in bin_models], axis=0)

        # Fold metrics
        fold_mc_preds = oof_mc_probs[val_idx].argmax(axis=1)
        fold_mc_f1 = f1_score(y_val_mc, fold_mc_preds, average="macro")
        fold_bin_preds = (oof_bin_probs[val_idx] > 0.5).astype(int)
        fold_bin_f1 = f1_score(y_val_bin, fold_bin_preds, average="binary")
        log.info("  Fold %d: MC macro F1=%.4f, Bin F1=%.4f", fold + 1, fold_mc_f1, fold_bin_f1)

    # Overall OOF metrics
    oof_mc_preds = oof_mc_probs.argmax(axis=1)
    log.info("\nOOF Multi-class Macro F1: %.4f", f1_score(y_multi, oof_mc_preds, average="macro"))
    log.info("\n%s", classification_report(y_multi, oof_mc_preds, target_names=LABELS, zero_division=0))

    oof_bin_preds = (oof_bin_probs > 0.5).astype(int)
    log.info("OOF Binary F1: %.4f", f1_score(y_bin, oof_bin_preds, average="binary"))

    return oof_mc_probs, oof_bin_probs


# ============================================================================
# Stacking meta-learner
# ============================================================================

def train_meta_learner_mc(oof_probs, y_true, feature_df, feature_cols):
    """Train a logistic regression meta-learner on OOF multiclass probabilities + meta features."""
    meta_features = np.column_stack([
        oof_probs,
        feature_df["sent_position_rel"].values,
        feature_df["n_sentences_in_doc"].values,
        feature_df["novel_token_ratio"].values,
        feature_df["high_novel_streak"].values,
        feature_df["tail_novel_mean"].values,
    ])

    meta_model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", n_jobs=-1,
    )
    meta_model.fit(meta_features, y_true)
    meta_preds = meta_model.predict(meta_features)
    log.info("Meta-learner MC train Macro F1: %.4f", f1_score(y_true, meta_preds, average="macro"))
    return meta_model


def train_meta_learner_bin(oof_probs, y_true, feature_df):
    """Train a logistic regression meta-learner on OOF binary probabilities + meta features."""
    meta_features = np.column_stack([
        oof_probs.reshape(-1, 1),
        feature_df["sent_position_rel"].values,
        feature_df["n_sentences_in_doc"].values,
        feature_df["novel_token_ratio"].values,
        feature_df["high_novel_streak"].values,
        feature_df["tail_novel_mean"].values,
    ])

    meta_model = LogisticRegression(
        C=1.0, max_iter=1000, solver="lbfgs",
        class_weight="balanced", n_jobs=-1,
    )
    meta_model.fit(meta_features, y_true)
    meta_preds = meta_model.predict(meta_features)
    log.info("Meta-learner Bin train F1: %.4f", f1_score(y_true, meta_preds, average="binary"))
    return meta_model


def build_meta_features_mc(base_probs, feature_df):
    return np.column_stack([
        base_probs,
        feature_df["sent_position_rel"].values,
        feature_df["n_sentences_in_doc"].values,
        feature_df["novel_token_ratio"].values,
        feature_df["high_novel_streak"].values,
        feature_df["tail_novel_mean"].values,
    ])


def build_meta_features_bin(base_probs, feature_df):
    return np.column_stack([
        base_probs.reshape(-1, 1),
        feature_df["sent_position_rel"].values,
        feature_df["n_sentences_in_doc"].values,
        feature_df["novel_token_ratio"].values,
        feature_df["high_novel_streak"].values,
        feature_df["tail_novel_mean"].values,
    ])


# ============================================================================
# Threshold optimization
# ============================================================================

def optimize_joint_threshold(probs, y_true, doc_ids, sent_weight=0.5, doc_weight=0.5):
    """Optimize threshold jointly for sentence-level and document-level F1."""
    best_score = 0
    best_thresh = 0.5
    best_sent_f1 = 0
    best_doc_f1 = 0

    unique_docs = np.unique(doc_ids)

    for thresh in np.arange(0.10, 0.90, 0.005):
        preds = (probs > thresh).astype(int)
        sent_f1 = f1_score(y_true, preds, average="binary")

        doc_true, doc_pred = [], []
        for doc_id in unique_docs:
            mask = doc_ids == doc_id
            doc_true.append(int(any(y_true[mask] == 1)))
            doc_pred.append(int(any(preds[mask] == 1)))
        doc_f1 = f1_score(doc_true, doc_pred, average="binary")

        score = sent_weight * sent_f1 + doc_weight * doc_f1
        if score > best_score:
            best_score = score
            best_thresh = thresh
            best_sent_f1 = sent_f1
            best_doc_f1 = doc_f1

    log.info("  Joint threshold: %.3f (sent F1=%.4f, doc F1=%.4f, combined=%.4f)",
             best_thresh, best_sent_f1, best_doc_f1, best_score)
    return best_thresh


def optimize_sent_threshold(probs, y_true):
    """Optimize threshold for sentence-level F1."""
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.10, 0.90, 0.005):
        preds = (probs > thresh).astype(int)
        f = f1_score(y_true, preds, average="binary")
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
    log.info("  Sent threshold: %.3f → F1=%.4f", best_thresh, best_f1)
    return best_thresh


def optimize_doc_threshold(probs, y_true, doc_ids):
    """Optimize threshold for document-level F1."""
    best_f1, best_thresh = 0, 0.5
    unique_docs = np.unique(doc_ids)
    for thresh in np.arange(0.10, 0.90, 0.005):
        preds = (probs > thresh).astype(int)
        doc_true, doc_pred = [], []
        for doc_id in unique_docs:
            mask = doc_ids == doc_id
            doc_true.append(int(any(y_true[mask] == 1)))
            doc_pred.append(int(any(preds[mask] == 1)))
        f = f1_score(doc_true, doc_pred, average="binary")
        if f > best_f1:
            best_f1 = f
            best_thresh = thresh
    log.info("  Doc threshold: %.3f → F1=%.4f", best_thresh, best_f1)
    return best_thresh


def optimize_perclass_thresholds(probs, y_true):
    """Per-class threshold tuning to maximize macro F1 for multiclass.

    Instead of simple argmax, adjust per-class thresholds to boost recall
    on rare classes while maintaining precision on common ones.
    """
    n_classes = probs.shape[1]
    # Start with uniform thresholds
    thresholds = np.ones(n_classes) * 0.0  # offsets to add to probabilities before argmax

    # Greedy per-class optimization
    best_macro_f1 = f1_score(y_true, probs.argmax(axis=1), average="macro")
    log.info("  Baseline argmax macro F1: %.4f", best_macro_f1)

    for cls in range(1, n_classes):  # skip None (class 0)
        best_offset = 0.0
        for offset in np.arange(-0.15, 0.30, 0.005):
            adjusted = probs.copy()
            adjusted[:, cls] += offset
            preds = adjusted.argmax(axis=1)
            mf1 = f1_score(y_true, preds, average="macro")
            if mf1 > best_macro_f1:
                best_macro_f1 = mf1
                best_offset = offset
        thresholds[cls] = best_offset
        if best_offset != 0:
            log.info("    Class %d (%s): offset=%.3f", cls, LABELS[cls], best_offset)

    # Apply all offsets
    adjusted = probs.copy()
    for cls in range(n_classes):
        adjusted[:, cls] += thresholds[cls]
    final_preds = adjusted.argmax(axis=1)
    final_f1 = f1_score(y_true, final_preds, average="macro")
    log.info("  Per-class tuned macro F1: %.4f (improvement: +%.4f)",
             final_f1, final_f1 - f1_score(y_true, probs.argmax(axis=1), average="macro"))

    return thresholds


# ============================================================================
# Main
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-nli", action="store_true", help="Skip NLI cross-encoder")
    parser.add_argument("--no-embed", action="store_true", help="Skip embedding similarity")
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation (faster)")
    args = parser.parse_args()

    use_nli = not args.no_nli
    use_embed = not args.no_embed
    use_cv = not args.no_cv

    # Load models
    embed_model = None
    if use_embed:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model (all-mpnet-base-v2)...")
        embed_model = SentenceTransformer("all-mpnet-base-v2")

    nli_model = None
    if use_nli:
        try:
            from sentence_transformers import CrossEncoder
            log.info("Loading NLI cross-encoder (nli-MiniLM2-L6-H768)...")
            nli_model = CrossEncoder("cross-encoder/nli-MiniLM2-L6-H768")
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
    # Cross-validation on train+dev combined (for stacking)
    # ========================================================================
    full_df = pd.concat([train_df, dev_df], ignore_index=True)
    y_full_multi = full_df["label"].values
    y_full_bin = (y_full_multi != 0).astype(int)

    if use_cv:
        log.info("\n" + "=" * 60)
        log.info("  CROSS-VALIDATION (5-fold GroupKFold)")
        log.info("=" * 60)
        oof_mc_probs, oof_bin_probs = cross_validate_with_stacking(full_df, feature_cols)

        # Train meta-learners on OOF predictions
        log.info("\n" + "=" * 60)
        log.info("  STACKING META-LEARNERS")
        log.info("=" * 60)
        meta_mc = train_meta_learner_mc(oof_mc_probs, y_full_multi, full_df, feature_cols)
        meta_bin = train_meta_learner_bin(oof_bin_probs, y_full_bin, full_df)

        # Per-class threshold tuning on OOF multiclass predictions
        log.info("\nPer-class threshold tuning on OOF predictions:")
        # Use meta-learner adjusted probabilities for threshold tuning
        meta_mc_features = build_meta_features_mc(oof_mc_probs, full_df)
        oof_mc_meta_probs = meta_mc.predict_proba(meta_mc_features)
        perclass_offsets = optimize_perclass_thresholds(oof_mc_meta_probs, y_full_multi)

        # Binary threshold optimization on OOF
        meta_bin_features = build_meta_features_bin(oof_bin_probs, full_df)
        oof_bin_meta_probs = meta_bin.predict_proba(meta_bin_features)[:, 1]
        doc_ids_full = full_df["doc_id"].values

        log.info("\nJoint threshold optimization on OOF predictions:")
        joint_thresh = optimize_joint_threshold(oof_bin_meta_probs, y_full_bin, doc_ids_full)
        sent_thresh = optimize_sent_threshold(oof_bin_meta_probs, y_full_bin)
        doc_thresh = optimize_doc_threshold(oof_bin_meta_probs, y_full_bin, doc_ids_full)
    else:
        meta_mc = None
        meta_bin = None
        perclass_offsets = np.zeros(len(LABELS))
        joint_thresh = 0.5
        sent_thresh = 0.5
        doc_thresh = 0.5

    # ========================================================================
    # Train final models on ALL data (train + dev)
    # ========================================================================
    log.info("\n" + "=" * 60)
    log.info("  TRAINING FINAL MODELS ON ALL DATA")
    log.info("=" * 60)

    X_full = full_df[feature_cols].values
    y_full_mc = full_df["label"].values
    y_full_bn = (y_full_mc != 0).astype(int)

    # We still need a validation set for early stopping -- use dev portion
    n_train = len(train_df)
    X_train_final, X_val_final = X_full[:n_train], X_full[n_train:]
    y_train_mc_final, y_val_mc_final = y_full_mc[:n_train], y_full_mc[n_train:]
    y_train_bn_final, y_val_bn_final = y_full_bn[:n_train], y_full_bn[n_train:]

    log.info("\nTraining final multiclass ensemble (5 seeds)...")
    final_mc_models = train_multiclass_models(X_train_final, y_train_mc_final,
                                              X_val_final, y_val_mc_final,
                                              feature_cols, seeds=SEEDS)

    log.info("\nTraining final binary ensemble (5 seeds)...")
    final_bin_models = train_binary_models(X_train_final, y_train_bn_final,
                                           X_val_final, y_val_bn_final,
                                           feature_cols, seeds=SEEDS)

    # Dev set evaluation with all improvements
    log.info("\n" + "=" * 60)
    log.info("  DEV SET EVALUATION")
    log.info("=" * 60)

    dev_mc_probs = np.mean([m.predict(X_val_final) for m in final_mc_models], axis=0)
    dev_bin_probs = np.mean([m.predict(X_val_final) for m in final_bin_models], axis=0)

    # Without meta-learner
    dev_mc_preds_base = dev_mc_probs.argmax(axis=1)
    log.info("Dev MC Macro F1 (base): %.4f", f1_score(y_val_mc_final, dev_mc_preds_base, average="macro"))

    if meta_mc is not None:
        # With meta-learner
        dev_meta_mc_features = build_meta_features_mc(dev_mc_probs, dev_df)
        dev_mc_meta_probs = meta_mc.predict_proba(dev_meta_mc_features)
        dev_mc_preds_meta = dev_mc_meta_probs.argmax(axis=1)
        log.info("Dev MC Macro F1 (meta): %.4f", f1_score(y_val_mc_final, dev_mc_preds_meta, average="macro"))

        # With per-class offsets
        dev_mc_adjusted = dev_mc_meta_probs.copy()
        for cls in range(len(LABELS)):
            dev_mc_adjusted[:, cls] += perclass_offsets[cls]
        dev_mc_preds_tuned = dev_mc_adjusted.argmax(axis=1)
        log.info("Dev MC Macro F1 (tuned): %.4f", f1_score(y_val_mc_final, dev_mc_preds_tuned, average="macro"))
        log.info("\n%s", classification_report(y_val_mc_final, dev_mc_preds_tuned, target_names=LABELS, zero_division=0))

    if meta_bin is not None:
        dev_meta_bin_features = build_meta_features_bin(dev_bin_probs, dev_df)
        dev_bin_meta_probs = meta_bin.predict_proba(dev_meta_bin_features)[:, 1]

        dev_bin_preds_joint = (dev_bin_meta_probs > joint_thresh).astype(int)
        log.info("Dev Bin F1 (joint thresh): %.4f", f1_score(y_val_bn_final, dev_bin_preds_joint, average="binary"))

        dev_bin_preds_sent = (dev_bin_meta_probs > sent_thresh).astype(int)
        log.info("Dev Bin F1 (sent thresh): %.4f", f1_score(y_val_bn_final, dev_bin_preds_sent, average="binary"))

    # ========================================================================
    # Generate test predictions
    # ========================================================================
    if not os.path.exists("test_data.json"):
        log.info("\nNo test_data.json found. Skipping test prediction.")
        return

    log.info("\n" + "=" * 60)
    log.info("  GENERATING TEST SUBMISSIONS")
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

    # Base model predictions
    X_test = test_df[feature_cols].values
    test_mc_probs = np.mean([m.predict(X_test) for m in final_mc_models], axis=0)
    test_bin_probs = np.mean([m.predict(X_test) for m in final_bin_models], axis=0)

    # Apply meta-learners if available
    if meta_mc is not None:
        test_meta_mc_features = build_meta_features_mc(test_mc_probs, test_df)
        test_mc_final_probs = meta_mc.predict_proba(test_meta_mc_features)
        # Apply per-class offsets
        for cls in range(len(LABELS)):
            test_mc_final_probs[:, cls] += perclass_offsets[cls]
    else:
        test_mc_final_probs = test_mc_probs

    if meta_bin is not None:
        test_meta_bin_features = build_meta_features_bin(test_bin_probs, test_df)
        test_bin_final_probs = meta_bin.predict_proba(test_meta_bin_features)[:, 1]
    else:
        test_bin_final_probs = test_bin_probs

    # Build submissions
    mc_results = []
    bin_results_sent = []
    bin_results_joint = []
    bin_results_doc = []
    idx = 0
    for doc in test_docs:
        n = len(doc["sentences"])
        dp_mc = test_mc_final_probs[idx:idx + n]
        dp_bin = test_bin_final_probs[idx:idx + n]
        idx += n

        # Task 2.2: Multi-class labels (with per-class tuning)
        multi_labels = [LABELS[p] for p in dp_mc.argmax(axis=1)]
        mc_results.append({
            "id": doc["id"],
            "labels": multi_labels,
            "run_id": "writerslogic_Task22_v4",
        })

        # Task 2.1: Binary identification - sentence-optimized threshold
        bin_labels_sent = ["Overgeneration" if p > sent_thresh else "None" for p in dp_bin]
        bin_results_sent.append({
            "id": doc["id"],
            "labels": bin_labels_sent,
            "run_id": "writerslogic_Task21_v4_sent",
        })

        # Task 2.1: Binary identification - joint-optimized threshold
        bin_labels_joint = ["Overgeneration" if p > joint_thresh else "None" for p in dp_bin]
        bin_results_joint.append({
            "id": doc["id"],
            "labels": bin_labels_joint,
            "run_id": "writerslogic_Task21_v4_joint",
        })

        # Task 2.1: Binary identification - doc-optimized threshold
        bin_labels_doc = ["Overgeneration" if p > doc_thresh else "None" for p in dp_bin]
        bin_results_doc.append({
            "id": doc["id"],
            "labels": bin_labels_doc,
            "run_id": "writerslogic_Task21_v4_doc",
        })

    # Save and zip
    os.makedirs("submissions_v4", exist_ok=True)

    submissions = [
        ("submissions_v4/writerslogic_Task22_v4.json", mc_results, "Task 2.2 (classification, tuned)"),
        ("submissions_v4/writerslogic_Task21_v4_sent.json", bin_results_sent, "Task 2.1 (sent-optimized)"),
        ("submissions_v4/writerslogic_Task21_v4_joint.json", bin_results_joint, "Task 2.1 (joint-optimized)"),
        ("submissions_v4/writerslogic_Task21_v4_doc.json", bin_results_doc, "Task 2.1 (doc-optimized)"),
    ]

    for path, results, desc in submissions:
        with open(path, "w") as f:
            json.dump(results, f)
        zip_path = path.replace(".json", ".zip")
        subprocess.run(["zip", "-j", zip_path, path], check=True)
        total_sents = sum(len(r["labels"]) for r in results)
        non_none = sum(1 for r in results for l in r["labels"] if l != "None")
        log.info("  %s: %s (%d/%d non-None)", desc, zip_path, non_none, total_sents)

    # Feature importance
    importance = final_mc_models[0].feature_importance(importance_type="gain")
    top_features = sorted(zip(feature_cols, importance), key=lambda x: -x[1])[:25]
    log.info("\nTop 25 features (multiclass, gain):")
    for name, imp in top_features:
        log.info("  %-40s %10.0f", name, imp)

    log.info("\nAll submissions ready in submissions_v4/")
    log.info("Thresholds: sent=%.3f, joint=%.3f, doc=%.3f", sent_thresh, joint_thresh, doc_thresh)
    log.info("Per-class offsets: %s", {LABELS[i]: f"{o:.3f}" for i, o in enumerate(perclass_offsets) if o != 0})


if __name__ == "__main__":
    main()
