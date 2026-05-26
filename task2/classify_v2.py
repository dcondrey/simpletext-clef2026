"""
SimpleText Task 2 v2: Improved classifier.
Improvements over v1:
- Embedding similarity (source vs sentence) computed in batch
- Longer substring matching features
- Threshold tuning for doc-level F1
- Both binary and multi-class submissions
- More features (trigram overlap, position patterns)
"""

import json
import logging
import os
import re
import subprocess
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.metrics import f1_score, classification_report
from sentence_transformers import SentenceTransformer
import lightgbm as lgb
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LABELS = ["None", "LEAKED_INSTRUCTIONS", "UNGROUNDED_INJECTION",
          "GENERATION_FAILURE", "REPETITIVE_CONTENT", "GROUNDED_OVERGENERATION"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


def load_data(path):
    with open(path) as f:
        return json.load(f)


def compute_embedding_sims(docs, embed_model):
    log.info("Computing embedding similarities in batch...")
    all_sources = []
    all_sents = []
    doc_sent_map = []
    for doc in docs:
        src_idx = len(all_sources)
        all_sources.append(doc["source"][:1000])
        for sent in doc["sentences"]:
            all_sents.append(sent)
            doc_sent_map.append(src_idx)

    log.info(f"  Encoding {len(all_sources)} sources...")
    src_embs = embed_model.encode(all_sources, batch_size=128, show_progress_bar=True)
    log.info(f"  Encoding {len(all_sents)} sentences...")
    sent_embs = embed_model.encode(all_sents, batch_size=256, show_progress_bar=True)

    sims = []
    for i in range(len(all_sents)):
        src_e = src_embs[doc_sent_map[i]]
        sent_e = sent_embs[i]
        cos = np.dot(src_e, sent_e) / (np.linalg.norm(src_e) * np.linalg.norm(sent_e) + 1e-8)
        sims.append(float(cos))

    log.info(f"  Done. Mean sim: {np.mean(sims):.3f}")
    return sims


def extract_features(doc):
    source = doc["source"]
    sentences = doc["sentences"]
    source_lower = source.lower()
    source_tokens = set(source_lower.split())
    source_words = source_lower.split()
    n_sents = len(sentences)

    source_bigrams = set(zip(source_words[:-1], source_words[1:])) if len(source_words) > 1 else set()
    source_trigrams = set(zip(source_words[:-2], source_words[1:-1], source_words[2:])) if len(source_words) > 2 else set()

    features_list = []
    for i, sent in enumerate(sentences):
        sent_lower = sent.lower()
        sent_tokens = sent_lower.split()
        sent_set = set(sent_tokens)
        f = {}

        f["sent_len_chars"] = len(sent)
        f["sent_len_words"] = len(sent_tokens)
        f["sent_position"] = i
        f["sent_position_rel"] = i / max(n_sents - 1, 1)
        f["is_last_sent"] = int(i == n_sents - 1)
        f["is_last_3"] = int(i >= n_sents - 3)
        f["is_last_5"] = int(i >= n_sents - 5)
        f["is_first_sent"] = int(i == 0)
        f["n_sentences_in_doc"] = n_sents

        if source_tokens:
            overlap = sent_set & source_tokens
            f["token_overlap_ratio"] = len(overlap) / max(len(sent_set), 1)
            f["novel_token_ratio"] = 1.0 - f["token_overlap_ratio"]
            f["novel_token_count"] = len(sent_set - source_tokens)
            f["source_coverage"] = len(overlap) / max(len(source_tokens), 1)
        else:
            f["token_overlap_ratio"] = 0
            f["novel_token_ratio"] = 1
            f["novel_token_count"] = len(sent_set)
            f["source_coverage"] = 0

        sent_bigrams = set(zip(sent_tokens[:-1], sent_tokens[1:])) if len(sent_tokens) > 1 else set()
        f["bigram_overlap_with_source"] = len(sent_bigrams & source_bigrams) / max(len(sent_bigrams), 1) if sent_bigrams else 0

        sent_trigrams = set(zip(sent_tokens[:-2], sent_tokens[1:-1], sent_tokens[2:])) if len(sent_tokens) > 2 else set()
        f["trigram_overlap_with_source"] = len(sent_trigrams & source_trigrams) / max(len(sent_trigrams), 1) if sent_trigrams else 0

        f["source_len_words"] = len(source_tokens)
        f["sent_to_source_ratio"] = len(sent_tokens) / max(len(source.split()), 1)

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
        ]
        f["prompt_pattern_count"] = sum(1 for p in prompt_patterns if re.search(p, sent, re.IGNORECASE))
        f["has_prompt_marker"] = int(f["prompt_pattern_count"] > 0)

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

        if i > 0:
            prev = sentences[i-1].lower()
            prev_tokens = set(prev.split())
            f["overlap_with_prev"] = len(sent_set & prev_tokens) / max(len(sent_set), 1)
            f["exact_match_prev"] = int(sent_lower == prev)
        else:
            f["overlap_with_prev"] = 0
            f["exact_match_prev"] = 0

        if i > 1:
            f["overlap_with_prev2"] = len(sent_set & set(sentences[i-2].lower().split())) / max(len(sent_set), 1)
        else:
            f["overlap_with_prev2"] = 0

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

        rep_chars = sum(1 for j in range(1, len(sent)) if sent[j] == sent[j-1])
        f["repeated_char_ratio"] = rep_chars / max(len(sent), 1)

        f["avg_word_len"] = np.mean([len(w) for w in sent_tokens]) if sent_tokens else 0
        f["max_word_len"] = max((len(w) for w in sent_tokens), default=0)
        f["digit_ratio"] = sum(1 for c in sent if c.isdigit()) / max(len(sent), 1)

        longest_match = 0
        for start in range(len(sent_tokens)):
            for end in range(start + 1, min(start + 20, len(sent_tokens) + 1)):
                substr = " ".join(sent_tokens[start:end])
                if substr in source_lower:
                    longest_match = max(longest_match, end - start)
                else:
                    break
        f["longest_source_match"] = longest_match
        f["source_match_ratio"] = longest_match / max(len(sent_tokens), 1)

        features_list.append(f)

    return features_list


def build_dataset(docs, embedding_sims=None):
    all_features = []
    all_labels = []
    all_doc_ids = []
    sim_idx = 0

    for doc in tqdm(docs, desc="Extracting features"):
        features = extract_features(doc)
        for i, f in enumerate(features):
            if embedding_sims:
                f["embedding_sim_to_source"] = embedding_sims[sim_idx]
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


def find_best_threshold(model, dev_df, feature_cols):
    probs = model.predict(dev_df[feature_cols].values)
    none_probs = probs[:, 0]
    best_f1, best_thresh = 0, 0.5

    for thresh in np.arange(0.3, 0.95, 0.005):
        preds = (none_probs < thresh).astype(int)
        doc_true, doc_pred = [], []
        for doc_id in dev_df["doc_id"].unique():
            mask = dev_df["doc_id"] == doc_id
            doc_true.append(int(any(dev_df.loc[mask, "label"].values != 0)))
            doc_pred.append(int(any(preds[mask.values] == 1)))
        f1 = f1_score(doc_true, doc_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    log.info(f"Best threshold: {best_thresh:.3f} → doc F1: {best_f1:.4f}")
    return best_thresh


def main():
    log.info("Loading embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    log.info("Loading data...")
    train_docs = load_data("train_data.json")
    dev_docs = load_data("dev_data.json")

    log.info("Computing embeddings...")
    train_sims = compute_embedding_sims(train_docs, embed_model)
    dev_sims = compute_embedding_sims(dev_docs, embed_model)

    log.info("Building features...")
    train_df = build_dataset(train_docs, train_sims)
    dev_df = build_dataset(dev_docs, dev_sims)

    feature_cols = [c for c in train_df.columns if c not in ["doc_id", "label"]]
    X_train, y_train = train_df[feature_cols].values, train_df["label"].values
    X_dev, y_dev = dev_df[feature_cols].values, dev_df["label"].values

    log.info(f"Training: {X_train.shape}, Dev: {X_dev.shape}, Features: {len(feature_cols)}")

    class_counts = Counter(y_train)
    total = len(y_train)
    sample_weights = np.array([total / (len(class_counts) * class_counts[y]) for y in y_train])

    params = {
        "objective": "multiclass", "num_class": len(LABELS), "metric": "multi_logloss",
        "learning_rate": 0.05, "num_leaves": 127, "max_depth": 10,
        "min_child_samples": 15, "subsample": 0.8, "colsample_bytree": 0.8,
        "reg_alpha": 0.1, "reg_lambda": 1.0, "verbose": -1, "n_jobs": -1, "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights, feature_name=feature_cols)
    dev_data = lgb.Dataset(X_dev, label=y_dev, reference=train_data, feature_name=feature_cols)

    model = lgb.train(params, train_data, num_boost_round=3000, valid_sets=[dev_data],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])

    dev_preds = model.predict(X_dev).argmax(axis=1)
    log.info(f"\nMacro F1: {f1_score(y_dev, dev_preds, average='macro'):.4f}")
    log.info("\n" + classification_report(y_dev, dev_preds, target_names=LABELS, zero_division=0))

    best_thresh = find_best_threshold(model, dev_df, feature_cols)

    doc_y, doc_p_multi, doc_p_binary = [], [], []
    probs_dev = model.predict(X_dev)
    for doc_id in dev_df["doc_id"].unique():
        mask = dev_df["doc_id"] == doc_id
        true_l = dev_df.loc[mask, "label"].values
        pred_l = dev_preds[mask.values]
        none_p = probs_dev[mask.values, 0]
        doc_y.append(int(any(l != 0 for l in true_l)))
        doc_p_multi.append(int(any(l != 0 for l in pred_l)))
        doc_p_binary.append(int(any(p < best_thresh for p in none_p)))

    log.info(f"Doc F1 (multi): {f1_score(doc_y, doc_p_multi):.4f}")
    log.info(f"Doc F1 (thresh={best_thresh:.3f}): {f1_score(doc_y, doc_p_binary):.4f}")

    if os.path.exists("test_data.json"):
        log.info("Generating test predictions...")
        test_docs = load_data("test_data.json")
        log.info(f"Test: {len(test_docs)} docs")
        test_sims = compute_embedding_sims(test_docs, embed_model)
        test_df = build_dataset(test_docs, test_sims)
        test_probs = model.predict(test_df[feature_cols].values)

        multi_results, binary_results = [], []
        idx = 0
        for doc in test_docs:
            n = len(doc["sentences"])
            dp = test_probs[idx:idx+n]
            multi_labels = [LABELS[p] for p in dp.argmax(axis=1)]
            binary_labels = ["Overgeneration" if dp[j, 0] < best_thresh else "None" for j in range(n)]
            idx += n
            multi_results.append({"id": doc["id"], "labels": multi_labels, "run_id": "writerslogic_Task22b_LGBMv2"})
            binary_results.append({"id": doc["id"], "labels": binary_labels, "run_id": "writerslogic_Task21b_LGBMv2"})

        with open("writerslogic_Task22b_LGBMv2.json", "w") as f:
            json.dump(multi_results, f)
        subprocess.run(["zip", "writerslogic_Task22b_LGBMv2.zip", "writerslogic_Task22b_LGBMv2.json"])

        with open("writerslogic_Task21b_LGBMv2.json", "w") as f:
            json.dump(binary_results, f)
        subprocess.run(["zip", "writerslogic_Task21b_LGBMv2.zip", "writerslogic_Task21b_LGBMv2.json"])

        log.info("Submissions ready:")
        log.info("  Multi-class: writerslogic_Task22b_LGBMv2.zip")
        log.info("  Binary: writerslogic_Task21b_LGBMv2.zip")


if __name__ == "__main__":
    main()
