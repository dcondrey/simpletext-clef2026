"""
SimpleText Task 2: Identify and classify overgeneration in simplified text.
Approach: Feature engineering + LightGBM/XGBoost ensemble (same as our winning RTD approach).

Features per sentence:
1. Text-based: length, word count, punctuation patterns
2. Source alignment: NLI entailment score, token overlap, novel token ratio
3. Pattern detection: prompt markers, repetition, structural anomalies
4. Position: sentence position in document, relative position
"""

import json
import logging
import os
import re
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
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


def extract_features(doc):
    source = doc["source"]
    sentences = doc["sentences"]
    source_tokens = set(source.lower().split())
    source_bigrams = set(zip(source.lower().split()[:-1], source.lower().split()[1:]))

    features_list = []
    n_sents = len(sentences)

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
        f["n_sentences_in_doc"] = n_sents

        if source_tokens:
            overlap = sent_set & source_tokens
            f["token_overlap_ratio"] = len(overlap) / max(len(sent_set), 1)
            f["novel_token_ratio"] = 1.0 - f["token_overlap_ratio"]
            f["novel_token_count"] = len(sent_set - source_tokens)
        else:
            f["token_overlap_ratio"] = 0
            f["novel_token_ratio"] = 1
            f["novel_token_count"] = len(sent_set)

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

        if i > 0:
            prev = sentences[i-1].lower()
            prev_tokens = set(prev.split())
            f["overlap_with_prev"] = len(sent_set & prev_tokens) / max(len(sent_set), 1)
        else:
            f["overlap_with_prev"] = 0

        if i > 1:
            prev2 = sentences[i-2].lower()
            f["overlap_with_prev2"] = len(sent_set & set(prev2.split())) / max(len(sent_set), 1)
        else:
            f["overlap_with_prev2"] = 0

        rep_chars = 0
        for j in range(1, len(sent)):
            if sent[j] == sent[j-1]:
                rep_chars += 1
        f["repeated_char_ratio"] = rep_chars / max(len(sent), 1)

        words = sent_tokens
        if len(words) > 2:
            bigrams = list(zip(words[:-1], words[1:]))
            unique_bigrams = set(bigrams)
            f["bigram_repetition"] = 1.0 - len(unique_bigrams) / max(len(bigrams), 1)
        else:
            f["bigram_repetition"] = 0

        f["contains_bracket_content"] = int(bool(re.search(r"\[.*?\]", sent)))
        f["contains_parenthetical"] = int(bool(re.search(r"\(.*?\)", sent)))
        f["ends_with_colon"] = int(sent.rstrip().endswith(":"))

        f["avg_word_len"] = np.mean([len(w) for w in sent_tokens]) if sent_tokens else 0
        f["max_word_len"] = max((len(w) for w in sent_tokens), default=0)
        f["digit_ratio"] = sum(1 for c in sent if c.isdigit()) / max(len(sent), 1)

        features_list.append(f)

    return features_list


def build_dataset(docs):
    all_features = []
    all_labels = []
    all_doc_ids = []
    all_sent_ids = []

    for doc in tqdm(docs, desc="Extracting features"):
        features = extract_features(doc)
        for i, f in enumerate(features):
            all_features.append(f)
            if "labels" in doc:
                all_labels.append(LABEL2IDX.get(doc["labels"][i], 0))
            all_doc_ids.append(doc["id"])
            all_sent_ids.append(i)

    df = pd.DataFrame(all_features)
    df["doc_id"] = all_doc_ids
    df["sent_idx"] = all_sent_ids
    if all_labels:
        df["label"] = all_labels
    return df


def train_and_evaluate(train_df, dev_df):
    feature_cols = [c for c in train_df.columns if c not in ["doc_id", "sent_idx", "label"]]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_dev = dev_df[feature_cols].values
    y_dev = dev_df["label"].values

    log.info(f"Training: {X_train.shape}, Dev: {X_dev.shape}")
    log.info(f"Features: {len(feature_cols)}")

    class_counts = Counter(y_train)
    total = len(y_train)
    class_weights = {c: total / (len(class_counts) * count) for c, count in class_counts.items()}
    sample_weights = np.array([class_weights[y] for y in y_train])

    params = {
        "objective": "multiclass",
        "num_class": len(LABELS),
        "metric": "multi_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights, feature_name=feature_cols)
    dev_data = lgb.Dataset(X_dev, label=y_dev, reference=train_data, feature_name=feature_cols)

    model = lgb.train(
        params, train_data,
        num_boost_round=2000,
        valid_sets=[dev_data],
        callbacks=[
            lgb.early_stopping(50),
            lgb.log_evaluation(100),
        ],
    )

    dev_preds = model.predict(X_dev).argmax(axis=1)

    log.info("\n=== Dev Results ===")
    log.info(f"Macro F1: {f1_score(y_dev, dev_preds, average='macro'):.4f}")
    log.info(f"Weighted F1: {f1_score(y_dev, dev_preds, average='weighted'):.4f}")
    log.info("\n" + classification_report(y_dev, dev_preds, target_names=LABELS))

    doc_y = []
    doc_pred = []
    for doc_id in dev_df["doc_id"].unique():
        mask = dev_df["doc_id"] == doc_id
        true_labels = dev_df.loc[mask, "label"].values
        pred_labels = dev_preds[mask.values]
        doc_y.append(int(any(l != 0 for l in true_labels)))
        doc_pred.append(int(any(l != 0 for l in pred_labels)))
    log.info(f"Doc-level binary F1: {f1_score(doc_y, doc_pred, average='binary'):.4f}")

    return model, feature_cols


def predict(model, test_df, feature_cols):
    X = test_df[feature_cols].values
    probs = model.predict(X)
    preds = probs.argmax(axis=1)
    return [LABELS[p] for p in preds]


def create_submission(test_docs, predictions, output_path, run_id="writerslogic_Task22b_LightGBM"):
    results = []
    idx = 0
    for doc in test_docs:
        n = len(doc["sentences"])
        doc_preds = predictions[idx:idx+n]
        idx += n
        results.append({
            "id": doc["id"],
            "labels": doc_preds,
            "run_id": run_id,
        })
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"Submission saved to {output_path}")


def main():
    log.info("Loading data...")
    train_docs = load_data("train_data.json")
    dev_docs = load_data("dev_data.json")

    log.info("Building features...")
    train_df = build_dataset(train_docs)
    dev_df = build_dataset(dev_docs)

    model, feature_cols = train_and_evaluate(train_df, dev_df)

    log.info("Generating dev submission for validation...")
    dev_preds = predict(model, dev_df, feature_cols)
    create_submission(dev_docs, dev_preds, "dev_predictions.json")

    if os.path.exists("test_data.json"):
        log.info("Loading test data...")
        test_docs = load_data("test_data.json")
        log.info(f"Test: {len(test_docs)} docs")
        test_df = build_dataset(test_docs)
        test_preds = predict(model, test_df, feature_cols)
        sub_path = "writerslogic_Task22b_LightGBM.json"
        create_submission(test_docs, test_preds, sub_path)

        import subprocess
        subprocess.run(["zip", "writerslogic_Task22b_LightGBM.zip", sub_path])
        log.info(f"Submission zip ready: writerslogic_Task22b_LightGBM.zip")

    importance = model.feature_importance(importance_type="gain")
    top_features = sorted(zip(feature_cols, importance), key=lambda x: -x[1])[:15]
    log.info("\nTop features:")
    for name, imp in top_features:
        log.info(f"  {name}: {imp:.0f}")


if __name__ == "__main__":
    main()
