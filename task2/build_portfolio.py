"""Build 5-submission portfolio by blending DeBERTa, LGB, and LogReg.

Run after DeBERTa probs are saved to cache/deberta_dev_probs.npy and cache/deberta_test_probs.npy.

Usage:
    uv run python build_portfolio.py
"""

import json
import logging
import os
import pickle
import subprocess

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from feature_extraction import LABELS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SUBMISSION_DIR = "submissions_portfolio"


def load_lgb():
    with open("models_v7/artifacts.pkl", "rb") as f:
        art = pickle.load(f)
    dev_df = pd.read_pickle("models_v7/cache/dev_df.pkl")
    test_df = pd.read_pickle("models_v7/cache/test_df.pkl")
    for c in art["feature_cols"]:
        if c not in dev_df.columns:
            dev_df[c] = 0.0
        if c not in test_df.columns:
            test_df[c] = 0.0
    dev_probs = np.mean([m.predict(dev_df[art["feature_cols"]].values) for m in art["bin_models"]], axis=0)
    test_probs = np.mean([m.predict(test_df[art["feature_cols"]].values) for m in art["bin_models"]], axis=0)
    return dev_probs, test_probs


def load_deberta():
    dev_probs = np.load("cache/deberta_dev_probs.npy")
    test_probs = np.load("cache/deberta_test_probs.npy")
    return dev_probs, test_probs


def train_logreg():
    log.info("Training LogReg on TF-IDF...")
    train_docs = json.load(open("train_data.json"))
    dev_docs = json.load(open("dev_data.json"))
    test_docs = json.load(open("test_data.json"))

    def flatten_text(docs):
        sents, sources, combined = [], [], []
        for d in docs:
            for s in d["sentences"]:
                sents.append(s)
                sources.append(d["source"][:1500])
                combined.append(f"{d['source'][:1500]} [SEP] {s}")
        return sents, sources, combined

    train_s, _, train_c = flatten_text(train_docs)
    dev_s, _, dev_c = flatten_text(dev_docs)
    test_s, _, test_c = flatten_text(test_docs)

    train_y = np.array([0 if l == "None" else 1 for d in train_docs for l in d["labels"]])

    tfidf_s = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), max_features=30000,
                               min_df=3, sublinear_tf=True, dtype=np.float32)
    tfidf_c = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), max_features=20000,
                               min_df=3, sublinear_tf=True, dtype=np.float32)

    X_train = hstack([tfidf_s.fit_transform(train_s), tfidf_c.fit_transform(train_c)])
    X_dev = hstack([tfidf_s.transform(dev_s), tfidf_c.transform(dev_c)])
    X_test = hstack([tfidf_s.transform(test_s), tfidf_c.transform(test_c)])

    lr = LogisticRegression(C=1.0, max_iter=2000, solver="saga", class_weight="balanced")
    lr.fit(X_train, train_y)

    dev_probs = lr.predict_proba(X_dev)[:, 1]
    test_probs = lr.predict_proba(X_test)[:, 1]
    log.info("LogReg trained.")
    return dev_probs, test_probs


def eval_doc_f1(probs, doc_ids, gt_doc, threshold):
    df = pd.DataFrame({"doc_id": doc_ids, "prob": probs})
    dp = {did: int(g["prob"].max() > threshold) for did, g in df.groupby("doc_id")}
    y_t = [gt_doc[d] for d in gt_doc]
    y_p = [dp.get(d, 0) for d in gt_doc]
    return f1_score(y_t, y_p)


def best_threshold(probs, doc_ids, gt_doc):
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.10, 0.90, 0.005):
        f1 = eval_doc_f1(probs, doc_ids, gt_doc, t)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_f1, best_t


def make_submission(test_docs, bin_probs, doc_thresh, sent_thresh, run_name):
    mc_art = pickle.load(open("models_v7/artifacts.pkl", "rb"))
    test_df = pd.read_pickle("models_v7/cache/test_df.pkl")
    for c in mc_art["feature_cols"]:
        if c not in test_df.columns:
            test_df[c] = 0.0
    mc_probs = np.mean([m.predict(test_df[mc_art["feature_cols"]].values) for m in mc_art["mc_models"]], axis=0)

    r21, r22 = [], []
    idx = 0
    for doc in test_docs:
        n = len(doc["sentences"])
        s_bin = bin_probs[idx:idx + n]
        s_mc = mc_probs[idx:idx + n]
        idx += n

        if s_bin.max() > doc_thresh:
            bl = ["Overgeneration" if p > sent_thresh else "None" for p in s_bin]
            if all(l == "None" for l in bl):
                bl[int(np.argmax(s_bin))] = "Overgeneration"
        else:
            bl = ["None"] * n

        ml = []
        for j, b in enumerate(bl):
            if b != "None":
                pc = int(np.argmax(s_mc[j]))
                if pc == 0:
                    pc = int(np.argmax(s_mc[j, 1:])) + 1
                ml.append(LABELS[pc])
            else:
                ml.append("None")

        r21.append({"id": doc["id"], "labels": bl, "run_id": f"writerslogic_Task21b_{run_name}"})
        r22.append({"id": doc["id"], "labels": ml, "run_id": f"writerslogic_Task22b_{run_name}"})
    return r21, r22


def save(results, filename):
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    jpath = os.path.join(SUBMISSION_DIR, filename)
    zpath = jpath.replace(".json", ".zip")
    with open(jpath, "w") as f:
        json.dump(results, f)
    subprocess.run(["zip", "-j", zpath, jpath], check=True, capture_output=True)
    pos = sum(1 for d in results if any(l != "None" for l in d["labels"]))
    log.info("  %s: %d pos docs", filename, pos)


def main():
    dev_docs = json.load(open("dev_data.json"))
    test_docs = json.load(open("test_data.json"))

    dev_doc_ids = [d["id"] for d in dev_docs for _ in d["sentences"]]
    gt_doc = {d["id"]: int(any(l != "None" for l in d["labels"])) for d in dev_docs}

    log.info("Loading LGB probabilities...")
    lgb_dev, lgb_test = load_lgb()

    has_deberta = os.path.exists("cache/deberta_dev_probs.npy")
    if has_deberta:
        log.info("Loading DeBERTa probabilities...")
        deb_dev, deb_test = load_deberta()
    else:
        log.info("No DeBERTa probs yet, skipping DeBERTa submissions")

    log.info("Training LogReg...")
    lr_dev, lr_test = train_logreg()

    log.info("\n=== Dev Evaluation ===")

    f1, t = best_threshold(lgb_dev, dev_doc_ids, gt_doc)
    log.info("LGB alone:        dev F1=%.4f t=%.3f", f1, t)
    lgb_t = t

    f1, t = best_threshold(lr_dev, dev_doc_ids, gt_doc)
    log.info("LogReg alone:     dev F1=%.4f t=%.3f", f1, t)

    for alpha in [0.7, 0.8, 0.85, 0.9]:
        blend = alpha * lgb_dev + (1 - alpha) * lr_dev
        f1, t = best_threshold(blend, dev_doc_ids, gt_doc)
        log.info("LGB+LR a=%.2f:    dev F1=%.4f t=%.3f", alpha, f1, t)

    if has_deberta:
        f1, t = best_threshold(deb_dev, dev_doc_ids, gt_doc)
        log.info("DeBERTa alone:    dev F1=%.4f t=%.3f", f1, t)
        deb_t = t

        for alpha in [0.3, 0.4, 0.5, 0.6, 0.7]:
            blend = alpha * deb_dev + (1 - alpha) * lgb_dev
            f1, t = best_threshold(blend, dev_doc_ids, gt_doc)
            log.info("DEB+LGB a=%.2f:   dev F1=%.4f t=%.3f", alpha, f1, t)

        for da, la, lra in [(0.5, 0.35, 0.15), (0.4, 0.4, 0.2), (0.6, 0.3, 0.1)]:
            blend = da * deb_dev + la * lgb_dev + lra * lr_dev
            f1, t = best_threshold(blend, dev_doc_ids, gt_doc)
            log.info("Triple %.1f/%.1f/%.1f: dev F1=%.4f t=%.3f", da, la, lra, f1, t)

    log.info("\n=== Generating Submissions ===")

    # Sub 1: LGB v9 with optimal threshold
    blend_lgb_lr = 0.8 * lgb_test + 0.2 * lr_test
    f1_lgblr, t_lgblr = best_threshold(0.8 * lgb_dev + 0.2 * lr_dev, dev_doc_ids, gt_doc)
    r21, r22 = make_submission(test_docs, lgb_test, lgb_t, 0.78, "portfolio_lgb")
    save(r21, "writerslogic_Task21b_portfolio_lgb.json")
    save(r22, "writerslogic_Task22b_portfolio_lgb.json")

    # Sub 2: LGB + LogReg blend
    r21, r22 = make_submission(test_docs, blend_lgb_lr, t_lgblr, 0.78, "portfolio_lgblr")
    save(r21, "writerslogic_Task21b_portfolio_lgblr.json")
    save(r22, "writerslogic_Task22b_portfolio_lgblr.json")

    if has_deberta:
        # Sub 3: DeBERTa alone
        r21, r22 = make_submission(test_docs, deb_test, deb_t, deb_t, "portfolio_deberta")
        save(r21, "writerslogic_Task21b_portfolio_deberta.json")
        save(r22, "writerslogic_Task22b_portfolio_deberta.json")

        # Sub 4: DeBERTa + LGB best blend
        best_blend_f1, best_alpha, best_t_blend = 0, 0.5, 0.5
        for alpha in np.arange(0.3, 0.8, 0.05):
            blend = alpha * deb_dev + (1 - alpha) * lgb_dev
            f1, t = best_threshold(blend, dev_doc_ids, gt_doc)
            if f1 > best_blend_f1:
                best_blend_f1, best_alpha, best_t_blend = f1, alpha, t

        blend_test = best_alpha * deb_test + (1 - best_alpha) * lgb_test
        r21, r22 = make_submission(test_docs, blend_test, best_t_blend, best_t_blend, "portfolio_deblgb")
        save(r21, "writerslogic_Task21b_portfolio_deblgb.json")
        save(r22, "writerslogic_Task22b_portfolio_deblgb.json")
        log.info("  DEB+LGB blend alpha=%.2f t=%.3f dev F1=%.4f", best_alpha, best_t_blend, best_blend_f1)

        # Sub 5: Triple blend
        best_triple_f1 = 0
        best_weights = (0.5, 0.35, 0.15)
        best_t_triple = 0.5
        for da in np.arange(0.3, 0.7, 0.05):
            for la in np.arange(0.2, 0.6, 0.05):
                lra = 1.0 - da - la
                if lra < 0.05 or lra > 0.4:
                    continue
                blend = da * deb_dev + la * lgb_dev + lra * lr_dev
                f1, t = best_threshold(blend, dev_doc_ids, gt_doc)
                if f1 > best_triple_f1:
                    best_triple_f1 = f1
                    best_weights = (da, la, lra)
                    best_t_triple = t

        da, la, lra = best_weights
        blend_test = da * deb_test + la * lgb_test + lra * lr_test
        r21, r22 = make_submission(test_docs, blend_test, best_t_triple, best_t_triple, "portfolio_triple")
        save(r21, "writerslogic_Task21b_portfolio_triple.json")
        save(r22, "writerslogic_Task22b_portfolio_triple.json")
        log.info("  Triple blend %.2f/%.2f/%.2f t=%.3f dev F1=%.4f",
                 da, la, lra, best_t_triple, best_triple_f1)

    log.info("\nAll submissions in %s/", SUBMISSION_DIR)


if __name__ == "__main__":
    main()
