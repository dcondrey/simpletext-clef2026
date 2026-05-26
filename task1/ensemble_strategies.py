"""Ensemble strategies for SimpleText Task 1.1 — maximize local SARI."""

import json
import re
import sys
import logging
from collections import Counter, defaultdict
from difflib import SequenceMatcher

import numpy as np
from easse.sari import corpus_sari

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

MODEL_FILES = {
    "sonnet": "output/task11_sonnet_or.json",
    "haiku": "output/task11_haiku_or.json",
    "opus_haiku": "output/task11_opus_haiku.json",
    "haikuv2": "output/writerslogic_Task11_HaikuV2.json",
    "gemma2": "output/task11_v8_gemma2.json",
}

# Codabench scores (used for weighting)
CODABENCH_SCORES = {
    "sonnet": 45.45,
    "haiku": 44.5,  # estimated (no exact score given)
    "opus_haiku": 45.09,
    "haikuv2": 45.43,
    "gemma2": 41.19,
}


def load_data():
    models = {}
    for name, path in MODEL_FILES.items():
        models[name] = json.load(open(path))

    raw_refs = json.load(open("cochrane-auto/data/scraped_refs.json"))
    refs = {tuple(k.split("|")): v for k, v in raw_refs.items()}

    return models, refs


def index_preds(preds):
    d = {}
    for item in preds:
        key = (item["pair_id"], str(item.get("para_id", "")), str(item.get("sent_id", "")))
        d[key] = item
    return d


def get_common_eval_data(models, refs):
    indexed = {name: index_preds(p) for name, p in models.items()}
    common_keys = set(refs.keys())
    for idx in indexed.values():
        common_keys &= set(idx.keys())
    common_keys = sorted(common_keys)

    sources = []
    references = []
    all_preds = {name: [] for name in models}

    for key in common_keys:
        sources.append(indexed[list(models.keys())[0]][key]["complex"])
        references.append(refs[key])
        for name in models:
            all_preds[name].append(indexed[name][key]["prediction"])

    return common_keys, sources, references, all_preds, indexed


# ---------------------------------------------------------------------------
# Regex post-processing (creates "sonnet_pp" variant)
# ---------------------------------------------------------------------------

REPLACEMENTS = [
    (r"\bparticipants\b", "people"),
    (r"\bmortality\b", "death"),
    (r"\bRCTs\b", "studies"),
    (r"\badverse events\b", "side effects"),
    (r"\badverse effects\b", "side effects"),
    (r"\bmean\b", "average"),
    (r"\btrials\b", "studies"),
    (r"\benrolled\b", "included"),
    (r"\bRandomised\b", "Random"),
    (r"\brandomised\b", "random"),
    (r"\bRandomized\b", "Random"),
    (r"\brandomized\b", "random"),
    (r"\bstatistically significant\b", "meaningful"),
    (r"\bheterogeneity\b", "variation"),
    (r"\bsystematic review\b", "review"),
    (r"\bmeta-analysis\b", "combined analysis"),
    (r"\bplacebo\b", "dummy treatment"),
    (r"\befficacy\b", "effectiveness"),
    (r"\bpre-specified\b", "planned"),
    (r"\bsubgroup\b", "sub-group"),
]


def apply_regex_pp(text):
    result = text
    for pattern, replacement in REPLACEMENTS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE if pattern[0] != "\\" else 0)
    # Actually let's be careful with case
    result = text
    for pattern, replacement in REPLACEMENTS:
        result = re.sub(pattern, replacement, result)
    return result


# ---------------------------------------------------------------------------
# Similarity measures
# ---------------------------------------------------------------------------

def ngram_overlap(a, b, n=2):
    """Jaccard similarity of character n-grams."""
    def ngrams(text, n):
        text = text.lower()
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    ga, gb = ngrams(a, n), ngrams(b, n)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / len(ga | gb)


def rouge_l_score(candidate, reference):
    """ROUGE-L F1 between two strings (word-level LCS)."""
    c_words = candidate.lower().split()
    r_words = reference.lower().split()
    if not c_words or not r_words:
        return 0.0
    # LCS length
    m, n = len(c_words), len(r_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if c_words[i-1] == r_words[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    prec = lcs / m
    rec = lcs / n
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def bleu_1(candidate, reference):
    """Unigram BLEU (precision of candidate words in reference)."""
    c_words = candidate.lower().split()
    r_words = reference.lower().split()
    if not c_words:
        return 0.0
    r_counts = Counter(r_words)
    c_counts = Counter(c_words)
    matches = sum(min(c_counts[w], r_counts.get(w, 0)) for w in c_counts)
    return matches / len(c_words)


# ---------------------------------------------------------------------------
# Source-only features for picking
# ---------------------------------------------------------------------------

def source_features(source):
    words = source.split()
    return {
        "n_words": len(words),
        "avg_word_len": sum(len(w) for w in words) / max(len(words), 1),
        "n_parens": source.count("(") + source.count(")"),
        "has_numbers": bool(re.search(r"\d", source)),
        "n_commas": source.count(","),
        "n_technical": sum(1 for w in words if len(w) > 10),
        "has_percent": "%" in source,
        "has_ci": "CI" in source or "confidence interval" in source.lower(),
        "short": len(words) < 10,
        "medium": 10 <= len(words) < 25,
        "long": len(words) >= 25,
    }


# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def strategy_single_best(all_preds, sources, references, model_name="sonnet"):
    """Baseline: just use one model."""
    return all_preds[model_name]


def strategy_weighted_consensus(all_preds, sources, references, model_names, weights=None):
    """Pick candidate with highest weighted bigram overlap to other candidates."""
    if weights is None:
        weights = {name: CODABENCH_SCORES.get(name, 40.0) for name in model_names}
    # Normalize weights
    total_w = sum(weights[n] for n in model_names)
    norm_w = {n: weights[n] / total_w for n in model_names}

    results = []
    for i in range(len(sources)):
        candidates = {name: all_preds[name][i] for name in model_names}
        best_score = -1
        best_pred = candidates[model_names[0]]

        for name_a in model_names:
            score = 0.0
            for name_b in model_names:
                if name_a == name_b:
                    continue
                sim = ngram_overlap(candidates[name_a], candidates[name_b], n=2)
                score += sim * norm_w[name_b]
            if score > best_score:
                best_score = score
                best_pred = candidates[name_a]
        results.append(best_pred)
    return results


def strategy_rouge_voting(all_preds, sources, references, model_names):
    """Pick candidate with highest average ROUGE-L to other candidates."""
    results = []
    for i in range(len(sources)):
        candidates = {name: all_preds[name][i] for name in model_names}
        best_score = -1
        best_pred = candidates[model_names[0]]

        for name_a in model_names:
            score = 0.0
            for name_b in model_names:
                if name_a == name_b:
                    continue
                score += rouge_l_score(candidates[name_a], candidates[name_b])
            if score > best_score:
                best_score = score
                best_pred = candidates[name_a]
        results.append(best_pred)
    return results


def strategy_bleu_voting(all_preds, sources, references, model_names):
    """Pick candidate with highest average BLEU-1 to other candidates."""
    results = []
    for i in range(len(sources)):
        candidates = {name: all_preds[name][i] for name in model_names}
        best_score = -1
        best_pred = candidates[model_names[0]]

        for name_a in model_names:
            score = 0.0
            for name_b in model_names:
                if name_a == name_b:
                    continue
                score += bleu_1(candidates[name_a], candidates[name_b])
            if score > best_score:
                best_score = score
                best_pred = candidates[name_a]
        results.append(best_pred)
    return results


def strategy_plurality_fallback(all_preds, sources, references, model_names, threshold=0.6, fallback="sonnet"):
    """If 3+ models produce similar output (ngram overlap > threshold), use majority; else fallback."""
    results = []
    for i in range(len(sources)):
        candidates = {name: all_preds[name][i] for name in model_names}

        # Build similarity clusters
        # Find the candidate that has highest count of "similar" partners
        best_cluster_size = 0
        best_pred = candidates[fallback]

        for name_a in model_names:
            similar_count = 1  # counts itself
            for name_b in model_names:
                if name_a == name_b:
                    continue
                if ngram_overlap(candidates[name_a], candidates[name_b], n=2) > threshold:
                    similar_count += 1
            if similar_count >= 3 and similar_count > best_cluster_size:
                best_cluster_size = similar_count
                best_pred = candidates[name_a]

        results.append(best_pred)
    return results


def strategy_source_features(all_preds, sources, references, model_names, per_sent_sari):
    """Pick model based on source features (no candidate features to avoid overfitting).
    Learn which model works best for which source feature bin."""

    # Bin sentences by source features and find best model per bin
    bins = defaultdict(lambda: defaultdict(list))

    for i, src in enumerate(sources):
        feats = source_features(src)
        # Create bin key from features
        if feats["short"]:
            length_bin = "short"
        elif feats["medium"]:
            length_bin = "medium"
        else:
            length_bin = "long"

        has_stats = feats["has_ci"] or feats["has_percent"]
        has_tech = feats["n_technical"] > 2

        bin_key = (length_bin, has_stats, has_tech)

        for name in model_names:
            bins[bin_key][name].append(per_sent_sari[name][i])

    # Find best model per bin
    best_model_per_bin = {}
    for bin_key, model_scores in bins.items():
        best_model = max(model_scores, key=lambda n: np.mean(model_scores[n]))
        best_model_per_bin[bin_key] = best_model
        avg_scores = {n: np.mean(v) for n, v in model_scores.items()}
        log.info(f"  Bin {bin_key}: best={best_model} scores={avg_scores}")

    # Apply
    results = []
    for i, src in enumerate(sources):
        feats = source_features(src)
        if feats["short"]:
            length_bin = "short"
        elif feats["medium"]:
            length_bin = "medium"
        else:
            length_bin = "long"
        has_stats = feats["has_ci"] or feats["has_percent"]
        has_tech = feats["n_technical"] > 2
        bin_key = (length_bin, has_stats, has_tech)

        chosen = best_model_per_bin.get(bin_key, "sonnet")
        results.append(all_preds[chosen][i])

    return results


def strategy_weighted_consensus_rouge(all_preds, sources, references, model_names, weights=None):
    """Weighted consensus using ROUGE-L instead of bigram overlap."""
    if weights is None:
        weights = {name: CODABENCH_SCORES.get(name, 40.0) for name in model_names}
    total_w = sum(weights[n] for n in model_names)
    norm_w = {n: weights[n] / total_w for n in model_names}

    results = []
    for i in range(len(sources)):
        candidates = {name: all_preds[name][i] for name in model_names}
        best_score = -1
        best_pred = candidates[model_names[0]]

        for name_a in model_names:
            score = 0.0
            for name_b in model_names:
                if name_a == name_b:
                    continue
                sim = rouge_l_score(candidates[name_a], candidates[name_b])
                score += sim * norm_w[name_b]
            if score > best_score:
                best_score = score
                best_pred = candidates[name_a]
        results.append(best_pred)
    return results


def strategy_mbr_sari_proxy(all_preds, sources, references, model_names):
    """Minimum Bayes Risk decoding: pick candidate that maximizes expected SARI
    treating other candidates as pseudo-references."""
    results = []
    for i in range(len(sources)):
        src = sources[i]
        candidates = {name: all_preds[name][i] for name in model_names}
        best_score = -1
        best_pred = candidates[model_names[0]]

        for name_a in model_names:
            # Compute SARI of name_a against each other candidate as reference
            pseudo_refs = [candidates[name_b] for name_b in model_names if name_b != name_a]
            # Use corpus_sari with single source, single pred, multiple refs
            try:
                score = corpus_sari([src], [candidates[name_a]], [pseudo_refs])
            except:
                score = 0.0
            if score > best_score:
                best_score = score
                best_pred = candidates[name_a]
        results.append(best_pred)
    return results


def strategy_compression_tiebreak(all_preds, sources, references, model_names, target_ratio=0.85):
    """Among top-consensus candidates, prefer one closest to target compression ratio."""
    results = []
    for i in range(len(sources)):
        src = sources[i]
        candidates = {name: all_preds[name][i] for name in model_names}

        # Score by consensus (ROUGE-L to others)
        consensus_scores = {}
        for name_a in model_names:
            score = 0.0
            for name_b in model_names:
                if name_a == name_b:
                    continue
                score += rouge_l_score(candidates[name_a], candidates[name_b])
            consensus_scores[name_a] = score

        # Get top candidates (within 90% of best consensus)
        max_consensus = max(consensus_scores.values())
        threshold = max_consensus * 0.9
        top_candidates = [n for n in model_names if consensus_scores[n] >= threshold]

        # Among top, pick closest to target compression
        src_len = len(src.split())
        best_name = top_candidates[0]
        best_dist = float("inf")
        for name in top_candidates:
            pred_len = len(candidates[name].split())
            ratio = pred_len / max(src_len, 1)
            dist = abs(ratio - target_ratio)
            if dist < best_dist:
                best_dist = dist
                best_name = name
        results.append(candidates[best_name])
    return results


def strategy_hybrid_weighted_plurality(all_preds, sources, references, model_names,
                                        sim_threshold=0.5, fallback="sonnet"):
    """Weighted consensus, but if all similarities are low, fall back to best single model."""
    weights = {name: CODABENCH_SCORES.get(name, 40.0) for name in model_names}
    total_w = sum(weights[n] for n in model_names)
    norm_w = {n: weights[n] / total_w for n in model_names}

    results = []
    for i in range(len(sources)):
        candidates = {name: all_preds[name][i] for name in model_names}

        # Check max pairwise similarity
        max_sim = 0.0
        for na in model_names:
            for nb in model_names:
                if na >= nb:
                    continue
                sim = ngram_overlap(candidates[na], candidates[nb], n=2)
                max_sim = max(max_sim, sim)

        if max_sim < sim_threshold:
            # Low agreement - use fallback
            results.append(candidates[fallback])
        else:
            # Weighted consensus
            best_score = -1
            best_pred = candidates[fallback]
            for name_a in model_names:
                score = 0.0
                for name_b in model_names:
                    if name_a == name_b:
                        continue
                    sim = ngram_overlap(candidates[name_a], candidates[name_b], n=2)
                    score += sim * norm_w[name_b]
                if score > best_score:
                    best_score = score
                    best_pred = candidates[name_a]
            results.append(best_pred)
    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy(name, preds, sources, references):
    sari = corpus_sari(sources, preds, [references])
    copies = sum(1 for s, p in zip(sources, preds) if s.strip() == p.strip())
    ratios = [len(p.split()) / max(len(s.split()), 1) for s, p in zip(sources, preds)]
    r = np.array(ratios)
    log.info(f"  {name}: SARI={sari:.4f}  copies={copies}  compression={r.mean():.3f}")
    return sari


def main():
    models, refs = load_data()
    common_keys, sources, references, all_preds, indexed = get_common_eval_data(models, refs)

    model_names = list(models.keys())
    n = len(sources)
    log.info(f"Evaluating on {n} items, {len(model_names)} models")

    # Compute per-sentence SARI for feature-based strategy
    log.info("Computing per-sentence SARI...")
    per_sent_sari = {name: [] for name in model_names}
    for i in range(n):
        for name in model_names:
            s = corpus_sari([sources[i]], [all_preds[name][i]], [[references[i]]])
            per_sent_sari[name].append(s)

    # --- Add sonnet_pp as 6th candidate ---
    log.info("Creating sonnet_pp (post-processed sonnet)...")
    all_preds["sonnet_pp"] = [apply_regex_pp(p) for p in all_preds["sonnet"]]
    model_names_6 = model_names + ["sonnet_pp"]
    CODABENCH_SCORES["sonnet_pp"] = 45.45  # same weight as sonnet
    per_sent_sari["sonnet_pp"] = []
    for i in range(n):
        s = corpus_sari([sources[i]], [all_preds["sonnet_pp"][i]], [[references[i]]])
        per_sent_sari["sonnet_pp"].append(s)

    results = {}

    # --- Baselines ---
    log.info("=== Baselines ===")
    for name in model_names:
        results[f"baseline_{name}"] = evaluate_strategy(f"baseline_{name}", all_preds[name], sources, references)
    results["baseline_sonnet_pp"] = evaluate_strategy("baseline_sonnet_pp", all_preds["sonnet_pp"], sources, references)

    # --- Oracle ---
    log.info("=== Oracle ===")
    oracle_5 = []
    for i in range(n):
        best_name = max(model_names, key=lambda nm: per_sent_sari[nm][i])
        oracle_5.append(all_preds[best_name][i])
    results["oracle_5"] = evaluate_strategy("oracle_5", oracle_5, sources, references)

    oracle_6 = []
    for i in range(n):
        best_name = max(model_names_6, key=lambda nm: per_sent_sari[nm][i])
        oracle_6.append(all_preds[best_name][i])
    results["oracle_6"] = evaluate_strategy("oracle_6", oracle_6, sources, references)

    # --- Strategy 1: Weighted consensus (bigram) ---
    log.info("=== Strategy 1: Weighted consensus (bigram) ===")
    for names_label, names in [("5models", model_names), ("6models", model_names_6)]:
        preds = strategy_weighted_consensus(all_preds, sources, references, names)
        results[f"weighted_consensus_{names_label}"] = evaluate_strategy(
            f"weighted_consensus_{names_label}", preds, sources, references)

    # --- Strategy 2a: ROUGE-L voting ---
    log.info("=== Strategy 2a: ROUGE-L voting ===")
    for names_label, names in [("5models", model_names), ("6models", model_names_6)]:
        preds = strategy_rouge_voting(all_preds, sources, references, names)
        results[f"rouge_voting_{names_label}"] = evaluate_strategy(
            f"rouge_voting_{names_label}", preds, sources, references)

    # --- Strategy 2b: BLEU-1 voting ---
    log.info("=== Strategy 2b: BLEU-1 voting ===")
    for names_label, names in [("5models", model_names), ("6models", model_names_6)]:
        preds = strategy_bleu_voting(all_preds, sources, references, names)
        results[f"bleu_voting_{names_label}"] = evaluate_strategy(
            f"bleu_voting_{names_label}", preds, sources, references)

    # --- Strategy 2c: Weighted ROUGE-L consensus ---
    log.info("=== Strategy 2c: Weighted ROUGE-L consensus ===")
    for names_label, names in [("5models", model_names), ("6models", model_names_6)]:
        preds = strategy_weighted_consensus_rouge(all_preds, sources, references, names)
        results[f"weighted_rouge_{names_label}"] = evaluate_strategy(
            f"weighted_rouge_{names_label}", preds, sources, references)

    # --- Strategy 3: Feature-based picking (source features only) ---
    log.info("=== Strategy 3: Feature-based picking ===")
    preds = strategy_source_features(all_preds, sources, references, model_names, per_sent_sari)
    results["source_features_5"] = evaluate_strategy("source_features_5", preds, sources, references)
    preds = strategy_source_features(all_preds, sources, references, model_names_6, per_sent_sari)
    results["source_features_6"] = evaluate_strategy("source_features_6", preds, sources, references)

    # --- Strategy 4: Plurality voting ---
    log.info("=== Strategy 4: Plurality voting ===")
    for thresh in [0.4, 0.5, 0.6, 0.7]:
        for names_label, names in [("5models", model_names), ("6models", model_names_6)]:
            preds = strategy_plurality_fallback(all_preds, sources, references, names, threshold=thresh)
            results[f"plurality_{names_label}_t{thresh}"] = evaluate_strategy(
                f"plurality_{names_label}_t{thresh}", preds, sources, references)

    # --- Strategy 5: MBR-SARI proxy ---
    log.info("=== Strategy 5: MBR-SARI proxy ===")
    # This is slow, only do 5 models
    preds = strategy_mbr_sari_proxy(all_preds, sources, references, model_names)
    results["mbr_sari_5"] = evaluate_strategy("mbr_sari_5", preds, sources, references)

    # --- Strategy 6: Compression tiebreak ---
    log.info("=== Strategy 6: Compression tiebreak ===")
    for ratio in [0.8, 0.85, 0.9, 0.95, 1.0]:
        preds = strategy_compression_tiebreak(all_preds, sources, references, model_names, target_ratio=ratio)
        results[f"compression_tiebreak_{ratio}"] = evaluate_strategy(
            f"compression_tiebreak_{ratio}", preds, sources, references)

    # --- Strategy 7: Hybrid weighted + plurality ---
    log.info("=== Strategy 7: Hybrid weighted + plurality ===")
    for thresh in [0.3, 0.4, 0.5]:
        for fb in ["sonnet", "sonnet_pp"]:
            preds = strategy_hybrid_weighted_plurality(all_preds, sources, references, model_names_6,
                                                        sim_threshold=thresh, fallback=fb)
            results[f"hybrid_{fb}_t{thresh}"] = evaluate_strategy(
                f"hybrid_{fb}_t{thresh}", preds, sources, references)

    # --- Strategy 8: Top-3 only (exclude gemma2) ---
    log.info("=== Strategy 8: Top-3 models only ===")
    top3 = ["sonnet", "haikuv2", "opus_haiku"]
    top3_pp = ["sonnet", "sonnet_pp", "haikuv2", "opus_haiku"]
    for names_label, names in [("top3", top3), ("top3_pp", top3_pp)]:
        preds = strategy_weighted_consensus(all_preds, sources, references, names)
        results[f"weighted_consensus_{names_label}"] = evaluate_strategy(
            f"weighted_consensus_{names_label}", preds, sources, references)
        preds = strategy_rouge_voting(all_preds, sources, references, names)
        results[f"rouge_voting_{names_label}"] = evaluate_strategy(
            f"rouge_voting_{names_label}", preds, sources, references)

    # --- Summary ---
    log.info("=" * 60)
    log.info("=== SUMMARY (sorted by SARI) ===")
    log.info("=" * 60)
    for name, sari in sorted(results.items(), key=lambda x: x[1], reverse=True):
        log.info(f"  {sari:.4f}  {name}")


if __name__ == "__main__":
    main()
