"""
Feature extraction module for SimpleText Task 2 v5.

Computes 26+ features per sentence, grouped by:
1. Source-sentence alignment (lexical, n-gram, embedding, per-source-sentence max)
2. Stylistic and structural divergence
3. Explicit overgeneration signals (length, entities, numbers, hedging, assistant phrases)
4. Category-specific signals
5. Sequential/contextual features
"""

import logging
import math
import re
import zlib
from collections import Counter
from typing import Optional

import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


# ============================================================================
# Utility functions
# ============================================================================

def compression_ratio(text: str) -> float:
    if not text or len(text) < 5:
        return 1.0
    raw = text.encode("utf-8")
    compressed = zlib.compress(raw, level=9)
    return len(compressed) / max(len(raw), 1)


def char_entropy(text: str) -> float:
    if not text:
        return 0.0
    freq = Counter(text)
    total = len(text)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def split_into_sentences_simple(text: str) -> list[str]:
    """Split source abstract into sentences using simple heuristics."""
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [p.strip() for p in parts if p.strip()]


def extract_numbers(text: str) -> set[str]:
    """Extract all numbers, percentages, and numeric expressions."""
    patterns = [
        r'\d+\.?\d*%',
        r'\d+\.?\d*',
        r'p\s*[=<>]\s*\d+\.?\d*',
        r'\d+/\d+',
    ]
    found = set()
    for p in patterns:
        for m in re.finditer(p, text, re.IGNORECASE):
            found.add(m.group().strip())
    return found


def extract_entities_simple(text: str) -> set[str]:
    """Extract capitalized multi-word phrases as proxy for named entities."""
    entities = set()
    for m in re.finditer(r'(?<!^)(?<!\. )([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', text):
        entities.add(m.group().lower())
    for m in re.finditer(r'\b[A-Z]{2,}\b', text):
        entities.add(m.group())
    return entities


# ============================================================================
# Prompt/instruction leak patterns
# ============================================================================

PROMPT_PATTERNS = [
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

STRONG_LEAK_PATTERNS = [
    r"\bInput\s*:", r"\bOutput\s*:", r"\bSource\s*:", r"\bTarget\s*:",
    r"\bAssistant\s*:", r"\bUser\s*:", r"\bSystem\s*:",
    r"\bInstruction\s*:", r"\bPrompt\s*:", r"\bTask\s*:",
    r"\bAs\s+an\s+AI\b", r"\bAs\s+a\s+language\s+model\b",
    r"\bI'm\s+an\s+AI\b", r"\bI\s+am\s+an?\s+AI\b",
]

HEDGING_PATTERNS = [
    r"\bmay\b", r"\bmight\b", r"\bcould\s+potentially\b",
    r"\bit\s+is\s+possible\b", r"\bpossibly\b", r"\bperhaps\b",
    r"\bseems?\s+to\b", r"\bappears?\s+to\b", r"\bsuggests?\b",
    r"\bgenerally\b", r"\btypically\b", r"\busually\b",
]

CAUSAL_PATTERNS = [
    r"\bcauses?\b", r"\bleads?\s+to\b", r"\bresults?\s+in\b",
    r"\bdue\s+to\b", r"\bbecause\s+of\b", r"\bcontributes?\s+to\b",
    r"\bprevents?\b", r"\breduces?\b", r"\bincreases?\b",
]


# ============================================================================
# Core feature extraction
# ============================================================================

def extract_features_for_doc(doc: dict) -> list[dict]:
    """Extract all features for sentences in a single document.

    Returns a list of feature dictionaries (one per sentence).
    Does NOT include embedding/NLI features (those are added separately).
    """
    source = doc["source"]
    sentences = doc["sentences"]
    source_lower = source.lower()
    source_tokens = source_lower.split()
    source_set = set(source_tokens)
    n_sents = len(sentences)

    # Source sentence-level decomposition
    source_sents = split_into_sentences_simple(source)
    source_sent_tokens = [set(s.lower().split()) for s in source_sents]
    source_sent_bigrams = [
        set(zip(s.lower().split()[:-1], s.lower().split()[1:])) if len(s.split()) > 1 else set()
        for s in source_sents
    ]

    source_bigrams = set(zip(source_tokens[:-1], source_tokens[1:])) if len(source_tokens) > 1 else set()
    source_trigrams = set(zip(source_tokens[:-2], source_tokens[1:-1], source_tokens[2:])) if len(source_tokens) > 2 else set()

    # Source-level entity and number extraction
    source_numbers = extract_numbers(source)
    source_entities = extract_entities_simple(source)

    # Document-level stats
    all_sent_lens = [len(s.split()) for s in sentences]
    doc_mean_len = np.mean(all_sent_lens) if all_sent_lens else 0
    doc_std_len = np.std(all_sent_lens) if len(all_sent_lens) > 1 else 0
    doc_total_words = sum(all_sent_lens)

    # Pre-compute per-sentence data
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

        # ==================================================================
        # GROUP 1: Source-sentence alignment features
        # ==================================================================

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

        # Per-source-sentence max alignment
        max_sent_overlap = 0.0
        max_sent_bigram_overlap = 0.0
        best_source_sent_idx = 0
        for j, src_st in enumerate(source_sent_tokens):
            if not src_st:
                continue
            ov = len(sent_set & src_st) / max(len(sent_set), 1)
            if ov > max_sent_overlap:
                max_sent_overlap = ov
                best_source_sent_idx = j
            if sent_bigrams and source_sent_bigrams[j]:
                bg_ov = len(sent_bigrams & source_sent_bigrams[j]) / max(len(sent_bigrams), 1)
                max_sent_bigram_overlap = max(max_sent_bigram_overlap, bg_ov)

        f["max_source_sent_overlap"] = max_sent_overlap
        f["max_source_sent_bigram_overlap"] = max_sent_bigram_overlap
        f["best_source_sent_position"] = best_source_sent_idx / max(len(source_sents) - 1, 1)

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

        f["sent_to_source_ratio"] = len(sent_tokens) / max(len(source_tokens), 1)

        # NEW: Additional lexical alignment
        source_4grams = set(zip(source_tokens[:-3], source_tokens[1:-2], source_tokens[2:-1], source_tokens[3:])) if len(source_tokens) > 3 else set()
        sent_4grams = set(zip(sent_tokens[:-3], sent_tokens[1:-2], sent_tokens[2:-1], sent_tokens[3:])) if len(sent_tokens) > 3 else set()
        f["fourgram_overlap_with_source"] = len(sent_4grams & source_4grams) / max(len(sent_4grams), 1) if sent_4grams else 0

        union_tokens = sent_set | source_set
        f["jaccard_similarity"] = len(sent_set & source_set) / max(len(union_tokens), 1)
        f["containment_score"] = len(sent_set & source_set) / max(len(sent_set), 1)
        f["dice_coefficient"] = 2 * len(sent_set & source_set) / max(len(sent_set) + len(source_set), 1)

        high_overlap_src_sents = sum(1 for st in source_sent_tokens if st and len(sent_set & st) / max(len(sent_set), 1) > 0.5)
        f["source_sentence_count_high_overlap"] = high_overlap_src_sents

        # ==================================================================
        # GROUP 2: Stylistic and structural divergence
        # ==================================================================

        f["sent_len_chars"] = len(sent)
        f["sent_len_words"] = len(sent_tokens)
        f["sent_position"] = i
        f["sent_position_rel"] = i / max(n_sents - 1, 1)
        f["is_last_3"] = int(i >= n_sents - 3)
        f["is_last_5"] = int(i >= n_sents - 5)
        f["is_first_3"] = int(i < 3)
        f["n_sentences_in_doc"] = n_sents

        f["starts_with_number"] = int(bool(re.match(r"^\d+[\.\)]\s", sent)))
        f["starts_with_bullet"] = int(bool(re.match(r"^[-\*\u2022]\s", sent)))
        f["has_colon"] = int(":" in sent)
        f["n_special_chars"] = sum(1 for c in sent if c in "[]{}|\\<>@#$%^&*")
        f["exclamation_count"] = sent.count("!")
        f["question_count"] = sent.count("?")
        f["uppercase_ratio"] = sum(1 for c in sent if c.isupper()) / max(len(sent), 1)
        f["has_parenthetical"] = int(bool(re.search(r"\(.*?\)", sent)))

        f["avg_word_len"] = np.mean([len(w) for w in sent_tokens]) if sent_tokens else 0
        f["digit_ratio"] = sum(1 for c in sent if c.isdigit()) / max(len(sent), 1)
        f["compression_ratio"] = compression_ratio(sent)
        f["char_entropy"] = char_entropy(sent)
        f["type_token_ratio"] = len(sent_set) / max(len(sent_tokens), 1)

        f["sent_len_vs_doc_mean"] = len(sent_tokens) / max(doc_mean_len, 1)
        f["sent_len_zscore"] = (len(sent_tokens) - doc_mean_len) / max(doc_std_len, 1) if doc_std_len > 0 else 0

        # NEW: Additional structural
        f["is_first_sent"] = int(i == 0)
        f["is_last_sent"] = int(i == n_sents - 1)
        f["distance_from_end"] = n_sents - i
        f["sent_len_ratio_to_prev"] = len(sent_tokens) / max(all_sent_lens[i - 1], 1) if i > 0 else 1.0
        f["sent_len_ratio_to_next"] = len(sent_tokens) / max(all_sent_lens[i + 1], 1) if i < n_sents - 1 else 1.0
        f["is_longest_in_doc"] = int(len(sent_tokens) == max(all_sent_lens))
        f["is_shortest_in_doc"] = int(len(sent_tokens) == min(all_sent_lens)) if all_sent_lens else 0
        f["position_quartile"] = min(int(i / max(n_sents, 1) * 4), 3)

        # NEW: Additional stylistic
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
                     "have", "has", "had", "do", "does", "did", "will", "would", "could",
                     "should", "may", "might", "shall", "can", "to", "of", "in", "for",
                     "on", "with", "at", "by", "from", "as", "into", "through", "during",
                     "before", "after", "and", "but", "or", "nor", "not", "so", "yet",
                     "both", "either", "neither", "each", "every", "all", "any", "few",
                     "more", "most", "other", "some", "such", "no", "only", "own", "same",
                     "than", "too", "very", "just", "about", "above", "also", "it", "its",
                     "that", "this", "these", "those", "i", "we", "they", "he", "she",
                     "what", "which", "who", "whom", "when", "where", "how", "if", "then"}
        f["stopword_ratio"] = sum(1 for t in sent_tokens if t in stopwords) / max(len(sent_tokens), 1)
        f["punct_density"] = sum(1 for c in sent if c in ".,;:!?-()\"'") / max(len(sent), 1)
        f["hedging_word_count"] = sum(1 for t in sent_tokens if t in {"may", "might", "could", "possibly", "perhaps", "suggests", "appears", "seems"})
        f["certainty_word_count"] = sum(1 for t in sent_tokens if t in {"clearly", "definitely", "always", "never", "certainly", "obviously", "undoubtedly", "proven"})
        f["first_person_count"] = sum(1 for t in sent_tokens if t in {"i", "we", "my", "our", "me", "us"})
        f["modal_verb_count"] = sum(1 for t in sent_tokens if t in {"can", "could", "may", "might", "must", "shall", "should", "will", "would"})
        f["conjunction_count"] = sum(1 for t in sent_tokens if t in {"and", "but", "or", "nor", "yet", "so", "because", "although", "however", "therefore", "moreover", "furthermore"})
        f["passive_indicator"] = int(bool(re.search(r"\b(?:was|were|been|being)\s+\w+ed\b", sent_lower)))

        # ==================================================================
        # GROUP 3: Explicit overgeneration signals
        # ==================================================================

        # Entity grounding
        sent_entities = extract_entities_simple(sent)
        if sent_entities:
            grounded_ents = sum(1 for e in sent_entities if e.lower() in source_lower or e in source_entities)
            f["entity_grounding_score"] = grounded_ents / len(sent_entities)
            f["novel_entity_count"] = len(sent_entities) - grounded_ents
        else:
            f["entity_grounding_score"] = 1.0
            f["novel_entity_count"] = 0

        # Numeric grounding
        sent_numbers = extract_numbers(sent)
        if sent_numbers:
            grounded_nums = sum(1 for n in sent_numbers if n in source)
            f["numeric_grounding_score"] = grounded_nums / len(sent_numbers)
            f["novel_number_count"] = len(sent_numbers) - grounded_nums
        else:
            f["numeric_grounding_score"] = 1.0
            f["novel_number_count"] = 0

        # Hedging density
        hedge_count = sum(1 for p in HEDGING_PATTERNS if re.search(p, sent, re.IGNORECASE))
        f["hedging_density"] = hedge_count / max(len(sent_tokens), 1)

        # Causal language without source grounding
        causal_count = sum(1 for p in CAUSAL_PATTERNS if re.search(p, sent, re.IGNORECASE))
        f["unsupported_causal"] = int(causal_count > 0 and max_sent_overlap < 0.5)

        # Prompt/instruction patterns
        f["prompt_pattern_count"] = sum(1 for p in PROMPT_PATTERNS if re.search(p, sent, re.IGNORECASE))
        f["has_prompt_marker"] = int(f["prompt_pattern_count"] > 0)
        f["strong_leak_count"] = sum(1 for p in STRONG_LEAK_PATTERNS if re.search(p, sent, re.IGNORECASE))

        # NEW: Additional entity/claim features
        f["novel_entity_ratio"] = f["novel_entity_count"] / max(len(sent_entities), 1) if sent_entities else 0.0
        f["novel_number_ratio"] = f["novel_number_count"] / max(len(sent_numbers), 1) if sent_numbers else 0.0
        f["has_percentage_not_in_source"] = int(any(n.endswith('%') and n not in source for n in sent_numbers))
        f["has_statistic_not_in_source"] = int(bool(
            re.search(r'p\s*[=<>]\s*\d', sent) and not re.search(r'p\s*[=<>]\s*\d', source)
        ) or bool(
            re.search(r'CI\s*[:=]', sent, re.IGNORECASE) and not re.search(r'CI\s*[:=]', source, re.IGNORECASE)
        ))
        f["named_entity_count"] = len(sent_entities)
        f["total_number_count"] = len(sent_numbers)

        # NEW v8: Precise numeric claim grounding
        # Extract all decimal numbers from sentence and source
        sent_decimals = set(re.findall(r'\d+\.\d+', sent))
        source_decimals = set(re.findall(r'\d+\.\d+', source))
        f["novel_decimal_count"] = len(sent_decimals - source_decimals)
        f["novel_decimal_ratio"] = len(sent_decimals - source_decimals) / max(len(sent_decimals), 1)
        f["has_novel_decimal"] = int(len(sent_decimals - source_decimals) > 0)

        # CI/HR/OR/RR patterns -- hallucinated statistical claims
        stat_patterns = [
            r'(?:HR|OR|RR|RD)\s*[=:]?\s*\d+\.\d+',
            r'95%?\s*CI\s*[:=]?\s*\d',
            r'\bCI\b.*\d+\.\d+\s*to\s*\d+\.\d+',
            r'\d+\.\d+\s*,\s*95%',
            r'\(\s*\d+\.\d+\s*[-–to]+\s*\d+\.\d+\s*\)',
        ]
        sent_has_stats = sum(1 for p in stat_patterns if re.search(p, sent, re.IGNORECASE))
        source_has_stats = sum(1 for p in stat_patterns if re.search(p, source, re.IGNORECASE))
        f["stat_claim_count"] = sent_has_stats
        f["novel_stat_claims"] = max(0, sent_has_stats - source_has_stats)
        f["has_novel_stat_claim"] = int(f["novel_stat_claims"] > 0)

        # Numeric density relative to source
        sent_num_density = len(sent_numbers) / max(len(sent_tokens), 1)
        source_num_density = len(source_numbers) / max(len(source_tokens), 1)
        f["numeric_density"] = sent_num_density
        f["numeric_density_vs_source"] = sent_num_density / max(source_num_density, 0.001)

        # Best source sentence overlap BUT with novel numbers = suspicious
        f["high_overlap_novel_numbers"] = int(max_sent_overlap > 0.5 and f["novel_number_count"] > 0)
        f["high_overlap_novel_decimals"] = int(max_sent_overlap > 0.5 and f["novel_decimal_count"] > 0)

        # Sentence-source char-level containment (catches paraphrased insertions)
        sent_char_3grams = set(sent_lower[i:i+3] for i in range(len(sent_lower)-2)) if len(sent_lower) > 2 else set()
        source_char_3grams = set(source_lower[i:i+3] for i in range(min(len(source_lower)-2, 10000))) if len(source_lower) > 2 else set()
        f["char_trigram_overlap"] = len(sent_char_3grams & source_char_3grams) / max(len(sent_char_3grams), 1)
        f["char_trigram_novel_ratio"] = 1.0 - f["char_trigram_overlap"]

        # NEW: Additional pattern detection
        f["starts_with_however"] = int(sent_lower.startswith("however"))
        f["starts_with_additionally"] = int(sent_lower.startswith(("additionally", "furthermore", "moreover")))
        f["has_citation_pattern"] = int(bool(re.search(r'\[\d+\]|\(\w+(?:\s+et\s+al\.?)?,?\s*\d{4}\)', sent)))
        f["has_url"] = int(bool(re.search(r'https?://|www\.', sent)))
        f["has_list_marker"] = int(bool(re.match(r'^(?:\d+[.)]\s|[-*•]\s|[a-z][.)]\s)', sent)))
        f["ends_with_period"] = int(sent.rstrip().endswith('.'))
        f["ends_with_colon"] = int(sent.rstrip().endswith(':'))

        # NEW v9: Information-theoretic features
        # Conditional compression: how much does sentence compress given source?
        sent_bytes = sent.encode("utf-8")
        source_bytes = source[:3000].encode("utf-8")
        compressed_sent_alone = len(zlib.compress(sent_bytes, level=9))
        compressed_source = len(zlib.compress(source_bytes, level=9))
        compressed_joint = len(zlib.compress(source_bytes + sent_bytes, level=9))
        conditional_compressed = compressed_joint - compressed_source
        f["conditional_compression"] = conditional_compressed / max(compressed_sent_alone, 1)
        f["compression_gain"] = 1.0 - f["conditional_compression"]

        # Cumulative source coverage: how much source has been covered by sentences so far
        covered_tokens = set()
        for j in range(i + 1):
            covered_tokens |= (sent_token_sets[j] & source_set)
        f["cumulative_source_coverage"] = len(covered_tokens) / max(len(source_set), 1)
        if i > 0:
            prev_covered = set()
            for j in range(i):
                prev_covered |= (sent_token_sets[j] & source_set)
            f["source_coverage_delta"] = (len(covered_tokens) - len(prev_covered)) / max(len(source_set), 1)
        else:
            f["source_coverage_delta"] = f["cumulative_source_coverage"]
        f["coverage_efficiency"] = f["source_coverage_delta"] / max(f["sent_to_source_ratio"], 0.001)

        # ==================================================================
        # GROUP 4: Category-specific signals
        # ==================================================================

        # REPETITIVE_CONTENT
        f["max_overlap_with_any_prev"] = 0
        f["exact_match_any_prev"] = 0
        for j in range(max(0, i - 10), i):
            prev_j_set = sent_token_sets[j]
            ov = len(sent_set & prev_j_set) / max(len(sent_set), 1)
            f["max_overlap_with_any_prev"] = max(f["max_overlap_with_any_prev"], ov)
            if sent_lower == sentences[j].lower():
                f["exact_match_any_prev"] = 1

        max_overlap_any = 0
        for j in range(n_sents):
            if j == i:
                continue
            ov = len(sent_set & sent_token_sets[j]) / max(len(sent_set), 1)
            max_overlap_any = max(max_overlap_any, ov)
        f["max_overlap_any_sent"] = max_overlap_any

        # GENERATION_FAILURE
        if len(sent_tokens) > 3:
            tg = list(zip(sent_tokens[:-2], sent_tokens[1:-1], sent_tokens[2:]))
            f["trigram_self_repetition"] = 1.0 - len(set(tg)) / max(len(tg), 1)
        else:
            f["trigram_self_repetition"] = 0

        f["generation_failure_flag"] = int(
            (len(sent_tokens) < 3) or
            (not sent.rstrip().endswith(('.', '!', '?', ')'))) or
            (f["trigram_self_repetition"] > 0.3)
        )

        # UNGROUNDED_INJECTION composite
        f["ungrounded_injection_score"] = (
            (1.0 - max_sent_overlap) *
            (f["novel_entity_count"] + 1) *
            min(f["sent_to_source_ratio"] * 10, 3.0)
        )

        # GROUNDED_OVERGENERATION
        f["grounded_overgen_score"] = float(
            max_sent_overlap > 0.6 and f["novel_token_ratio"] > 0.15
        )

        # ==================================================================
        # GROUP 5: Sequential / contextual features
        # ==================================================================

        if i > 0:
            f["prev_novel_ratio"] = sent_novel_ratios[i - 1]
            f["novel_ratio_delta"] = sent_novel_ratios[i] - sent_novel_ratios[i - 1]
        else:
            f["prev_novel_ratio"] = 0
            f["novel_ratio_delta"] = 0

        if i > 1:
            f["novel_ratio_accel"] = (sent_novel_ratios[i] - sent_novel_ratios[i - 1]) - \
                                     (sent_novel_ratios[i - 1] - sent_novel_ratios[i - 2])
        else:
            f["novel_ratio_accel"] = 0

        tail_novels = sent_novel_ratios[i:]
        f["tail_novel_mean"] = np.mean(tail_novels)
        f["tail_novel_max"] = max(tail_novels)

        head_novels = sent_novel_ratios[:i + 1]
        f["head_novel_mean"] = np.mean(head_novels)

        if i > 0:
            f["max_prev_novel_ratio"] = max(sent_novel_ratios[:i])
        else:
            f["max_prev_novel_ratio"] = 0

        streak = 0
        for j in range(i, -1, -1):
            if sent_novel_ratios[j] > 0.3:
                streak += 1
            else:
                break
        f["high_novel_streak"] = streak

        window = 3
        window_novel = []
        for j in range(max(0, i - window), min(n_sents, i + window + 1)):
            if j == i:
                continue
            window_novel.append(sent_novel_ratios[j])
        f["novel_vs_window"] = f["novel_token_ratio"] - (np.mean(window_novel) if window_novel else 0)

        if i > 0:
            f["prev_prompt_pattern_count"] = sum(
                1 for p in PROMPT_PATTERNS if re.search(p, sentences[i - 1], re.IGNORECASE)
            )
        else:
            f["prev_prompt_pattern_count"] = 0

        # NEW: Cross-sentence lexical features
        if i > 0:
            prev_set = sent_token_sets[i - 1]
            f["overlap_with_prev_sent"] = len(sent_set & prev_set) / max(len(sent_set), 1)
        else:
            f["overlap_with_prev_sent"] = 0

        if i < n_sents - 1:
            next_set = sent_token_sets[i + 1]
            f["overlap_with_next_sent"] = len(sent_set & next_set) / max(len(sent_set), 1)
        else:
            f["overlap_with_next_sent"] = 0

        # Topic shift: how different is this sentence from the doc average novel ratio
        doc_mean_novel = np.mean(sent_novel_ratios) if sent_novel_ratios else 0
        doc_std_novel = np.std(sent_novel_ratios) if len(sent_novel_ratios) > 1 else 0
        f["novel_ratio_zscore"] = (sent_novel_ratios[i] - doc_mean_novel) / max(doc_std_novel, 0.01)
        f["is_novel_outlier"] = int(abs(f["novel_ratio_zscore"]) > 2.0)

        # Coherence: avg overlap with surrounding ±2 sentences
        neighbor_overlaps = []
        for j in range(max(0, i - 2), min(n_sents, i + 3)):
            if j == i:
                continue
            ov = len(sent_set & sent_token_sets[j]) / max(len(sent_set), 1)
            neighbor_overlaps.append(ov)
        f["neighbor_coherence"] = np.mean(neighbor_overlaps) if neighbor_overlaps else 0

        # Transition pattern: does novel ratio spike at this sentence?
        f["novel_spike"] = int(
            sent_novel_ratios[i] > 0.4 and
            (i == 0 or sent_novel_ratios[i] - sent_novel_ratios[i - 1] > 0.2)
        )

        features_list.append(f)

    return features_list


# ============================================================================
# Embedding similarity computation
# ============================================================================

def compute_embedding_features(docs: list[dict], embed_model, chunk_size: int = 500) -> list[dict]:
    """Compute embedding-based features in memory-efficient chunks.

    Processes documents in chunks to avoid OOM when encoding hundreds of thousands
    of sentences simultaneously.

    Returns a list of dicts (one per sentence across all docs) with keys:
        - embedding_sim_to_source: cosine sim to full source
        - max_source_sent_embedding_sim: max cosine sim to any source sentence
        - mean_source_sent_embedding_sim: mean cosine sim to source sentences
        - embedding_sim_rank_in_doc: rank of sim within document (0=highest)
    """
    total_sents = sum(len(d["sentences"]) for d in docs)
    log.info("  Processing %d docs (%d sentences) in chunks of %d docs...",
             len(docs), total_sents, chunk_size)

    results = []
    for chunk_start in range(0, len(docs), chunk_size):
        chunk_docs = docs[chunk_start:chunk_start + chunk_size]
        chunk_end = min(chunk_start + chunk_size, len(docs))
        log.info("  Chunk %d-%d/%d...", chunk_start, chunk_end, len(docs))

        # Collect texts for this chunk
        chunk_sources = []
        chunk_source_sents = []
        chunk_simp_sents = []
        source_sent_ranges = []
        doc_sent_ranges = []

        for doc in chunk_docs:
            chunk_sources.append(doc["source"][:2000])

            src_sents = split_into_sentences_simple(doc["source"])
            s_start = len(chunk_source_sents)
            chunk_source_sents.extend(src_sents)
            source_sent_ranges.append((s_start, len(chunk_source_sents)))

            d_start = len(chunk_simp_sents)
            chunk_simp_sents.extend(doc["sentences"])
            doc_sent_ranges.append((d_start, len(chunk_simp_sents)))

        # Encode this chunk
        src_embs = embed_model.encode(chunk_sources, batch_size=64, show_progress_bar=False,
                                      normalize_embeddings=True)
        src_sent_embs = embed_model.encode(chunk_source_sents, batch_size=128,
                                           show_progress_bar=False, normalize_embeddings=True)
        simp_embs = embed_model.encode(chunk_simp_sents, batch_size=128,
                                       show_progress_bar=False, normalize_embeddings=True)

        # Compute features for this chunk
        chunk_results = []
        for doc_idx in range(len(chunk_docs)):
            d_start, d_end = doc_sent_ranges[doc_idx]
            s_start, s_end = source_sent_ranges[doc_idx]
            doc_src_emb = src_embs[doc_idx]
            doc_src_sent_embs = src_sent_embs[s_start:s_end]

            doc_sims = []
            for sent_idx in range(d_start, d_end):
                sent_emb = simp_embs[sent_idx]
                sim_full = float(np.dot(doc_src_emb, sent_emb))

                if len(doc_src_sent_embs) > 0:
                    sims_to_src_sents = doc_src_sent_embs @ sent_emb
                    max_src_sent_sim = float(np.max(sims_to_src_sents))
                    mean_src_sent_sim = float(np.mean(sims_to_src_sents))
                    min_src_sent_sim = float(np.min(sims_to_src_sents))
                    std_src_sent_sim = float(np.std(sims_to_src_sents))
                else:
                    max_src_sent_sim = sim_full
                    mean_src_sent_sim = sim_full
                    min_src_sent_sim = sim_full
                    std_src_sent_sim = 0.0

                doc_sims.append(sim_full)
                chunk_results.append({
                    "embedding_sim_to_source": sim_full,
                    "max_source_sent_embedding_sim": max_src_sent_sim,
                    "mean_source_sent_embedding_sim": mean_src_sent_sim,
                    "min_source_sent_embedding_sim": min_src_sent_sim,
                    "std_source_sent_embedding_sim": std_src_sent_sim,
                    "embedding_sim_gap": max_src_sent_sim - mean_src_sent_sim,
                })

            # Rank feature within document
            sorted_sims = sorted(doc_sims, reverse=True)
            base_idx = len(chunk_results) - len(doc_sims)
            for j in range(len(doc_sims)):
                rank = sorted_sims.index(doc_sims[j]) / max(len(doc_sims) - 1, 1)
                chunk_results[base_idx + j]["embedding_sim_rank_in_doc"] = rank

        results.extend(chunk_results)

    mean_sim = np.mean([r["embedding_sim_to_source"] for r in results])
    mean_max = np.mean([r["max_source_sent_embedding_sim"] for r in results])
    log.info("  Mean source sim: %.3f, Mean max source-sent sim: %.3f", mean_sim, mean_max)

    return results


# ============================================================================
# NLI scoring
# ============================================================================

def compute_nli_features(docs: list[dict], nli_model) -> list[dict]:
    """Compute NLI entailment/contradiction scores.

    Returns list of dicts with keys: nli_entailment, nli_contradiction
    """
    pairs = []
    for doc in docs:
        src = doc["source"][:1500]
        for sent in doc["sentences"]:
            pairs.append((src, sent))

    log.info("  NLI scoring %d pairs...", len(pairs))
    batch_size = 128
    all_scores = []
    for start in tqdm(range(0, len(pairs), batch_size), desc="NLI"):
        batch = pairs[start:start + batch_size]
        scores = nli_model.predict(batch)
        if isinstance(scores, np.ndarray) and scores.ndim == 2:
            for row in scores:
                all_scores.append({
                    "nli_entailment": float(row[-1]),
                    "nli_contradiction": float(row[0]),
                })
        else:
            for s in scores:
                all_scores.append({
                    "nli_entailment": float(s),
                    "nli_contradiction": 0.0,
                })

    mean_ent = np.mean([s["nli_entailment"] for s in all_scores])
    mean_con = np.mean([s["nli_contradiction"] for s in all_scores])
    log.info("  Mean entailment: %.3f, Mean contradiction: %.3f", mean_ent, mean_con)

    return all_scores


# ============================================================================
# Dataset builder
# ============================================================================

LABELS = ["None", "LEAKED_INSTRUCTIONS", "UNGROUNDED_INJECTION",
          "GENERATION_FAILURE", "REPETITIVE_CONTENT", "GROUNDED_OVERGENERATION"]
LABEL2IDX = {l: i for i, l in enumerate(LABELS)}


def build_dataset(docs: list[dict],
                  embedding_features: Optional[list[dict]] = None,
                  nli_features: Optional[list[dict]] = None):
    """Build feature DataFrame from documents.

    Returns DataFrame with features, doc_id column, and label column (if labels present).
    """
    import pandas as pd

    all_features = []
    all_labels = []
    all_doc_ids = []
    idx = 0

    for doc in tqdm(docs, desc="Building features"):
        features = extract_features_for_doc(doc)

        for i, f in enumerate(features):
            if embedding_features is not None:
                f.update(embedding_features[idx])
            if nli_features is not None:
                f.update(nli_features[idx])
            idx += 1

            all_features.append(f)
            if "labels" in doc:
                all_labels.append(LABEL2IDX.get(doc["labels"][i], 0))
            all_doc_ids.append(doc["id"])

    df = pd.DataFrame(all_features)
    df["doc_id"] = all_doc_ids
    if all_labels:
        df["label"] = all_labels

    return df
