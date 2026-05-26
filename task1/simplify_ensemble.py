"""
SimpleText Task 1: Local ensemble simplification.
Generates multiple candidates with different prompting strategies,
scores them locally, and picks the best.
"""

import json
import os
import logging
import re
import math
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen3:8b"


def ollama_generate(prompt, max_tokens=512, temperature=0.3):
    resp = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": temperature},
    }, timeout=180)
    resp.raise_for_status()
    text = resp.json()["response"].strip()
    if "<think>" in text:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


# ============================================================================
# Prompting strategies (different simplification aggressiveness)
# ============================================================================

SENT_PROMPTS = {
    "balanced": (
        "/no_think\nSimplify this biomedical sentence for a general audience. "
        "Use simpler words and shorter sentences. Avoid jargon. "
        "Preserve key factual information. Do NOT add new information. "
        "Output ONLY the simplified sentence.\n\n"
        "Complex: {text}\n\nSimplified:"
    ),
    "aggressive": (
        "/no_think\nRewrite this biomedical sentence so a 12-year-old can understand it. "
        "Replace ALL technical terms with everyday words. "
        "Make it as short and simple as possible while keeping the main point. "
        "Do NOT add information. Output ONLY the simplified sentence.\n\n"
        "Complex: {text}\n\nSimplified:"
    ),
    "plain": (
        "/no_think\nConvert this medical research sentence into plain language. "
        "Remove unnecessary details and technical terms. "
        "Keep only the essential message. Use common words everyone knows. "
        "Output ONLY the plain language version.\n\n"
        "Medical: {text}\n\nPlain:"
    ),
}

SENT_PROMPTS_MULTILINGUAL = {
    "balanced": (
        "/no_think\nSimplify this biomedical sentence for a general audience. "
        "Keep the response in the SAME language as the input. "
        "Use simpler words. Preserve key facts. Do NOT add information. "
        "Output ONLY the simplified sentence.\n\n"
        "Complex: {text}\n\nSimplified:"
    ),
    "aggressive": (
        "/no_think\nRewrite this biomedical sentence so a 12-year-old can understand it. "
        "Keep the response in the SAME language as the input. "
        "Replace ALL technical terms with everyday words. Make it very short and simple. "
        "Output ONLY the simplified sentence.\n\n"
        "Complex: {text}\n\nSimplified:"
    ),
}

DOC_PROMPTS = {
    "balanced": (
        "/no_think\nSimplify this biomedical abstract for a general audience. "
        "Rewrite as a plain language summary. Use simpler words and shorter sentences. "
        "Avoid technical jargon. You may merge, split, or reorder sentences. "
        "Do NOT add new information. Output ONLY the simplified text.\n\n"
        "Complex:\n{text}\n\nSimplified:"
    ),
    "aggressive": (
        "/no_think\nRewrite this biomedical abstract so anyone can understand it. "
        "Use only common everyday words. Shorten aggressively. "
        "Remove unnecessary technical details but keep the key findings. "
        "Do NOT add information. Output ONLY the simplified text.\n\n"
        "Complex:\n{text}\n\nSimplified:"
    ),
}

DOC_PROMPTS_MULTILINGUAL = {
    "balanced": (
        "/no_think\nSimplify this biomedical abstract for a general audience. "
        "Keep the response in the SAME language as the input. "
        "Use simpler words. You may merge or reorder sentences. "
        "Output ONLY the simplified text.\n\n"
        "Complex:\n{text}\n\nSimplified:"
    ),
}


# ============================================================================
# Local scoring to pick best candidate
# ============================================================================

def word_set(text):
    return set(text.lower().split())


def flesch_kincaid_approx(text):
    """Lower = simpler. Rough approximation."""
    words = text.split()
    if not words:
        return 100
    sentences = max(text.count('.') + text.count('!') + text.count('?'), 1)
    syllables = sum(max(1, len(re.findall(r'[aeiouy]+', w.lower()))) for w in words)
    n_words = len(words)
    return 0.39 * (n_words / sentences) + 11.8 * (syllables / n_words) - 15.59


def avg_word_length(text):
    words = text.split()
    return sum(len(w) for w in words) / max(len(words), 1)


def score_candidate(original, candidate):
    """Score a simplification candidate. Higher = better simplification."""
    if not candidate or candidate == original:
        return -999

    orig_words = word_set(original)
    cand_words = word_set(candidate)

    # Reward: simpler reading level
    fk_orig = flesch_kincaid_approx(original)
    fk_cand = flesch_kincaid_approx(candidate)
    simplicity_gain = fk_orig - fk_cand  # positive = simpler

    # Reward: shorter average word length
    awl_gain = avg_word_length(original) - avg_word_length(candidate)

    # Reward: compression (shorter is simpler, but not too short)
    compression = len(candidate.split()) / max(len(original.split()), 1)
    compression_score = 0
    if 0.4 <= compression <= 0.95:
        compression_score = 1.0 - abs(compression - 0.7)  # sweet spot around 70%
    elif compression > 0.95:
        compression_score = -0.5  # barely changed

    # Penalty: too much content loss
    content_overlap = len(orig_words & cand_words) / max(len(orig_words), 1)

    # Penalty: candidate is longer than original
    if len(candidate.split()) > len(original.split()) * 1.1:
        compression_score -= 1.0

    score = (
        simplicity_gain * 0.3 +
        awl_gain * 2.0 +
        compression_score * 3.0 +
        content_overlap * 1.0  # some overlap is good (preserves meaning)
    )
    return score


def generate_candidates(text, prompts, max_tokens, language="en"):
    """Generate multiple simplification candidates."""
    candidates = []
    for name, template in prompts.items():
        prompt = template.format(text=text)
        try:
            result = ollama_generate(prompt, max_tokens=max_tokens, temperature=0.3)
            if result and len(result) > 5:
                candidates.append((name, result))
        except Exception as e:
            log.warning(f"  {name} failed: {e}")
    return candidates


def best_simplification(original, candidates):
    """Pick the best candidate based on local scoring."""
    if not candidates:
        return original

    scored = [(score_candidate(original, c), name, c) for name, c in candidates]
    scored.sort(reverse=True)
    return scored[0][2]


# ============================================================================
# Main processing
# ============================================================================

def process_sentence(item):
    lang = item.get("language", "en")
    prompts = SENT_PROMPTS if lang == "en" else SENT_PROMPTS_MULTILINGUAL
    candidates = generate_candidates(item["complex"], prompts, max_tokens=300, language=lang)
    prediction = best_simplification(item["complex"], candidates)
    return {**item, "prediction": prediction, "run_id": f"{TEAM_ID}_task11_ensemble"}


def process_document(item):
    lang = item.get("language", "en")
    prompts = DOC_PROMPTS if lang == "en" else DOC_PROMPTS_MULTILINGUAL
    candidates = generate_candidates(item["complex"], prompts, max_tokens=1000, language=lang)
    prediction = best_simplification(item["complex"], candidates)
    return {**item, "prediction": prediction, "run_id": f"{TEAM_ID}_task12_ensemble"}


def run_task(input_path, output_path, task):
    with open(input_path) as f:
        data = json.load(f)
    log.info(f"Task {task}: {len(data)} items")

    process_fn = process_sentence if task == "1.1" else process_document

    # Resume
    done = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        for item in existing:
            if task == "1.1":
                key = (item["pair_id"], str(item.get("para_id")), str(item.get("sent_id")))
            else:
                key = item["pair_id"]
            done[key] = item
        log.info(f"  Resuming: {len(done)} done")

    remaining = []
    for item in data:
        if task == "1.1":
            key = (item["pair_id"], str(item.get("para_id")), str(item.get("sent_id")))
        else:
            key = item["pair_id"]
        if key not in done:
            remaining.append(item)

    if not remaining:
        log.info("  All done")
        return

    log.info(f"  {len(remaining)} to process ({len(SENT_PROMPTS if task == '1.1' else DOC_PROMPTS)} candidates each for en)")

    all_results = list(done.values())
    save_every = 100

    for i, item in enumerate(tqdm(remaining, desc=f"Task {task}")):
        result = process_fn(item)
        all_results.append(result)

        if (i + 1) % save_every == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(all_results, f, ensure_ascii=False)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False)
    log.info(f"Done: {len(all_results)} predictions in {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1.1", "1.2"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="qwen3:8b")
    args = parser.parse_args()
    MODEL = args.model
    run_task(args.input, args.output, args.task)
