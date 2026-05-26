"""Run LLM grounding classifiers on all sentences.

For each sentence, asks 3 LLMs whether it is fully grounded in the source.
Caches results to avoid re-processing.

Usage:
    uv run python llm_grounding.py --split train
    uv run python llm_grounding.py --split dev
    uv run python llm_grounding.py --split test
"""

import asyncio
import json
import logging
import os
from pathlib import Path

import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CACHE_DIR = Path("cache/llm_grounding")

SYSTEM_PROMPT = """You are evaluating whether a sentence from a simplified scientific abstract is fully grounded in the original source abstract.

A sentence is GROUNDED if all its factual claims, numbers, and assertions can be found in or directly inferred from the source.
A sentence is NOT GROUNDED if it introduces any information not present in the source, including:
- Facts, statistics, or percentages not in the source
- Causal claims not supported by the source
- Generalizations beyond what the source states
- Names, dates, or entities not mentioned in the source
- Editorial commentary or instructions (e.g., "Note:", "In summary,")

Respond with exactly one word: GROUNDED or HALLUCINATED"""

USER_TEMPLATE = "SOURCE ABSTRACT:\n{source}\n\nSENTENCE TO EVALUATE:\n{sentence}"


async def call_gemini(client, source, sentence, semaphore):
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return None
    async with semaphore:
        for attempt in range(2):
            try:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                        "contents": [{"parts": [{"text": USER_TEMPLATE.format(source=source[:3000], sentence=sentence)}]}],
                        "generationConfig": {"temperature": 0.0, "maxOutputTokens": 10},
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    try:
                        text = data["candidates"][0]["content"]["parts"][0]["text"].strip().upper()
                        return 1.0 if "HALLUC" in text else 0.0
                    except (KeyError, IndexError):
                        return None
                elif resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt * 3)
                else:
                    return None
            except Exception:
                await asyncio.sleep(2 ** attempt)
    return None


async def call_openai(client, source, sentence, semaphore, model="gpt-4o-mini"):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None
    async with semaphore:
        for attempt in range(2):
            try:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": model,
                        "max_tokens": 10,
                        "temperature": 0.0,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": USER_TEMPLATE.format(source=source[:3000], sentence=sentence)},
                        ],
                    },
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"].strip().upper()
                    return 1.0 if "HALLUC" in text else 0.0
                elif resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt * 2)
                else:
                    return None
            except Exception:
                await asyncio.sleep(2 ** attempt)
    return None


async def call_deepseek(client, source, sentence, semaphore):
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None
    async with semaphore:
        for attempt in range(2):
            try:
                resp = await client.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "deepseek-chat",
                        "max_tokens": 10,
                        "temperature": 0.0,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": USER_TEMPLATE.format(source=source[:3000], sentence=sentence)},
                        ],
                    },
                )
                if resp.status_code == 200:
                    text = resp.json()["choices"][0]["message"]["content"].strip().upper()
                    return 1.0 if "HALLUC" in text else 0.0
                elif resp.status_code == 429:
                    await asyncio.sleep(2 ** attempt * 2)
                else:
                    return None
            except Exception:
                await asyncio.sleep(2 ** attempt)
    return None


async def classify_sentence(client, source, sentence, sems):
    results = await asyncio.gather(
        call_openai(client, source, sentence, sems["openai"]),
        call_deepseek(client, source, sentence, sems["deepseek"]),
    )
    return {"gpt4o_mini": results[0], "deepseek": results[1]}


async def run_split(split_name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{split_name}.json"

    docs = json.load(open(f"{split_name}_data.json"))
    log.info("Loaded %d docs from %s_data.json", len(docs), split_name)

    cached = {}
    if cache_path.exists():
        cached = json.load(open(cache_path))
        log.info("Loaded %d cached results", len(cached))

    total_sents = sum(len(d["sentences"]) for d in docs)
    to_process = []
    for d in docs:
        for i, sent in enumerate(d["sentences"]):
            key = f"{d['id']}_{i}"
            if key not in cached:
                to_process.append((d["id"], i, d["source"], sent, key))

    log.info("Total sentences: %d, cached: %d, to process: %d",
             total_sents, len(cached), len(to_process))

    if not to_process:
        log.info("All cached!")
        return cached

    sems = {
        "gemini": asyncio.Semaphore(2),
        "openai": asyncio.Semaphore(20),
        "deepseek": asyncio.Semaphore(10),
    }

    async with httpx.AsyncClient(timeout=90) as client:
        batch_size = 30
        for i in range(0, len(to_process), batch_size):
            batch = to_process[i:i + batch_size]
            tasks = [classify_sentence(client, src, sent, sems)
                     for _, _, src, sent, _ in batch]
            results = await asyncio.gather(*tasks)

            for (_, _, _, _, key), result in zip(batch, results):
                cached[key] = result

            if (i + batch_size) % 300 == 0 or i + batch_size >= len(to_process):
                with open(cache_path, "w") as f:
                    json.dump(cached, f)

            done = min(i + batch_size, len(to_process))
            if done % 300 == 0 or done == len(to_process):
                log.info("  %d/%d done", done, len(to_process))

    with open(cache_path, "w") as f:
        json.dump(cached, f)
    log.info("Saved %d results to %s", len(cached), cache_path)
    return cached


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "dev", "test"])
    args = parser.parse_args()
    asyncio.run(run_split(args.split))


if __name__ == "__main__":
    main()
