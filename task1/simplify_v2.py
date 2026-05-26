"""
SimpleText Task 1 v2: Parallel simplification with incremental saves.
Handles the 2026 competition format with multilingual data.
"""

import anthropic
import asyncio
import json
import os
import logging
import time
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
CONCURRENT = 20


def simplify_sentence_prompt(complex_sentence: str, language: str = "en") -> str:
    if language != "en":
        return (
            f"Simplify the following biomedical sentence for a general audience. "
            f"Keep the response in the SAME language as the input. "
            f"Use simpler words and avoid jargon. Preserve key facts. "
            f"Output ONLY the simplified sentence, nothing else.\n\n"
            f"Complex: {complex_sentence}\n\nSimplified:"
        )
    return (
        f"Simplify this biomedical sentence so a non-expert can understand it. "
        f"Replace ALL medical and technical terms with common everyday words. "
        f"Split long sentences into shorter ones. "
        f"Keep all the key facts but express them more simply. "
        f"Do NOT remove important information. Do NOT add new information. "
        f"Output ONLY the simplified sentence(s).\n\n"
        f"Complex: {complex_sentence}\n\nSimplified:"
    )


def simplify_document_prompt(complex_document: str, language: str = "en") -> str:
    if language != "en":
        return (
            f"Simplify this biomedical abstract for a general audience. "
            f"Keep the response in the SAME language as the input. "
            f"Replace technical terms with everyday words. Keep ALL the key information. "
            f"The simplified version should be similar in length to the original. "
            f"Output ONLY the simplified text.\n\n"
            f"Complex:\n{complex_document}\n\nSimplified:"
        )
    return (
        f"Simplify this biomedical abstract so a non-expert can understand it. "
        f"Replace ALL medical and technical terms with common everyday words. "
        f"Keep ALL the key information and findings - do NOT remove content. "
        f"The simplified version should be a similar length to the original. "
        f"You may split complex sentences but do NOT delete sentences. "
        f"Do NOT add new information. Output ONLY the simplified text.\n\n"
        f"Complex:\n{complex_document}\n\nSimplified:"
    )


async def process_batch(client, items, prompt_fn, task_name, model, max_tokens):
    sem = asyncio.Semaphore(CONCURRENT)

    async def process_one(item):
        lang = item.get("language", "en")
        prompt = prompt_fn(item["complex"], lang)
        async with sem:
            try:
                resp = await client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()
            except Exception as e:
                log.warning(f"API error: {e}")
                await asyncio.sleep(2)
                return item["complex"]

    tasks = [process_one(item) for item in items]
    results = []
    for coro in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=task_name):
        results.append(await coro)

    # as_completed returns out of order, so we need indexed approach instead
    return results


async def run_task(input_path, output_path, task, model="claude-haiku-4-5-20251001"):
    with open(input_path) as f:
        data = json.load(f)
    log.info(f"Task {task}: {len(data)} items from {input_path}")

    # Resume from partial output
    done = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            existing = json.load(f)
        for item in existing:
            if task == "1.1":
                key = (item["pair_id"], item.get("para_id"), item.get("sent_id"))
            else:
                key = item["pair_id"]
            done[key] = item
        log.info(f"  Resuming: {len(done)} already done")

    remaining = []
    for item in data:
        if task == "1.1":
            key = (item["pair_id"], item.get("para_id"), item.get("sent_id"))
        else:
            key = item["pair_id"]
        if key not in done:
            remaining.append(item)

    if not remaining:
        log.info("  All items already processed")
        return

    log.info(f"  {len(remaining)} items to process")

    is_sentence = task == "1.1"
    prompt_fn = simplify_sentence_prompt if is_sentence else simplify_document_prompt
    max_tokens = 300 if is_sentence else 1000
    run_id = f"{TEAM_ID}_task{task.replace('.', '')}_claude"

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(CONCURRENT)

    save_interval = 100
    processed_count = 0

    async def process_one(item):
        lang = item.get("language", "en")
        prompt = prompt_fn(item["complex"], lang)
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return {**item, "prediction": resp.content[0].text.strip(), "run_id": run_id}
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2 ** attempt)
                    else:
                        log.warning(f"Failed after 3 attempts: {e}")
                        return {**item, "prediction": item["complex"], "run_id": run_id}

    # Process in chunks to save incrementally
    chunk_size = 500
    all_results = list(done.values())

    for start in range(0, len(remaining), chunk_size):
        chunk = remaining[start:start + chunk_size]
        log.info(f"  Processing chunk {start}-{start + len(chunk)} of {len(remaining)}")

        tasks_list = [process_one(item) for item in chunk]
        chunk_results = await asyncio.gather(*tasks_list)
        all_results.extend(chunk_results)

        # Save after each chunk
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False)
        log.info(f"  Saved {len(all_results)} total predictions")

    log.info(f"Done: {len(all_results)} predictions in {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1.1", "1.2"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    asyncio.run(run_task(args.input, args.output, args.task, model=args.model))


if __name__ == "__main__":
    main()
