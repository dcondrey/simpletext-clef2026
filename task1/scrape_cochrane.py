"""Scrape Cochrane reviews for abstract + PLS pairs."""

import asyncio
import json
import logging
import os
import re
import httpx

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CONCURRENT = 5
OUTPUT_PATH = "cochrane-auto/data/cochrane_scraped_2026.json"


def extract_sections(html):
    abstract = ""
    pls = ""

    abs_match = re.search(
        r'<section[^>]*class="[^"]*abstract[^"]*"[^>]*>(.*?)</section>',
        html, re.DOTALL | re.IGNORECASE
    )
    if abs_match:
        abstract = re.sub(r'<[^>]+>', ' ', abs_match.group(1))
        abstract = re.sub(r'\s+', ' ', abstract).strip()

    pls_match = re.search(
        r'<section[^>]*id="[^"]*plainLanguageSummary[^"]*"[^>]*>(.*?)</section>',
        html, re.DOTALL | re.IGNORECASE
    )
    if not pls_match:
        pls_match = re.search(
            r'<section[^>]*class="[^"]*pls[^"]*"[^>]*>(.*?)</section>',
            html, re.DOTALL | re.IGNORECASE
        )
    if pls_match:
        pls = re.sub(r'<[^>]+>', ' ', pls_match.group(1))
        pls = re.sub(r'\s+', ' ', pls).strip()

    return abstract, pls


async def fetch_review(client, sem, cd_id):
    url = f"https://www.cochranelibrary.com/cdsr/doi/10.1002/14651858.{cd_id}.pub2/full"
    async with sem:
        try:
            r = await client.get(url, follow_redirects=True, timeout=20)
            if r.status_code != 200:
                url_v1 = url.replace(".pub2", "")
                r = await client.get(url_v1, follow_redirects=True, timeout=20)
            if r.status_code == 200:
                abstract, pls = extract_sections(r.text)
                if abstract and pls:
                    return {"pair_id": cd_id, "abstract": abstract, "pls": pls}
        except Exception as e:
            log.warning("Error fetching %s: %s", cd_id, e)
    return None


async def main():
    cd_ids = [f"CD{i:06d}" for i in range(14000, 16500)]
    log.info("Fetching %d reviews...", len(cd_ids))

    results = []
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH) as f:
            results = json.load(f)
        done_ids = {r["pair_id"] for r in results}
        cd_ids = [c for c in cd_ids if c not in done_ids]
        log.info("Resuming: %d done, %d remaining", len(results), len(cd_ids))

    sem = asyncio.Semaphore(CONCURRENT)
    async with httpx.AsyncClient(headers={"User-Agent": "Mozilla/5.0"}) as client:
        for start in range(0, len(cd_ids), 50):
            chunk = cd_ids[start:start + 50]
            tasks = [fetch_review(client, sem, cd) for cd in chunk]
            chunk_results = await asyncio.gather(*tasks)
            new = [r for r in chunk_results if r is not None]
            results.extend(new)

            with open(OUTPUT_PATH, "w") as f:
                json.dump(results, f, ensure_ascii=False)

            log.info("Chunk %d-%d: %d new, %d total",
                     start, start + len(chunk), len(new), len(results))

    log.info("Done: %d reviews with abstract+PLS pairs", len(results))


if __name__ == "__main__":
    asyncio.run(main())
