"""
SimpleText Task 1 v3: Few-shot prompting with Sonnet.
Uses real Cochrane complex→simple pairs as examples to match target style.
"""

import asyncio
import json
import os
import logging
import time
import openai
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
CONCURRENT = 15  # slightly lower for Sonnet rate limits

SENT_EXAMPLES = [
    {
        "complex": "Three trials (total 70 participants, aged 8 to 46 years) assessed as having a moderate risk of bias were included.",
        "simple": "We included three trials (total of 70 participants aged between 8 and 46 years) in the review.",
    },
    {
        "complex": "Prophylactic antibiotics did not have an important effect on dyspareunia (difficult or painful sexual intercourse) or breastfeeding at six weeks.",
        "simple": "Prophylactic antibiotics did not have a clear effect on pain during sex and breastfeeding at six weeks.",
    },
    {
        "complex": "Taxane-containing regimens appear to improve overall survival, time to progression, and tumour response rate in women with metastatic breast cancer.",
        "simple": "This review showed that chemotherapy regimens including taxanes improved survival and decreased the progression of metastatic breast cancer.",
    },
    {
        "complex": "There may be little or no difference between the skin closure techniques in terms of incisional hernia and operative time, though the evidence for these two outcomes is very uncertain.",
        "simple": "There may be little or no difference between the two techniques in the risk of incisional hernia or in operative time, but the results are very uncertain.",
    },
    {
        "complex": "The effect of amorphous hydrogel dressings compared with other types of dressings is uncertain for pain at the donor site and wound complications, including scarring and itching (very low-certainty evidence).",
        "simple": "We are not sure about the results for hydrogel dressings compared with other dressings for pain at the donor site and wound complications, including scarring and itching.",
    },
]

DOC_EXAMPLE = {
    "complex": "We included five trials, in which 1406 infants participated. They were conducted in 13 neonatal centres across Europe and Australia. Each of these trials compared clinical outcomes in preterm infants who received respiratory support via a nasal interface compared with those who received support via a face mask in the delivery room. None of the included trials were blinded to the intervention and assessors were not blinded to treatment allocation. Four trials had their randomisation codes within opaque, sealed envelopes and one trial did not report its method. There was an unclear risk of bias in at least one domain in each trial. When the results of included trials were combined, there was no significant difference in the rate of intubation in the delivery room between nasal interface and face mask groups (risk ratio (RR) 0.86, 95% confidence interval (CI) 0.72 to 1.04; 3 trials, 898 participants; moderate-quality evidence, downgraded for imprecision). No significant difference was seen in death before hospital discharge (RR 0.72, 95% CI 0.36 to 1.43; 5 trials, 1406 participants; low-quality evidence, downgraded for imprecision and lack of blinding). There were no significant differences in any of the secondary outcomes: air leak (3 trials; 898 participants), intubation within 72 hours of delivery (2 trials, 469 participants), bronchopulmonary dysplasia (3 trials, 853 participants), intraventricular haemorrhage (3 trials, 898 participants), or duration of respiratory support (2 trials, 469 participants).",
    "simple": "We found five studies that involved 1406 babies whose breathing was supported with a nasal interface compared to a face mask in the delivery room. The studies were conducted in 13 neonatal centres across Europe and Australia. None of the included trials were blinded, which means that the caregivers knew which intervention the babies received. There was an unclear risk of bias in at least one domain in each trial. When the results of included trials were combined, there was no important difference in the number of babies that needed a tube to be inserted in the delivery room. There was no important difference in the number of deaths before discharge from hospital. There was no important difference in any of the secondary outcomes: air leak, having a tube inserted within 72 hours, chronic lung disease, brain haemorrhage, or how long breathing support was needed.",
}

SENT_EXAMPLES_STR = "\n\n".join(
    f"Complex: {ex['complex']}\nSimplified: {ex['simple']}"
    for ex in SENT_EXAMPLES
)


def simplify_sentence_prompt(complex_sentence, language="en"):
    if language != "en":
        return (
            f"You are a Cochrane plain language summary writer. Simplify biomedical sentences "
            f"for a general audience following Cochrane style.\n"
            f"Keep the response in the SAME language as the input.\n"
            f"Output ONLY the simplified sentence.\n\n"
            f"Complex: {complex_sentence}\n\nSimplified:"
        )
    return (
        f"You are a Cochrane plain language summary writer. Your task is to simplify "
        f"biomedical sentences for a general audience. Follow the Cochrane style shown "
        f"in these examples:\n\n"
        f"{SENT_EXAMPLES_STR}\n\n"
        f"Now simplify the following sentence in the same style. "
        f"Replace technical terms with plain language. Keep the key facts. "
        f"Output ONLY the simplified sentence.\n\n"
        f"Complex: {complex_sentence}\n\nSimplified:"
    )


def simplify_document_prompt(complex_document, language="en"):
    if language != "en":
        return (
            f"You are a Cochrane plain language summary writer. Rewrite this biomedical "
            f"abstract as a plain language summary for a general audience.\n"
            f"Keep the response in the SAME language as the input.\n"
            f"Replace technical terms with everyday words. Keep all key findings.\n"
            f"Output ONLY the simplified text.\n\n"
            f"Complex:\n{complex_document}\n\nPlain language summary:"
        )
    return (
        f"You are a Cochrane plain language summary writer. Rewrite biomedical abstracts "
        f"as plain language summaries for a general audience.\n\n"
        f"Example:\n\n"
        f"Complex:\n{DOC_EXAMPLE['complex']}\n\n"
        f"Plain language summary:\n{DOC_EXAMPLE['simple']}\n\n"
        f"Now rewrite the following abstract in the same Cochrane plain language style. "
        f"Replace technical terms with everyday words. Keep all key findings and conclusions. "
        f"The summary should be similar in length and coverage to the original. "
        f"Output ONLY the plain language summary.\n\n"
        f"Complex:\n{complex_document}\n\nPlain language summary:"
    )


async def run_task(input_path, output_path, task, model="anthropic/claude-sonnet-4"):
    with open(input_path) as f:
        data = json.load(f)

    model_short = model.split("/")[-1].split("-")[0].capitalize()
    run_id = f"{TEAM_ID}_Task{task.replace('.', '')}_{model_short}V3"
    log.info(f"Task {task}: {len(data)} items, model={model}, run_id={run_id}")

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

    log.info(f"  {len(remaining)} to process")

    is_sentence = task == "1.1"
    prompt_fn = simplify_sentence_prompt if is_sentence else simplify_document_prompt
    max_tokens = 400 if is_sentence else 1500

    client = openai.AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    sem = asyncio.Semaphore(CONCURRENT)

    async def process_one(item):
        lang = item.get("language", "en")
        prompt = prompt_fn(item["complex"], lang)
        async with sem:
            for attempt in range(3):
                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        max_tokens=max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    return {**item, "prediction": resp.choices[0].message.content.strip(), "run_id": run_id}
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2 ** (attempt + 1))
                    else:
                        log.warning(f"Failed: {e}")
                        return {**item, "prediction": item["complex"], "run_id": run_id}

    chunk_size = 500
    all_results = list(done.values())

    for start in range(0, len(remaining), chunk_size):
        chunk = remaining[start:start + chunk_size]
        log.info(f"  Chunk {start}-{start + len(chunk)} of {len(remaining)}")
        chunk_results = await asyncio.gather(*[process_one(item) for item in chunk])
        all_results.extend(chunk_results)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False)
        log.info(f"  Saved {len(all_results)} total")

    log.info(f"Done: {len(all_results)} predictions")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1.1", "1.2"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="anthropic/claude-sonnet-4")
    args = parser.parse_args()
    asyncio.run(run_task(args.input, args.output, args.task, model=args.model))
