"""
SimpleText Task 1 v4: SARI-optimized simplification.

Key insight: SARI rewards keeping original words that appear in the reference,
deleting jargon/stats that don't, and adding only necessary simpler replacements.
Our v2 over-paraphrased (too many additions, expansions). This version:
- Preserves original wording wherever possible
- Only substitutes actual technical terms
- Removes statistical details (CI, RR, p-values) rather than explaining them
- Targets ~55-65% compression ratio
- Does not split sentences unless genuinely complex
"""

import asyncio
import json
import os
import logging
import openai
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"
CONCURRENT = 50

SYSTEM_PROMPT = (
    "You are a Cochrane plain language summary writer. "
    "You simplify biomedical text by making minimal, targeted changes: "
    "replace medical jargon with everyday words, remove statistical measures "
    "in parentheses (confidence intervals, risk ratios, p-values, I² values, IQR), "
    "but keep important numbers (participant counts, study counts, percentages of improvement). "
    "Keep the original sentence structure and wording. Never explain or expand."
)

SENT_EXAMPLES_STR = """Complex: Computer reminders achieved a median improvement in process adherence of 4.2% (interquartile range (IQR): 0.8% to 18.8%) across all reported process outcomes, 3.3% (IQR: 0.5% to 10.6%) for medication ordering, 3.8% (IQR: 0.5% to 6.6%) for vaccinations, and 3.8% (IQR: 0.4% to 16.3%) for test ordering.
Output: Computer reminders achieved a median improvement of 4.2% across all reported process outcomes, 3.3% for medication ordering, 3.8% for vaccinations, and 3.8% for test ordering.

Complex: TCV compared to control may result in a large reduction in acute typhoid fever (risk ratio (RR) 0.20, 95% confidence interval (CI) 0.12 to 0.32; I2 = 32%; 7 studies, 105,839 participants; low-certainty evidence).
Output: The typhoid conjugate vaccine compared to control may greatly reduce acute typhoid fever (7 studies, 105,839 participants; low-certainty evidence).

Complex: Prophylactic antibiotics did not have an important effect on dyspareunia (difficult or painful sexual intercourse) or breastfeeding at six weeks.
Output: Prophylactic antibiotics did not have a clear effect on pain during sex or breastfeeding at six weeks.

Complex: We included 19 trials (17 RCTs and two cluster-RCTs).
Output: We included 19 trials.

Complex: The 19 trials enrolled 395,650 participants, with ages ranging from six weeks to 60 years.
Output: The 19 trials enrolled 395,650 participants, with ages ranging from six weeks to 60 years.

Complex: There may be little or no difference between the skin closure techniques in terms of incisional hernia and operative time, though the evidence for these two outcomes is very uncertain.
Output: There may be little or no difference between the two techniques in the risk of incisional hernia or in operative time, but the results are very uncertain.

Complex: Taxane-containing regimens appear to improve overall survival, time to progression, and tumour response rate in women with metastatic breast cancer.
Output: Chemotherapy regimens including taxanes improved survival and decreased the progression of metastatic breast cancer."""


def simplify_sentence_prompt(complex_sentence: str, language: str = "en") -> str:
    if language != "en":
        return (
            f"Simplify this biomedical sentence for a general audience. "
            f"Keep the response in the SAME language as the input ({language}). "
            f"Replace medical/technical terms with simpler words. "
            f"Remove statistical measures in parentheses (CI, RR, p-values, IQR). "
            f"Keep important numbers (participant counts, percentages, ages). "
            f"Keep the sentence structure. "
            f"Output ONLY the simplified sentence.\n\n"
            f"Complex: {complex_sentence}\n\nOutput:"
        )
    return (
        f"Simplify this biomedical sentence for a general audience.\n"
        f"- Remove ONLY statistical measures in parentheses (CI, RR, OR, p-values, IQR, I²)\n"
        f"- Keep important numbers: participant counts, study counts, percentages, ages\n"
        f"- Replace medical/technical terms with common words\n"
        f"- Keep the sentence structure and non-technical words exactly as they are\n"
        f"- Output ONLY the simplified sentence\n\n"
        f"Examples:\n\n"
        f"{SENT_EXAMPLES_STR}\n\n"
        f"Complex: {complex_sentence}\n\nOutput:"
    )


DOC_EXAMPLE = {
    "complex": (
        "We included five trials, in which 1406 infants participated. They were conducted "
        "in 13 neonatal centres across Europe and Australia. Each of these trials compared "
        "clinical outcomes in preterm infants who received respiratory support via a nasal "
        "interface compared with those who received support via a face mask in the delivery "
        "room. None of the included trials were blinded to the intervention and assessors "
        "were not blinded to treatment allocation. Four trials had their randomisation codes "
        "within opaque, sealed envelopes and one trial did not report its method. There was "
        "an unclear risk of bias in at least one domain in each trial. When the results of "
        "included trials were combined, there was no significant difference in the rate of "
        "intubation in the delivery room between nasal interface and face mask groups (risk "
        "ratio (RR) 0.86, 95% confidence interval (CI) 0.72 to 1.04; 3 trials, 898 "
        "participants; moderate-quality evidence, downgraded for imprecision). No significant "
        "difference was seen in death before hospital discharge (RR 0.72, 95% CI 0.36 to "
        "1.43; 5 trials, 1406 participants; low-quality evidence, downgraded for imprecision "
        "and lack of blinding). There were no significant differences in any of the secondary "
        "outcomes: air leak (3 trials; 898 participants), intubation within 72 hours of "
        "delivery (2 trials, 469 participants), bronchopulmonary dysplasia (3 trials, 853 "
        "participants), intraventricular haemorrhage (3 trials, 898 participants), or "
        "duration of respiratory support (2 trials, 469 participants)."
    ),
    "simple": (
        "We found five studies that involved 1406 babies whose breathing was supported with "
        "a nasal interface compared to a face mask in the delivery room. The studies were "
        "conducted in 13 neonatal centres across Europe and Australia. None of the included "
        "trials were blinded, which means that the caregivers knew which intervention the "
        "babies received. There was an unclear risk of bias in at least one domain in each "
        "trial. When the results of included trials were combined, there was no important "
        "difference in the number of babies that needed a tube to be inserted in the delivery "
        "room. There was no important difference in the number of deaths before discharge "
        "from hospital. There was no important difference in any of the secondary outcomes: "
        "air leak, having a tube inserted within 72 hours, chronic lung disease, brain "
        "haemorrhage, or how long breathing support was needed."
    ),
}


def simplify_document_prompt(complex_document: str, language: str = "en") -> str:
    if language != "en":
        return (
            f"Simplify this biomedical abstract for a general audience. "
            f"Keep the response in the SAME language as the input ({language}). "
            f"Replace medical terms with everyday words. "
            f"Remove statistical details (CI, RR, p-values). "
            f"Keep ALL sentences and findings — do not delete content. "
            f"The simplified version should be 50-70% of the original length. "
            f"Output ONLY the simplified text.\n\n"
            f"Complex:\n{complex_document}\n\nSimplified:"
        )
    return (
        f"Example of how to simplify a biomedical abstract:\n\n"
        f"Complex:\n{DOC_EXAMPLE['complex']}\n\n"
        f"Plain language summary:\n{DOC_EXAMPLE['simple']}\n\n"
        f"Now simplify the following abstract using the same approach:\n"
        f"- Replace medical/technical terms with everyday words\n"
        f"- Remove ALL statistical details (confidence intervals, risk ratios, p-values, "
        f"I² values, sample sizes in parentheses)\n"
        f"- Remove sentences about methodology only (risk of bias, blinding, allocation, "
        f"heterogeneity, sensitivity analysis) if they contain no findings\n"
        f"- Keep all sentences about findings, results, and conclusions\n"
        f"- Preserve the original wording for non-technical content\n"
        f"- The result should be 50-65% of the original length\n"
        f"- Output ONLY the plain language summary\n\n"
        f"Complex:\n{complex_document}\n\nPlain language summary:"
    )


async def run_task(input_path, output_path, task, model="meta-llama/Llama-3.3-70B-Instruct-Turbo"):
    with open(input_path) as f:
        data = json.load(f)

    model_tag = model.replace("-", "").replace("claude", "").replace(".", "")
    run_id = f"{TEAM_ID}_Task{task.replace('.', '')}_V4"
    log.info(f"Task {task}: {len(data)} items, model={model}, run_id={run_id}")

    # Resume from partial output
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
    max_tokens = 300 if is_sentence else 1500

    if model.startswith("gpt-"):
        base_url = "https://api.openai.com/v1"
        api_key = os.environ["OPENAI_API_KEY"]
    elif model.startswith("anthropic/") or model.startswith("google/"):
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ["OPENROUTER_API_KEY"]
    elif model.startswith("meta-llama/"):
        base_url = "https://api.together.xyz/v1"
        api_key = os.environ["TOGETHER_API_KEY"]
    else:
        base_url = "https://api.groq.com/openai/v1"
        api_key = os.environ["GROQ_API_KEY"]
    client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
    sem = asyncio.Semaphore(CONCURRENT)

    async def _call(prompt, temp=0.2):
        resp = await client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temp,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        return resp.choices[0].message.content.strip()

    async def process_one(item):
        lang = item.get("language", "en")
        prompt = prompt_fn(item["complex"], lang)
        async with sem:
            for attempt in range(3):
                try:
                    prediction = await _call(prompt)

                    if is_sentence:
                        orig_words = len(item["complex"].split())
                        pred_words = len(prediction.split())
                        if pred_words > orig_words * 1.3:
                            retry_prompt = (
                                f"Simplify and shorten this biomedical sentence. "
                                f"Replace technical terms with simple words. Remove all "
                                f"statistical details in parentheses. Make it shorter. "
                                f"Output ONLY the simplified sentence.\n\n"
                                f"Original: {item['complex']}\n\nSimplified:"
                            )
                            prediction = await _call(retry_prompt, temp=0.1)

                    return {**item, "prediction": prediction, "run_id": run_id}
                except Exception as e:
                    if attempt < 2:
                        await asyncio.sleep(2 ** (attempt + 1))
                    else:
                        log.warning(f"Failed: {e}")
                        return {**item, "prediction": item["complex"], "run_id": run_id}

    chunk_size = 500
    all_results = list(done.values())

    for start in range(0, len(remaining), chunk_size):
        chunk = remaining[start : start + chunk_size]
        log.info(f"  Chunk {start}-{start + len(chunk)} of {len(remaining)}")
        chunk_results = await asyncio.gather(*[process_one(item) for item in chunk])
        all_results.extend(chunk_results)

        # Save after each chunk
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False)
        log.info(f"  Saved {len(all_results)} total")

        # Log compression stats for this chunk
        ratios = []
        for r in chunk_results:
            orig = len(r["complex"].split())
            pred = len(r["prediction"].split())
            if orig > 0:
                ratios.append(pred / orig)
        if ratios:
            avg = sum(ratios) / len(ratios)
            log.info(f"  Chunk avg compression: {avg:.2f}")

    log.info(f"Done: {len(all_results)} predictions")


def package_submission(json_path, task):
    """Package JSON predictions into a submission ZIP."""
    import zipfile

    with open(json_path) as f:
        data = json.load(f)

    task_num = task.replace(".", "")
    zip_name = f"task{task_num}_2026_submission.zip"
    zip_path = os.path.join(os.path.dirname(json_path), zip_name)
    json_name = os.path.basename(json_path)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(json_name, json.dumps(data, ensure_ascii=False))

    log.info(f"Packaged {len(data)} predictions -> {zip_path}")
    return zip_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1.1", "1.2"], required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    parser.add_argument("--package", action="store_true", help="Package output as submission ZIP")
    args = parser.parse_args()

    asyncio.run(run_task(args.input, args.output, args.task, model=args.model))

    if args.package:
        package_submission(args.output, args.task)
