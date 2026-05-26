"""
SimpleText Task 1: Simplify Scientific Text
Task 1.1: Sentence-level simplification
Task 1.2: Document-level simplification

Uses Claude to simplify biomedical abstracts from Cochrane systematic reviews.
Submission format: JSON with prediction field added to source data.
"""

import anthropic
import json
import os
import sys
import time
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

TEAM_ID = "writerslogic"


class Simplifier:
    def __init__(self, model="claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic()
        self.model = model

    def simplify_sentence(self, complex_sentence: str) -> str:
        prompt = f"""Simplify the following biomedical sentence for a general audience.
Make it easier to understand while preserving the key factual information.
Use simpler words, shorter sentences, and avoid jargon.
Do NOT add any information not present in the original.
Do NOT include any preamble or explanation - output ONLY the simplified sentence.

Complex sentence: {complex_sentence}

Simplified sentence:"""

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            log.warning(f"API error: {e}")
            time.sleep(2)
            return complex_sentence

    def simplify_document(self, complex_document: str) -> str:
        prompt = f"""Simplify the following biomedical abstract for a general audience.
Rewrite it as a plain language summary that is easy to understand.
Use simpler words, shorter sentences, and avoid technical jargon.
You may merge, split, or reorder sentences for clarity.
Do NOT add any information not present in the original.
Do NOT include any preamble, title, or explanation - output ONLY the simplified text.

Complex abstract:
{complex_document}

Plain language summary:"""

        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1000,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text.strip()
        except Exception as e:
            log.warning(f"API error: {e}")
            time.sleep(2)
            return complex_document


def run_task11(input_path: str, output_path: str, model: str = "claude-haiku-4-5-20251001"):
    """Task 1.1: Sentence-level simplification."""
    log.info(f"Task 1.1: Loading {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    log.info(f"  {len(data)} sentences to simplify")

    simplifier = Simplifier(model=model)
    results = []

    for item in tqdm(data, desc="Task 1.1"):
        prediction = simplifier.simplify_sentence(item["complex"])
        result = {**item, "prediction": prediction, "run_id": f"{TEAM_ID}_task11_claude"}
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"  Saved {len(results)} predictions to {output_path}")


def run_task12(input_path: str, output_path: str, model: str = "claude-haiku-4-5-20251001"):
    """Task 1.2: Document-level simplification."""
    log.info(f"Task 1.2: Loading {input_path}")
    with open(input_path) as f:
        data = json.load(f)
    log.info(f"  {len(data)} documents to simplify")

    simplifier = Simplifier(model=model)
    results = []

    for item in tqdm(data, desc="Task 1.2"):
        prediction = simplifier.simplify_document(item["complex"])
        result = {**item, "prediction": prediction, "run_id": f"{TEAM_ID}_task12_claude"}
        results.append(result)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log.info(f"  Saved {len(results)} predictions to {output_path}")


def convert_csv_to_json(csv_path: str, json_path: str, level: str = "sentence"):
    """Convert cochrane-auto CSV to the JSON format expected by CodaBench."""
    import csv
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    data = []
    if level == "sentence":
        for r in rows:
            data.append({
                "pair_id": r["pair_id"],
                "para_id": int(r["para_id"]),
                "sent_id": int(r["sent_id"]),
                "complex": r["complex"],
            })
    else:
        for r in rows:
            data.append({
                "pair_id": r["pair_id"],
                "source": "Cochrane",
                "complex": r["complex"],
            })

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Converted {csv_path} -> {json_path} ({len(data)} entries)")
    return json_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["1.1", "1.2", "both", "convert"], default="both")
    parser.add_argument("--input", help="Input JSON file (or CSV for convert)")
    parser.add_argument("--output", help="Output JSON file")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--level", choices=["sentence", "document"], default="sentence",
                        help="For convert: sentence or document level")
    args = parser.parse_args()

    if args.task == "convert":
        convert_csv_to_json(args.input, args.output, level=args.level)
    elif args.task == "1.1":
        run_task11(args.input, args.output, model=args.model)
    elif args.task == "1.2":
        run_task12(args.input, args.output, model=args.model)
    elif args.task == "both":
        # Default: convert test CSVs and run both tasks
        os.makedirs("output", exist_ok=True)

        # Convert
        sent_json = convert_csv_to_json(
            "cochrane-auto/data/cochraneauto_sents_test.csv",
            "output/test_sentences.json", level="sentence")
        doc_json = convert_csv_to_json(
            "cochrane-auto/data/cochraneauto_docs_test.csv",
            "output/test_documents.json", level="document")

        # Run
        run_task11(sent_json, "output/task11_predictions.json", model=args.model)
        run_task12(doc_json, "output/task12_predictions.json", model=args.model)
