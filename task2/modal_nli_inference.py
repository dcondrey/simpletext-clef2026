"""Run NLI inference using shidey/deberta-v3-mednli-scifact model on Modal.

Generates entailment/neutral/contradiction scores for all sentences.
No training needed -- uses pretrained checkpoint directly.

Usage:
    uv run modal run --detach modal_nli_inference.py
"""

import modal
import logging

log = logging.getLogger(__name__)

app = modal.App("simpletext-nli")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.40",
        "accelerate",
        "numpy",
        "tqdm",
    )
)


@app.function(
    image=image,
    gpu="A100",
    timeout=7200,
)
def run_nli(records: list[dict], model_name: str) -> list[dict]:
    import numpy as np
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from tqdm import tqdm
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    device = torch.device("cuda")
    log.info("Device: %s, GPU: %s", device, torch.cuda.get_device_name(0))
    log.info("Model: %s, Records: %d", model_name, len(records))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, torch_dtype=torch.float32)
    model.to(device)
    model.eval()

    label_names = model.config.id2label
    log.info("Labels: %s", label_names)

    results = []
    batch_size = 64

    for start in tqdm(range(0, len(records), batch_size), desc="NLI"):
        batch = records[start:start + batch_size]
        premises = [r["premise"] for r in batch]
        hypotheses = [r["hypothesis"] for r in batch]

        enc = tokenizer(
            premises, hypotheses,
            max_length=512, truncation=True, padding=True,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            probs = torch.softmax(outputs.logits.float(), dim=1).cpu().numpy()

        for i, row in enumerate(probs):
            result = {}
            for j, label in label_names.items():
                result[label.lower()] = float(row[j])
            results.append(result)

    log.info("Processed %d records", len(results))
    return results


@app.local_entrypoint()
def main():
    import json
    import re
    import numpy as np
    from pathlib import Path
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"

    def split_source_sentences(source):
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', source)
        return [p.strip() for p in parts if p.strip()]

    def build_records(docs):
        records = []
        for d in docs:
            source_sents = split_source_sentences(d["source"])
            for i, sent in enumerate(d["sentences"]):
                best_premise = d["source"][:1500]
                if source_sents:
                    sent_lower = sent.lower().split()
                    sent_set = set(sent_lower)
                    best_overlap = 0
                    for src_sent in source_sents:
                        src_set = set(src_sent.lower().split())
                        overlap = len(sent_set & src_set) / max(len(sent_set), 1)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_premise = src_sent

                records.append({
                    "premise": best_premise,
                    "hypothesis": sent,
                    "doc_id": d["id"],
                    "sent_idx": i,
                })
        return records

    log.info("Loading data...")
    train_docs = json.load(open("train_data.json"))
    dev_docs = json.load(open("dev_data.json"))
    test_docs = json.load(open("test_data.json"))

    Path("cache").mkdir(exist_ok=True)

    for split_name, docs in [("train", train_docs), ("dev", dev_docs), ("test", test_docs)]:
        out_path = f"cache/nli_{split_name}_scores.npz"
        if Path(out_path).exists() and split_name != "train":
            log.info("Skipping %s (already cached)", split_name)
            continue

        records = build_records(docs)
        log.info("Running NLI on %s (%d records)...", split_name, len(records))
        results = run_nli.remote(records, MODEL_NAME)

        scores = {
            "entailment": [r.get("entailment", 0) for r in results],
            "neutral": [r.get("neutral", 0) for r in results],
            "contradiction": [r.get("contradiction", 0) for r in results],
        }
        np.savez(out_path, **scores)

        contra = np.array(scores["contradiction"])
        log.info("%s: contradiction mean=%.3f, >0.5: %d/%d (%.1f%%)",
                 split_name, contra.mean(), (contra > 0.5).sum(), len(contra),
                 (contra > 0.5).sum() / len(contra) * 100)

    log.info("Done.")
