"""Fine-tune shidey/deberta-v3-mednli-scifact model on our training data.

Already trained for biomedical NLI/fact verification. We adapt it
to our specific hallucination detection task with 350K labeled examples.

Uses fp32 to avoid DeBERTa NaN issues.
Retrieves best source sentence as premise per candidate.

Usage:
    uv run modal run --detach modal_finetune_shidey.py
"""

import modal
import logging

log = logging.getLogger(__name__)

app = modal.App("simpletext-shidey")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.0",
        "transformers>=4.40",
        "accelerate",
        "scikit-learn",
        "numpy",
        "tqdm",
        "tiktoken",
        "sentencepiece",
        "protobuf",
    )
)

vol = modal.Volume.from_name("simpletext-checkpoints", create_if_missing=True)
CHECKPOINT_DIR = "/checkpoints"


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=36000,
    memory=32768,
    volumes={CHECKPOINT_DIR: vol},
)
def train_and_predict(
    train_records: list[dict],
    dev_records: list[dict],
    test_records: list[dict],
) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
    from sklearn.metrics import f1_score
    from tqdm import tqdm
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    device = torch.device("cuda")
    MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
    MAX_LEN = 256
    BATCH_SIZE = 32
    GRAD_ACCUM = 1
    EPOCHS = 3
    LR = 5e-6

    log.info("Device: %s, GPU: %s", device, torch.cuda.get_device_name(0))
    log.info("Train: %d, Dev: %d, Test: %d", len(train_records), len(dev_records), len(test_records))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    class NLIDataset(Dataset):
        def __init__(self, records, has_labels=True):
            self.records = records
            self.has_labels = has_labels

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            r = self.records[idx]
            enc = tokenizer(
                r["premise"],
                r["hypothesis"],
                max_length=MAX_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            item = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
            }
            if self.has_labels:
                item["label"] = torch.tensor(r["label"], dtype=torch.long)
            return item

    train_ds = NLIDataset(train_records)
    dev_ds = NLIDataset(dev_records)
    test_ds = NLIDataset(test_records, has_labels=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
                             num_workers=4, pin_memory=True)

    log.info("Loading model %s in fp32...", MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True,
        torch_dtype=torch.float32)
    model.to(device)

    pos_count = sum(1 for r in train_records if r["label"] == 1)
    neg_count = len(train_records) - pos_count
    pos_weight_val = neg_count / max(pos_count, 1)
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor([1.0, min(pos_weight_val, 10.0)], device=device))
    log.info("Class weights: neg=1.0 pos=%.2f (capped at 10)", min(pos_weight_val, 10.0))

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * 0.06), total_steps)

    best_f1 = -1
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    import os as _os
    ckpt_path = _os.path.join(CHECKPOINT_DIR, "latest.pt")
    start_epoch = 0
    start_step = 0
    if _os.path.exists(ckpt_path):
        log.info("Resuming from checkpoint %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt.get("epoch", 0)
        start_step = ckpt.get("step", 0)
        log.info("Resumed at epoch %d step %d", start_epoch, start_step)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        total_loss = 0
        optimizer.zero_grad()

        skip_to = start_step if epoch == start_epoch else 0

        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}",
                                          initial=skip_to, total=len(train_loader))):
            if step < skip_to:
                continue

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(input_ids=ids, attention_mask=mask)
            raw_loss = loss_fn(outputs.logits.float(), labels)
            loss = raw_loss / GRAD_ACCUM
            loss.backward()
            total_loss += raw_loss.item()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if (step + 1) % 1000 == 0:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "step": step + 1,
                }, ckpt_path)
                vol.commit()
                log.info("  Checkpoint saved at epoch %d step %d", epoch, step + 1)

        avg_loss = total_loss / max(len(train_loader) - skip_to, 1)

        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch + 1,
            "step": 0,
        }, ckpt_path)
        vol.commit()
        log.info("  End of epoch %d checkpoint saved", epoch + 1)

        model.eval()
        dev_probs, dev_labels = [], []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                dev_probs.extend(probs)
                dev_labels.extend(batch["label"].numpy())

        dev_probs_arr = np.array(dev_probs)
        dev_labels_arr = np.array(dev_labels)

        best_t, f1 = 0.5, 0
        for t in np.linspace(0.1, 0.9, 200):
            f = f1_score(dev_labels_arr, (dev_probs_arr > t).astype(int))
            if f > f1:
                f1, best_t = f, t

        log.info("  Epoch %d: loss=%.4f dev_sent_F1=%.4f t=%.3f",
                 epoch + 1, avg_loss, f1, best_t)

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            log.info("  New best! Saved state.")

    log.info("Best dev sent F1: %.4f", best_f1)

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    dev_probs = []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Dev predict"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            dev_probs.extend(probs)

    test_probs = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Test predict"):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            logits = model(input_ids=ids, attention_mask=mask).logits
            probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
            test_probs.extend(probs)

    return {
        "dev_probs": np.array(dev_probs).tolist(),
        "test_probs": np.array(test_probs).tolist(),
        "best_dev_f1": best_f1,
    }


@app.local_entrypoint()
def main():
    import json
    import re
    import numpy as np
    from pathlib import Path
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    def split_source_sentences(source):
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', source)
        return [p.strip() for p in parts if p.strip()]

    def build_records(docs, has_labels=True):
        records = []
        for d in docs:
            source_sents = split_source_sentences(d["source"])
            for i, sent in enumerate(d["sentences"]):
                best_premise = d["source"][:1500]
                if source_sents:
                    sent_set = set(sent.lower().split())
                    best_overlap = 0
                    for src_sent in source_sents:
                        src_set = set(src_sent.lower().split())
                        overlap = len(sent_set & src_set) / max(len(sent_set), 1)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_premise = src_sent

                r = {"premise": best_premise, "hypothesis": sent}
                if has_labels:
                    r["label"] = 0 if d["labels"][i] == "None" else 1
                records.append(r)
        return records

    log.info("Loading data...")
    train_docs = json.load(open("train_data.json"))
    dev_docs = json.load(open("dev_data.json"))
    test_docs = json.load(open("test_data.json"))

    train_records = build_records(train_docs)
    dev_records = build_records(dev_docs)
    test_records = build_records(test_docs, has_labels=False)

    log.info("Train: %d, Dev: %d, Test: %d", len(train_records), len(dev_records), len(test_records))

    log.info("Submitting to Modal GPU...")
    result = train_and_predict.remote(train_records, dev_records, test_records)

    log.info("Best dev sent F1: %.4f", result["best_dev_f1"])

    Path("cache").mkdir(exist_ok=True)
    np.save("cache/shidey_dev_probs.npy", np.array(result["dev_probs"]))
    np.save("cache/shidey_test_probs.npy", np.array(result["test_probs"]))
    log.info("Saved to cache/shidey_dev_probs.npy and cache/shidey_test_probs.npy")

    from sklearn.metrics import f1_score
    import pandas as pd

    dev_probs = np.array(result["dev_probs"])
    y_dev = np.array([0 if d["labels"][i] == "None" else 1
                      for d in dev_docs for i in range(len(d["sentences"]))])
    doc_ids = [d["id"] for d in dev_docs for _ in d["sentences"]]
    gt_doc = {d["id"]: int(any(l != "None" for l in d["labels"])) for d in dev_docs}

    for t in [0.3, 0.4, 0.5]:
        df = pd.DataFrame({"doc_id": doc_ids, "prob": dev_probs})
        dp = {did: int(g["prob"].max() > t) for did, g in df.groupby("doc_id")}
        y_t = [gt_doc[d] for d in gt_doc]
        y_p = [dp.get(d, 0) for d in gt_doc]
        f1 = f1_score(y_t, y_p)
        log.info("  Doc F1 at t=%.2f: %.4f", t, f1)

    log.info("Done.")
