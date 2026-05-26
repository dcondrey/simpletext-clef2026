"""Fine-tune DeBERTa-v3-large for hallucination detection on Modal GPU.

NLI-style: source is premise, sentence is hypothesis.
Returns dev and test probabilities for blending with LightGBM.

Usage:
    uv run modal run --detach modal_finetune_deberta.py
"""

import modal
import logging

log = logging.getLogger(__name__)

app = modal.App("simpletext-deberta")

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


@app.function(
    image=image,
    gpu="A100",
    timeout=36000,
)
def train_and_predict(
    train_data: list[dict],
    dev_data: list[dict],
    test_data: list[dict],
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
    MODEL_NAME = "microsoft/deberta-v3-base"
    MAX_LEN = 512
    BATCH_SIZE = 32
    GRAD_ACCUM = 1
    EPOCHS = 3
    LR = 2e-5
    SEEDS = [42]

    log.info("Device: %s, GPU: %s", device, torch.cuda.get_device_name(0))
    log.info("Train: %d, Dev: %d, Test: %d", len(train_data), len(dev_data), len(test_data))

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
                r["source"][:2000],
                r["sentence"],
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

    train_ds = NLIDataset(train_data)
    dev_ds = NLIDataset(dev_data)
    test_ds = NLIDataset(test_data, has_labels=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 4, shuffle=False,
                             num_workers=4, pin_memory=True)

    all_dev_probs = []
    all_test_probs = []

    for seed in SEEDS:
        log.info("=== Seed %d ===", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2, torch_dtype=torch.float32)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        total_steps = (len(train_loader) // GRAD_ACCUM) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer, int(total_steps * 0.1), total_steps)

        best_f1 = -1
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        pos_count = sum(1 for r in train_data if r["label"] == 1)
        neg_count = len(train_data) - pos_count
        pos_weight_val = neg_count / max(pos_count, 1)
        loss_fn = nn.CrossEntropyLoss(weight=torch.tensor([1.0, pos_weight_val], device=device))
        log.info("  Class weights: neg=1.0 pos=%.2f", pos_weight_val)

        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            optimizer.zero_grad()
            for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
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

            log.info("  Epoch %d: loss=%.4f dev_F1=%.4f t=%.3f",
                     epoch + 1, total_loss / len(train_loader), f1, best_t)

            if f1 > best_f1:
                best_f1 = f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        log.info("  Best F1: %.4f", best_f1)

        model.load_state_dict(best_state)
        model.to(device)
        model.eval()

        dev_probs = []
        with torch.no_grad():
            for batch in dev_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                dev_probs.extend(probs)
        all_dev_probs.append(dev_probs)

        test_probs = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Test"):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                logits = model(input_ids=ids, attention_mask=mask).logits
                probs = torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy()
                test_probs.extend(probs)
        all_test_probs.append(test_probs)

        del model
        torch.cuda.empty_cache()

    ens_dev = np.mean(all_dev_probs, axis=0)
    ens_test = np.mean(all_test_probs, axis=0)

    y_dev = np.array([r["label"] for r in dev_data])
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.1, 0.9, 200):
        f1 = f1_score(y_dev, (ens_dev > t).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    log.info("=== Ensemble: dev sent F1=%.4f t=%.3f ===", best_f1, best_t)

    return {
        "dev_probs": ens_dev.tolist(),
        "test_probs": ens_test.tolist(),
        "best_threshold": best_t,
        "best_dev_f1": best_f1,
    }


@app.local_entrypoint()
def main():
    import json
    import numpy as np
    from pathlib import Path
    import logging

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    log = logging.getLogger(__name__)

    log.info("Loading data...")
    train_docs = json.load(open("train_data.json"))
    dev_docs = json.load(open("dev_data.json"))
    test_docs = json.load(open("test_data.json"))

    def flatten(docs, has_labels=True):
        records = []
        for d in docs:
            source = d["source"]
            for i, sent in enumerate(d["sentences"]):
                r = {"source": source, "sentence": sent, "doc_id": d["id"], "sent_idx": i}
                if has_labels:
                    r["label"] = 0 if d["labels"][i] == "None" else 1
                records.append(r)
        return records

    train_flat = flatten(train_docs)
    dev_flat = flatten(dev_docs)
    test_flat = flatten(test_docs, has_labels=False)

    log.info("Train: %d, Dev: %d, Test: %d", len(train_flat), len(dev_flat), len(test_flat))
    log.info("Submitting to Modal GPU...")

    result = train_and_predict.remote(train_flat, dev_flat, test_flat)

    log.info("Dev sent F1: %.4f, threshold: %.3f",
             result["best_dev_f1"], result["best_threshold"])

    Path("cache").mkdir(exist_ok=True)
    np.save("cache/deberta_dev_probs.npy", np.array(result["dev_probs"]))
    np.save("cache/deberta_test_probs.npy", np.array(result["test_probs"]))

    log.info("Saved probabilities to cache/deberta_dev_probs.npy and cache/deberta_test_probs.npy")

    dev_probs = np.array(result["dev_probs"])
    y_dev = np.array([0 if d["labels"][i] == "None" else 1
                      for d in dev_docs for i in range(len(d["sentences"]))])

    from sklearn.metrics import f1_score
    doc_ids = [d["id"] for d in dev_docs for _ in d["sentences"]]
    gt_doc = {d["id"]: int(any(l != "None" for l in d["labels"])) for d in dev_docs}

    import pandas as pd
    for t in [0.3, 0.4, 0.5]:
        df = pd.DataFrame({"doc_id": doc_ids, "prob": dev_probs})
        dp = {did: int(g["prob"].max() > t) for did, g in df.groupby("doc_id")}
        y_t = [gt_doc[d] for d in gt_doc]
        y_p = [dp.get(d, 0) for d in gt_doc]
        f1 = f1_score(y_t, y_p)
        log.info("  DeBERTa doc F1 at t=%.2f: %.4f", t, f1)

    log.info("Done.")
