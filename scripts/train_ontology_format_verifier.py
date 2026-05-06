from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "datasets" / "main_trainable_dataset"
GROUPED_DATA_ROOT = ROOT / "datasets" / "original_verifier_dataset"
MODEL_ROOT = ROOT / "models"
RESULTS_ROOT = ROOT / "results" / "training"

LABEL2ID = {"INVALID": 0, "VALID": 1}
ID2LABEL = {0: "INVALID", 1: "VALID"}

SPECIAL_TOKENS = {
    "additional_special_tokens": [
        "[REL]",
        "[/REL]",
        "[E1]",
        "[/E1]",
        "[E2]",
        "[/E2]",
        "[SUBJ]",
        "[/SUBJ]",
        "[SUBJ_TYPE]",
        "[/SUBJ_TYPE]",
        "[OBJ]",
        "[/OBJ]",
        "[OBJ_TYPE]",
        "[/OBJ_TYPE]",
        "[DOMAIN]",
        "[/DOMAIN]",
        "[RANGE]",
        "[/RANGE]",
        "[VALIDATION]",
        "[/VALIDATION]",
        "[REL_TYPE]",
        "[/REL_TYPE]",
        "[REL_CHARACTERISTICS]",
        "[/REL_CHARACTERISTICS]",
        "[CARDINALITY_HINT]",
        "[/CARDINALITY_HINT]",
        "[INVERSE_OF]",
        "[/INVERSE_OF]",
        "[SUBPROPERTY_OF]",
        "[/SUBPROPERTY_OF]",
        "[DEPRECATED_RELATION]",
        "[/DEPRECATED_RELATION]",
        "[INFERRED_SUBJ_TYPES]",
        "[/INFERRED_SUBJ_TYPES]",
        "[INFERRED_OBJ_TYPES]",
        "[/INFERRED_OBJ_TYPES]",
        "[SUBJ_SUPERCLASSES]",
        "[/SUBJ_SUPERCLASSES]",
        "[OBJ_SUPERCLASSES]",
        "[/OBJ_SUPERCLASSES]",
        "[SUBJ_DISJOINT_GROUPS]",
        "[/SUBJ_DISJOINT_GROUPS]",
        "[OBJ_DISJOINT_GROUPS]",
        "[/OBJ_DISJOINT_GROUPS]",
        "[UNIT_CONTEXT]",
        "[/UNIT_CONTEXT]",
    ]
}


class EncodedDataset(Dataset):
    def __init__(self, encodings: dict[str, torch.Tensor], labels: list[int]):
        self.encodings = encodings
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_format_rows(format_name: str, data_root: Path = DATA_ROOT) -> tuple[list[dict], list[dict], list[dict]]:
    format_dir = data_root / format_name
    return (
        load_jsonl(format_dir / "train.jsonl"),
        load_jsonl(format_dir / "val.jsonl"),
        load_jsonl(format_dir / "test.jsonl"),
    )


def rows_to_xy(rows: list[dict[str, Any]]) -> tuple[list[str], list[int]]:
    return [row["text"] for row in rows], [LABEL2ID[row["label"]] for row in rows]


def choose_max_len(tokenizer, texts: list[str], cap: int = 256) -> int:
    lengths = [
        len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
        for text in texts[:5000]
    ]
    return max(64, min(cap, int(np.percentile(lengths, 99)))) if lengths else 128


def compute_metrics(gold: list[int], pred: list[int]) -> dict[str, Any]:
    tp = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 1)
    tn = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 0)
    fp = sum(1 for g, p in zip(gold, pred) if g == 0 and p == 1)
    fn = sum(1 for g, p in zip(gold, pred) if g == 1 and p == 0)
    total = len(gold)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "support": total,
    }


def autocast_ctx():
    if torch.cuda.is_available():
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[list[int], list[int], float]:
    model.eval()
    preds: list[int] = []
    golds: list[int] = []
    losses: list[float] = []
    for batch in loader:
        labels = batch["labels"]
        batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
        with autocast_ctx():
            out = model(**batch)
        losses.append(float(out.loss.detach().cpu()))
        preds.extend(out.logits.argmax(dim=1).cpu().numpy().tolist())
        golds.extend(labels.numpy().tolist())
    mean_loss = float(np.mean(losses)) if losses else 0.0
    return golds, preds, mean_loss


def train_format(
    format_name: str,
    model_name: str = "bert-base-uncased",
    epochs: int = 3,
    batch_size: int = 32,
    max_len: int | None = None,
    data_root: Path = DATA_ROOT,
    output_model_dir: Path | None = None,
    output_metrics_path: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    train_rows, val_rows, test_rows = load_format_rows(format_name, data_root=data_root)
    train_x, train_y = rows_to_xy(train_rows)
    val_x, val_y = rows_to_xy(val_rows)
    test_x, test_y = rows_to_xy(test_rows)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens(SPECIAL_TOKENS)
    if max_len is None:
        max_len = choose_max_len(tokenizer, train_x)

    def encode(texts: list[str]) -> dict[str, torch.Tensor]:
        return tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_len,
            return_tensors="pt",
        )

    train_ds = EncodedDataset(encode(train_x), train_y)
    val_ds = EncodedDataset(encode(val_x), val_y)
    test_ds = EncodedDataset(encode(test_x), test_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=torch.cuda.is_available())
    eval_batch_size = batch_size * 2
    train_eval_loader = DataLoader(train_ds, batch_size=eval_batch_size, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, pin_memory=torch.cuda.is_available())
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.config.label2id = LABEL2ID
    model.config.id2label = ID2LABEL
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = max(1, len(train_loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    history: list[dict[str, Any]] = []
    best_state = None
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"{format_name} epoch {epoch}/{epochs}")
        for batch in pbar:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx():
                out = model(**batch)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_value = float(loss.detach().cpu())
            train_losses.append(loss_value)
            pbar.set_postfix(loss=f"{loss_value:.4f}")

        val_gold, val_pred, val_loss = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_gold, val_pred)
        epoch_payload = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_loss": val_loss,
            "val": val_metrics,
        }
        history.append(epoch_payload)

        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    train_gold, train_pred, train_loss = evaluate(model, train_eval_loader, device)
    val_gold, val_pred, val_loss = evaluate(model, val_loader, device)
    test_gold, test_pred, test_loss = evaluate(model, test_loader, device)

    payload = {
        "format": format_name,
        "model_name": model_name,
        "data_root": str(data_root),
        "epochs": epochs,
        "batch_size": batch_size,
        "max_len": max_len,
        "seed": seed,
        "history": history,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "train": compute_metrics(train_gold, train_pred),
        "val": compute_metrics(val_gold, val_pred),
        "test": compute_metrics(test_gold, test_pred),
    }

    if output_model_dir is None:
        output_model_dir = MODEL_ROOT / format_name
    output_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)

    if output_metrics_path is None:
        output_metrics_path = RESULTS_ROOT / f"{format_name}_metrics.json"
    output_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    output_metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", required=True)
    parser.add_argument("--model-name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-model-dir", type=Path, default=None)
    parser.add_argument("--output-metrics-path", type=Path, default=None)
    args = parser.parse_args()
    payload = train_format(
        format_name=args.format,
        model_name=args.model_name,
        epochs=args.epochs,
        batch_size=args.batch_size,
        max_len=args.max_len,
        data_root=args.data_root,
        output_model_dir=args.output_model_dir,
        output_metrics_path=args.output_metrics_path,
        seed=args.seed,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
