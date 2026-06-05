from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def get_device(requested_device: str = "auto") -> torch.device:
    """Return the torch device used for training/prediction."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(0)})"
    return str(device)


def tokenize(text: str) -> List[str]:
    """Simple whitespace tokenizer for already-preprocessed input text."""
    return re.findall(r"\S+", (text or "").lower())


def build_vocab(
    rows: Iterable[Dict[str, str]],
    max_vocab_size: int = 50_000,
    min_freq: int = 1,
) -> Dict[str, int]:
    """Build token vocabulary from training input_text fields."""
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(tokenize(row["input_text"]))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token, freq in counter.most_common(max_vocab_size - len(vocab)):
        if freq < min_freq:
            continue
        vocab[token] = len(vocab)
    return vocab


def build_label_mapping(rows: Iterable[Dict[str, str]], label_key: str) -> Tuple[Dict[str, int], List[str]]:
    """Build label <-> id mappings for one prediction target."""
    labels = sorted({row[label_key] for row in rows})
    label_to_id = {label: idx for idx, label in enumerate(labels)}
    return label_to_id, labels


def encode_text(text: str, vocab: Dict[str, int], max_length: int) -> List[int]:
    ids = [vocab.get(token, vocab[UNK_TOKEN]) for token in tokenize(text)]
    if not ids:
        ids = [vocab[UNK_TOKEN]]
    return ids[:max_length]


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Dict[str, str]],
        vocab: Dict[str, int],
        label_to_id: Dict[str, int] | None = None,
        label_key: str | None = None,
        max_length: int = 192,
    ) -> None:
        self.rows = list(rows)
        self.vocab = vocab
        self.label_to_id = label_to_id
        self.label_key = label_key
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Dict[str, object]:
        row = self.rows[index]
        item: Dict[str, object] = {
            "input_ids": encode_text(row["input_text"], self.vocab, self.max_length),
        }
        if self.label_to_id is not None and self.label_key is not None:
            item["label"] = self.label_to_id[row[self.label_key]]
        return item


def collate_batch(batch: Sequence[Dict[str, object]]) -> Dict[str, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)  # type: ignore[arg-type]
    input_ids: List[List[int]] = []
    attention_mask: List[List[int]] = []
    labels: List[int] = []

    for item in batch:
        ids = list(item["input_ids"])  # type: ignore[arg-type]
        pad_len = max_len - len(ids)
        input_ids.append(ids + [0] * pad_len)
        attention_mask.append([1] * len(ids) + [0] * pad_len)
        if "label" in item:
            labels.append(int(item["label"]))

    result = {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.float32),
    }
    if labels:
        result["labels"] = torch.tensor(labels, dtype=torch.long)
    return result


class MeanPoolingTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_labels: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        masked = embeddings * attention_mask.unsqueeze(-1)
        lengths = attention_mask.sum(dim=1).clamp(min=1).unsqueeze(-1)
        pooled = masked.sum(dim=1) / lengths
        return self.classifier(pooled)


def train_label_model(
    rows: Iterable[Dict[str, str]],
    label_key: str,
    vocab: Dict[str, int],
    device: torch.device,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    max_length: int = 192,
    embedding_dim: int = 128,
    hidden_dim: int = 128,
) -> Dict[str, object]:
    """Train one GPU-capable PyTorch text classifier for a target label."""
    row_list = list(rows)
    label_to_id, id_to_label = build_label_mapping(row_list, label_key)
    dataset = TextClassificationDataset(row_list, vocab, label_to_id, label_key, max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)

    model = MeanPoolingTextClassifier(
        vocab_size=len(vocab),
        num_labels=len(id_to_label),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_items = 0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_ids, attention_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * labels.size(0)
            total_items += labels.size(0)

        print(f"  epoch {epoch:02d}/{epochs} loss={total_loss / max(total_items, 1):.4f}")

    return {
        "model": model,
        "label_to_id": label_to_id,
        "id_to_label": id_to_label,
        "config": {
            "vocab_size": len(vocab),
            "num_labels": len(id_to_label),
            "embedding_dim": embedding_dim,
            "hidden_dim": hidden_dim,
            "max_length": max_length,
        },
    }


def predict_with_model(
    rows: Iterable[Dict[str, str]],
    model_bundle: Dict[str, object],
    vocab: Dict[str, int],
    device: torch.device,
    batch_size: int = 256,
) -> List[str]:
    row_list = list(rows)
    model = model_bundle["model"]
    assert isinstance(model, nn.Module)
    id_to_label = model_bundle["id_to_label"]
    assert isinstance(id_to_label, list)
    config = model_bundle["config"]
    assert isinstance(config, dict)

    dataset = TextClassificationDataset(row_list, vocab, max_length=int(config["max_length"]))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    predictions: List[str] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            pred_ids = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(id_to_label[pred_id] for pred_id in pred_ids)
    return predictions


def predict_with_ml_models(
    rows: Iterable[Dict[str, str]],
    expanded_model: Dict[str, object],
    base_model: Dict[str, object],
    vocab: Dict[str, int],
    device: torch.device,
    batch_size: int = 256,
) -> List[Dict[str, str]]:
    """Predict expanded/base forms with PyTorch models only, without dictionary fallback."""
    row_list = list(rows)
    expanded_predictions = predict_with_model(row_list, expanded_model, vocab, device, batch_size)
    base_predictions = predict_with_model(row_list, base_model, vocab, device, batch_size)

    return [
        {
            "expanded": str(expanded).strip(),
            "base": str(base).strip(),
            "abbr": row["abbr"],
            "row_id": row.get("row_id", ""),
        }
        for row, expanded, base in zip(row_list, expanded_predictions, base_predictions)
    ]


def save_model_bundle(bundle: Dict[str, object], model_path: str | Path, vocab: Dict[str, int]) -> None:
    """Save a complete model artifact, including weights, config, labels, and vocabulary."""
    model = bundle["model"]
    assert isinstance(model, nn.Module)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": bundle["config"],
            "id_to_label": bundle["id_to_label"],
            "vocab": vocab,
        },
        model_path,
    )


def load_model_bundle(model_path: str | Path, device: torch.device) -> Dict[str, object]:
    """Load a complete model artifact saved by `save_model_bundle`."""
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint["config"]
    id_to_label = checkpoint["id_to_label"]
    vocab = checkpoint["vocab"]

    model = MeanPoolingTextClassifier(
        vocab_size=int(config["vocab_size"]),
        num_labels=int(config["num_labels"]),
        embedding_dim=int(config["embedding_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return {"model": model, "id_to_label": id_to_label, "config": config, "vocab": vocab}


def split_prediction_columns(predictions: Iterable[Dict[str, str]]) -> Tuple[List[str], List[str]]:
    """Return expanded/base lists from prediction dictionaries."""
    prediction_list = list(predictions)
    return [row["expanded"] for row in prediction_list], [row["base"] for row in prediction_list]