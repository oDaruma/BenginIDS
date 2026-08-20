from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class TransformerConfig:
    vocab_size: int = 16384
    max_length: int = 256
    d_model: int = 128
    nhead: int = 4
    layers: int = 4
    feedforward: int = 256
    dropout: float = 0.1
    classes: int = 2


def _torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("Install transformer support with: pip install -e '.[deep]'") from exc
    return torch, nn, DataLoader, TensorDataset


def build_traffic_transformer(config: TransformerConfig):
    torch, nn, _, _ = _torch()

    class TrafficTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
            self.position_embedding = nn.Embedding(config.max_length, config.d_model)
            layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.feedforward,
                dropout=config.dropout,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=config.layers)
            self.normalization = nn.LayerNorm(config.d_model)
            self.language_head = nn.Linear(config.d_model, config.vocab_size)
            self.classification_head = nn.Linear(config.d_model, config.classes)

        def encode(self, input_ids, attention_mask):
            positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
            hidden = self.encoder(hidden, src_key_padding_mask=attention_mask == 0)
            return self.normalization(hidden)

        def forward(self, input_ids, attention_mask, task="classification"):
            hidden = self.encode(input_ids, attention_mask)
            if task == "masked_language_model":
                return self.language_head(hidden)
            return self.classification_head(hidden[:, 0])

    return TrafficTransformer()


def _device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pretrain_masked_language_model(
    model,
    masked_input_ids,
    attention_mask,
    token_labels,
    epochs=5,
    batch_size=64,
    learning_rate=3e-4,
    seed=42,
):
    """Self-supervised stage: predict deliberately masked traffic tokens."""
    torch, nn, DataLoader, TensorDataset = _torch()
    torch.manual_seed(seed)
    device = _device(torch)
    model.to(device)
    dataset = TensorDataset(
        torch.as_tensor(masked_input_ids, dtype=torch.long),
        torch.as_tensor(attention_mask, dtype=torch.long),
        torch.as_tensor(token_labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for input_ids, mask, labels in loader:
            input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(input_ids, mask, task="masked_language_model")
            loss = loss_function(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        history.append({"epoch": epoch + 1, "loss": total / max(len(loader), 1)})
    return history


def fine_tune_classifier(
    model,
    train_input_ids,
    train_attention_mask,
    y_train,
    val_input_ids,
    val_attention_mask,
    y_val,
    epochs=10,
    batch_size=64,
    learning_rate=2e-4,
    patience=3,
    seed=42,
):
    """Supervised stage with validation accuracy early stopping for binary or multiclass labels."""
    from sklearn.metrics import accuracy_score

    torch, nn, DataLoader, TensorDataset = _torch()
    torch.manual_seed(seed)
    device = _device(torch)
    model.to(device)
    train_dataset = TensorDataset(
        torch.as_tensor(train_input_ids, dtype=torch.long),
        torch.as_tensor(train_attention_mask, dtype=torch.long),
        torch.as_tensor(np.asarray(y_train), dtype=torch.long),
    )
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    counts = np.bincount(np.asarray(y_train, dtype=int), minlength=model.config.classes)
    weights = counts.sum() / np.maximum(counts, 1)
    loss_function = nn.CrossEntropyLoss(weight=torch.as_tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_score, best_state, stale = -np.inf, None, 0
    history = []
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for input_ids, mask, labels in loader:
            input_ids, mask, labels = input_ids.to(device), mask.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_function(model(input_ids, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach())
        probabilities = predict_probabilities(model, val_input_ids, val_attention_mask, batch_size)
        score = float(accuracy_score(y_val, probabilities.argmax(axis=1)))
        history.append({"epoch": epoch + 1, "loss": total / max(len(loader), 1), "val_accuracy": score})
        if score > best_score + 1e-5:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def predict_scores(model, input_ids, attention_mask, batch_size=128):
    """Return positive-class scores for legacy binary callers."""
    probabilities = predict_probabilities(model, input_ids, attention_mask, batch_size)
    if probabilities.shape[1] != 2:
        raise ValueError("predict_scores requires a binary classifier")
    return probabilities[:, 1]


def predict_probabilities(model, input_ids, attention_mask, batch_size=128):
    """Return calibrated-like softmax outputs for every behavior class."""
    torch, _, DataLoader, TensorDataset = _torch()
    device = _device(torch)
    model.to(device)
    dataset = TensorDataset(
        torch.as_tensor(input_ids, dtype=torch.long),
        torch.as_tensor(attention_mask, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size)
    model.eval()
    probabilities = []
    with torch.no_grad():
        for batch_ids, batch_mask in loader:
            logits = model(batch_ids.to(device), batch_mask.to(device))
            probabilities.extend(torch.softmax(logits, dim=1).cpu().numpy())
    return np.asarray(probabilities)


def save_checkpoint(model, config: TransformerConfig, path: str | Path):
    torch, _, _, _ = _torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": asdict(config), "state_dict": model.state_dict()}, path)


def load_checkpoint(path: str | Path):
    torch, _, _, _ = _torch()
    checkpoint = torch.load(path, map_location="cpu")
    config = TransformerConfig(**checkpoint["config"])
    model = build_traffic_transformer(config)
    model.load_state_dict(checkpoint["state_dict"])
    return model, config
