from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(path: str | Path, sample_rows: int | None = None) -> pd.DataFrame:
    """Load CSV/Parquet data without silently changing its schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(path)
        return frame.head(sample_rows) if sample_rows else frame
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, nrows=sample_rows, low_memory=False)
    raise ValueError(f"Unsupported dataset format: {path.suffix}")


def load_payload_dataset(
    path: str | Path, payload_prefix_bytes: int = 256, sample_rows: int | None = None
) -> pd.DataFrame:
    """Load ordered payload bytes plus metadata, avoiding thousands of unused CSV columns."""
    path = Path(path)
    if path.suffix.lower() != ".csv":
        return load_dataset(path, sample_rows)
    columns = list(pd.read_csv(path, nrows=0).columns)
    payload = sorted(
        (column for column in columns if column.startswith("payload_byte_")),
        key=lambda column: int(column.rsplit("_", 1)[-1]),
    )[:payload_prefix_bytes]
    metadata = [column for column in columns if not column.startswith("payload_byte_")]
    # Payload archives are ordered by attack category. Reading only the first N rows can therefore
    # produce a one-class "quick" experiment. Load the reduced column projection, then sample rows.
    frame = pd.read_csv(path, usecols=payload + metadata, low_memory=False)
    if sample_rows and sample_rows < len(frame):
        frame = frame.sample(n=sample_rows, random_state=42).reset_index(drop=True)
    return frame


def make_binary_target(labels: pd.Series, benign_labels: Iterable[str]) -> pd.Series:
    """Return 0 for benign/normal traffic and 1 for every attack category."""
    numeric = pd.to_numeric(labels, errors="coerce")
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        return numeric.astype("int8").rename("is_attack")
    benign = {str(value).strip().lower() for value in benign_labels}
    normalized = labels.astype(str).str.strip().str.lower()
    return (~normalized.isin(benign)).astype("int8").rename("is_attack")


def inject_symmetric_label_noise(
    y: pd.Series, rate: float, random_state: int = 42
) -> tuple[pd.Series, np.ndarray]:
    """Flip a reproducible fraction of binary training labels for robustness experiments."""
    if not 0 <= rate < 0.5:
        raise ValueError("label noise rate must be in [0, 0.5)")
    noisy = y.copy()
    count = round(len(y) * rate)
    rng = np.random.default_rng(random_state)
    positions = rng.choice(len(y), size=count, replace=False) if count else np.array([], dtype=int)
    noisy.iloc[positions] = 1 - noisy.iloc[positions]
    return noisy, positions


def split_features_target(
    frame: pd.DataFrame, target: str, benign_labels: Iterable[str]
) -> tuple[pd.DataFrame, pd.Series]:
    if target not in frame.columns:
        raise KeyError(f"Target column '{target}' not found. Available: {list(frame.columns)}")
    return frame.drop(columns=[target]), make_binary_target(frame[target], benign_labels)
