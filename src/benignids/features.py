from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class PayloadHexFeatures(BaseEstimator, TransformerMixin):
    """Convert payload-like hex/text values into bounded numerical summaries."""

    def __init__(self, prefix_bytes: int = 64):
        self.prefix_bytes = prefix_bytes

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        series = pd.Series(np.asarray(X).reshape(-1)).fillna("").astype(str)
        rows = [self._summarize(value) for value in series]
        return np.asarray(rows, dtype=np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["payload_length", "payload_entropy", "payload_printable_ratio"])

    def _summarize(self, value: str) -> tuple[float, float, float]:
        compact = re.sub(r"[^0-9A-Fa-f]", "", value)
        if len(compact) % 2:
            compact = compact[:-1]
        try:
            raw = bytes.fromhex(compact[: self.prefix_bytes * 2])
        except ValueError:
            raw = value.encode("utf-8", errors="ignore")[: self.prefix_bytes]
        if not raw:
            return 0.0, 0.0, 0.0
        _, counts = np.unique(np.frombuffer(raw, dtype=np.uint8), return_counts=True)
        probabilities = counts / counts.sum()
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        printable = float(sum(32 <= byte < 127 for byte in raw) / len(raw))
        return float(len(raw)), entropy, printable
