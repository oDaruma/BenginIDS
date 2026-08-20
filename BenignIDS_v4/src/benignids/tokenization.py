from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PAD, CLS, MASK, UNK = 0, 1, 2, 3


@dataclass(frozen=True)
class TokenizerConfig:
    vocab_size: int = 16384
    max_length: int = 256
    payload_prefix_bytes: int = 64


class TrafficTokenizer:
    """Deterministically tokenize packet/flow fields without learning from test data."""

    def __init__(self, config: TokenizerConfig | None = None):
        self.config = config or TokenizerConfig()
        if self.config.vocab_size <= 260:
            raise ValueError("vocab_size must leave room for special and byte tokens")

    def _hash_token(self, text: str) -> int:
        digest = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
        return 260 + int.from_bytes(digest, "big") % (self.config.vocab_size - 260)

    @staticmethod
    def _bucket_number(value) -> str:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "missing"
        if not np.isfinite(number):
            return "missing"
        sign = "neg" if number < 0 else "pos"
        magnitude = int(np.floor(np.log10(abs(number) + 1)))
        mantissa = int(abs(number) / (10**magnitude) * 10) if number else 0
        return f"{sign}:e{magnitude}:m{mantissa}"

    def encode_record(self, record: dict, exclude: Iterable[str] = ("label", "attack_cat")):
        excluded = set(exclude)
        tokens = [CLS]
        byte_fields = sorted(
            (field for field in record if field.startswith("payload_byte_")),
            key=lambda field: int(field.rsplit("_", 1)[-1]),
        )
        ordinary_fields = sorted(
            field for field in record if field not in excluded and field not in byte_fields
        )
        # Metadata tokens come first, followed by payload bytes in packet order.
        for field in ordinary_fields:
            if field in excluded:
                continue
            value = record[field]
            tokens.append(self._hash_token(f"field:{field}"))
            if field.lower() == "payload":
                compact = "".join(character for character in str(value) if character in "0123456789abcdefABCDEF")
                if len(compact) % 2:
                    compact = compact[:-1]
                try:
                    raw = bytes.fromhex(compact[: self.config.payload_prefix_bytes * 2])
                except ValueError:
                    raw = str(value).encode(errors="ignore")[: self.config.payload_prefix_bytes]
                tokens.extend(4 + byte for byte in raw)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                tokens.append(self._hash_token(f"{field}:num:{self._bucket_number(value)}"))
            else:
                tokens.append(self._hash_token(f"{field}:cat:{str(value).strip().lower()}"))
            if len(tokens) >= self.config.max_length:
                break
        for field in byte_fields:
            try:
                byte = int(record[field])
            except (TypeError, ValueError):
                byte = 0
            tokens.append(4 + min(max(byte, 0), 255))
            if len(tokens) >= self.config.max_length:
                break
        attention = [1] * len(tokens)
        padding = self.config.max_length - len(tokens)
        return tokens + [PAD] * padding, attention + [0] * padding

    def encode_frame(self, frame: pd.DataFrame):
        encoded = [self.encode_record(record) for record in frame.to_dict(orient="records")]
        input_ids, attention_mask = zip(*encoded)
        return np.asarray(input_ids, dtype=np.int64), np.asarray(attention_mask, dtype=np.int64)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.config.__dict__, handle, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> TrafficTokenizer:
        path = Path(path)
        with path.open(encoding="utf-8") as handle:
            settings = json.load(handle)
        return cls(TokenizerConfig(**settings))


def mask_tokens(input_ids, attention_mask, mask_probability=0.15, random_state=42, vocab_size=16384):
    """Create BERT-style masked-token targets; -100 positions are ignored by loss."""
    rng = np.random.default_rng(random_state)
    masked = np.asarray(input_ids).copy()
    labels = np.full_like(masked, -100)
    candidates = (np.asarray(attention_mask) == 1) & (masked >= 4)
    selected = candidates & (rng.random(masked.shape) < mask_probability)
    labels[selected] = masked[selected]
    replace_with_mask = selected & (rng.random(masked.shape) < 0.8)
    masked[replace_with_mask] = MASK
    replace_random = selected & ~replace_with_mask & (rng.random(masked.shape) < 0.5)
    masked[replace_random] = rng.integers(4, vocab_size, size=replace_random.sum())
    return masked, labels
