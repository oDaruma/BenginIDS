"""BenignIDS v4 public API."""

__version__ = "4.0.0"

from .data import load_dataset, make_binary_target
from .preprocessing import build_preprocessor, infer_feature_groups

__all__ = ["build_preprocessor", "infer_feature_groups", "load_dataset", "make_binary_target"]

