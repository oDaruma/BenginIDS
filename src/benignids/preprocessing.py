from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .features import PayloadHexFeatures


@dataclass(frozen=True)
class FeatureGroups:
    numerical: tuple[str, ...]
    categorical: tuple[str, ...]
    payload: tuple[str, ...]
    dropped: tuple[str, ...]


def infer_feature_groups(
    frame: pd.DataFrame,
    drop_features: Iterable[str] = (),
    payload_column: str | None = "payload",
) -> FeatureGroups:
    """Resolve columns against the current frame, including safe empty groups."""
    requested_drop = set(drop_features)
    dropped = tuple(sorted(requested_drop.intersection(frame.columns)))
    available = frame.drop(columns=list(dropped), errors="ignore")
    payload = (payload_column,) if payload_column and payload_column in available.columns else ()
    ordinary = available.drop(columns=list(payload), errors="ignore")
    numerical = tuple(ordinary.select_dtypes(include="number").columns)
    categorical = tuple(column for column in ordinary.columns if column not in numerical)
    return FeatureGroups(numerical, categorical, payload, dropped)


def build_preprocessor(
    frame: pd.DataFrame,
    drop_features: Iterable[str] = (),
    payload_column: str | None = "payload",
    payload_prefix_bytes: int = 64,
) -> tuple[ColumnTransformer, FeatureGroups]:
    """Build only non-empty transformer branches (the v3.5.6 robustness invariant)."""
    groups = infer_feature_groups(frame, drop_features, payload_column)
    transformers = []
    if groups.numerical:
        numerical = Pipeline(
            [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        )
        transformers.append(("numeric", numerical, list(groups.numerical)))
    if groups.categorical:
        categorical = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]
        )
        transformers.append(("categorical", categorical, list(groups.categorical)))
    if groups.payload:
        transformers.append(
            ("payload", PayloadHexFeatures(prefix_bytes=payload_prefix_bytes), list(groups.payload))
        )
    if not transformers:
        raise ValueError("No usable features remain after applying drop_features")
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True), groups
