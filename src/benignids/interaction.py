from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

INTERACTION_LABELS = (
    "HUMAN_INTERACTIVE",
    "MACHINE_AUTOMATED",
    "MIXED",
    "UNKNOWN",
)

INTERACTION_FEATURES = (
    "duration",
    "packets",
    "total_len",
    "forward_packets",
    "reverse_packets",
    "forward_bytes",
    "reverse_bytes",
    "packet_size_mean",
    "packet_size_std",
    "interarrival_mean",
    "interarrival_std",
    "interarrival_cv",
    "packets_per_second",
    "bytes_per_second",
    "direction_changes",
    "request_response_turns",
    "burst_count",
    "idle_fraction",
    "timing_regularity",
    "syn_count",
    "fin_count",
    "rst_count",
)

_ALIASES = {
    "human": "HUMAN_INTERACTIVE",
    "interactive": "HUMAN_INTERACTIVE",
    "human_driven": "HUMAN_INTERACTIVE",
    "machine": "MACHINE_AUTOMATED",
    "automated": "MACHINE_AUTOMATED",
    "machine_to_machine": "MACHINE_AUTOMATED",
    "m2m": "MACHINE_AUTOMATED",
    "hybrid": "MIXED",
    "ambiguous": "UNKNOWN",
}


def normalize_interaction_label(value: object) -> str:
    token = re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")
    normalized = _ALIASES.get(token, token.upper())
    if normalized not in INTERACTION_LABELS:
        raise ValueError(f"Unsupported interaction label: {value!r}")
    return normalized


def _training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(INTERACTION_FEATURES).difference(frame.columns))
    if missing:
        raise ValueError(f"Session data is missing interaction features: {missing}")
    values = frame.loc[:, INTERACTION_FEATURES].apply(pd.to_numeric, errors="coerce")
    if values.dropna(how="all").empty:
        raise ValueError("Session data contains no numeric interaction features")
    return values


def interaction_bundle_path(output_dir: str | Path, model_name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", model_name):
        raise ValueError(
            "Interaction model names may contain only letters, numbers, dots, underscores, "
            "and hyphens"
        )
    return Path(output_dir).resolve() / "interaction_models" / model_name


def train_interaction_model(
    frame: pd.DataFrame,
    output_dir: str | Path,
    model_name: str,
    target: str = "interaction_label",
    random_state: int = 42,
) -> dict:
    if target not in frame:
        raise ValueError(f"Session data is missing target column: {target}")
    features = _training_frame(frame)
    labels = frame[target].map(normalize_interaction_label)
    if labels.nunique() < 2:
        raise ValueError("Interaction training requires at least two classes")
    if labels.value_counts().min() < 2:
        raise ValueError("Every interaction class needs at least two labelled sessions")

    group_column = next(
        (name for name in ("capture_id", "device_id", "endpoint_a") if name in frame), None
    )
    if group_column and frame[group_column].nunique() >= 2:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
        train_index, test_index = next(splitter.split(features, labels, frame[group_column]))
        split_method = f"grouped_by_{group_column}"
    else:
        train_index, test_index = train_test_split(
            np.arange(len(frame)),
            test_size=0.2,
            random_state=random_state,
            stratify=labels if labels.value_counts().min() >= 5 else None,
        )
        split_method = "random_rows_no_group_available"

    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=2,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    pipeline.fit(features.iloc[train_index], labels.iloc[train_index])
    predictions = pipeline.predict(features.iloc[test_index])
    report = classification_report(
        labels.iloc[test_index],
        predictions,
        labels=list(INTERACTION_LABELS),
        zero_division=0,
        output_dict=True,
    )

    bundle = interaction_bundle_path(output_dir, model_name)
    if bundle.exists():
        raise FileExistsError(f"Interaction model already exists: {bundle}")
    bundle.mkdir(parents=True)
    joblib.dump(pipeline, bundle / "model.joblib")
    manifest = {
        "model_type": "random_forest_session_interaction_classifier",
        "interaction_classes": list(pipeline.classes_),
        "supported_taxonomy": list(INTERACTION_LABELS),
        "features": list(INTERACTION_FEATURES),
        "target": target,
        "training_rows": len(train_index),
        "test_rows": len(test_index),
        "split_method": split_method,
        "random_state": random_state,
        "classification_report": report,
        "epistemic_note": (
            "Predictions estimate session interaction style; they do not identify a human "
            "with certainty."
        ),
    }
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def classify_interaction_sessions(
    sessions: pd.DataFrame,
    output_dir: str | Path,
    model_name: str,
    minimum_confidence: float = 0.60,
) -> pd.DataFrame:
    if not 0 <= minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be between 0 and 1")
    bundle = interaction_bundle_path(output_dir, model_name)
    model_path = bundle / "model.joblib"
    manifest_path = bundle / "manifest.json"
    if not model_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Incomplete interaction model bundle: {bundle}")
    pipeline = joblib.load(model_path)
    probabilities = pipeline.predict_proba(_training_frame(sessions))
    order = probabilities.argsort(axis=1)
    winners = order[:, -1]
    alternatives = order[:, -2]
    classes = pipeline.classes_
    confidence = probabilities.max(axis=1)
    predicted = np.asarray([classes[index] for index in winners], dtype=object)
    predicted[confidence <= minimum_confidence] = "UNKNOWN"

    result = sessions.copy()
    result["interaction"] = predicted
    result["confidence"] = confidence
    result["alternative"] = [classes[index] for index in alternatives]
    result["alternative_confidence"] = [
        probabilities[row, index] for row, index in enumerate(alternatives)
    ]
    result["interaction_evidence"] = result.apply(interaction_evidence, axis=1)
    return result


def interaction_evidence(row: pd.Series) -> str:
    return (
        f"{int(row['packets'])} packets/{float(row['duration']):.3f}s, "
        f"turns={int(row['request_response_turns'])}, bursts={int(row['burst_count'])}, "
        f"regularity={float(row['timing_regularity']):.3f}, "
        f"forward/reverse={int(row['forward_packets'])}/{int(row['reverse_packets'])}"
    )


def print_interaction_results(results: pd.DataFrame) -> None:
    printable = results.copy()
    for column in ("confidence", "alternative_confidence"):
        printable[column] = printable[column].map(lambda value: f"{value:.6f}")
    print(printable.to_string(index=False))
    counts = results["interaction"].value_counts().sort_index()
    summary = " ".join(f"{name}={count}" for name, count in counts.items())
    print(f"Summary: sessions={len(results)} {summary}")
    print(
        "Note: interaction labels are probabilistic session-style estimates, not proof of "
        "human identity or intent."
    )
