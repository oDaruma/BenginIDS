from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def positive_scores(estimator, X) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        return np.asarray(estimator.predict_proba(X))[:, 1]
    if hasattr(estimator, "decision_function"):
        raw = np.asarray(estimator.decision_function(X))
        return 1 / (1 + np.exp(-raw))
    raise TypeError("Estimator must expose predict_proba or decision_function")


def choose_threshold(
    y_true,
    scores,
    objective: str = "f1",
    minimum_precision: float | None = None,
) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if not len(thresholds):
        return {"threshold": 0.5, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    eligible = np.ones(len(thresholds), dtype=bool)
    if minimum_precision is not None:
        eligible &= precision[:-1] >= minimum_precision
    if not eligible.any():
        index = int(np.argmax(precision[:-1]))
    elif objective == "recall":
        index = int(np.argmax(np.where(eligible, recall[:-1], -1)))
    else:
        index = int(np.argmax(np.where(eligible, f1, -1)))
    return {
        "threshold": float(thresholds[index]),
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
    }


def classification_metrics(y_true, scores, threshold: float = 0.5) -> dict:
    predicted = (np.asarray(scores) >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y_true, scores)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, predicted).tolist(),
    }


def save_comparison(rows: list[dict], path: str | Path) -> pd.DataFrame:
    table = pd.DataFrame(rows).sort_values("pr_auc", ascending=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def dump_json(value: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")

