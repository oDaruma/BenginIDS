from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


def model_catalog(random_state: int = 42) -> dict[str, Any]:
    from lightgbm import LGBMClassifier

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            verbosity=-1,
        ),
    }


def build_ensembles(fitted_estimators: list[tuple[str, Any]], random_state: int = 42):
    """Return unfitted soft-voting and stacking estimators from model pipelines."""
    if len(fitted_estimators) < 2:
        raise ValueError("At least two estimators are required for an ensemble")
    voting = VotingClassifier(estimators=fitted_estimators, voting="soft", n_jobs=-1)
    stacking = StackingClassifier(
        estimators=fitted_estimators,
        final_estimator=LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state
        ),
        stack_method="predict_proba",
        cv=3,
        n_jobs=-1,
    )
    return {"soft_voting": voting, "stacking": stacking}

