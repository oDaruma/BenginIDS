from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.stats import loguniform, randint
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Integer, Real


@dataclass
class SearchResult:
    method: str
    estimator: object
    best_score: float
    best_params: dict
    elapsed_seconds: float
    trials: int


def search_spaces():
    grid = {
        "model__n_estimators": [200, 400],
        "model__learning_rate": [0.03, 0.08],
        "model__num_leaves": [15, 31, 63],
    }
    random = {
        "model__n_estimators": randint(150, 800),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__num_leaves": randint(15, 128),
        "model__max_depth": randint(3, 15),
        "model__subsample": np.linspace(0.6, 1.0, 9),
        "model__colsample_bytree": np.linspace(0.6, 1.0, 9),
    }
    bayesian = {
        "model__n_estimators": Integer(150, 800),
        "model__learning_rate": Real(0.01, 0.2, prior="log-uniform"),
        "model__num_leaves": Integer(15, 128),
        "model__max_depth": Integer(3, 15),
        "model__min_child_samples": Integer(5, 100),
        "model__subsample": Real(0.6, 1.0),
        "model__colsample_bytree": Real(0.6, 1.0),
        "model__reg_alpha": Real(1e-9, 10.0, prior="log-uniform"),
        "model__reg_lambda": Real(1e-9, 10.0, prior="log-uniform"),
    }
    return grid, random, bayesian


def optimize_lightgbm(
    pipeline,
    X,
    y,
    method: str,
    n_iter: int = 20,
    cv_folds: int = 5,
    scoring: str = "average_precision",
    n_jobs: int = -1,
    random_state: int = 42,
) -> SearchResult:
    """Compare equal-budget searches; Bayesian search is the primary reference method."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid, random, bayesian = search_spaces()
    if method == "grid":
        search = GridSearchCV(pipeline, grid, scoring=scoring, cv=cv, n_jobs=n_jobs, refit=True)
    elif method == "random":
        search = RandomizedSearchCV(
            pipeline,
            random,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=True,
        )
    elif method == "bayesian":
        search = BayesSearchCV(
            pipeline,
            bayesian,
            n_iter=n_iter,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
            random_state=random_state,
            refit=True,
            optimizer_kwargs={"base_estimator": "GP"},
        )
    else:
        raise ValueError("method must be one of: grid, random, bayesian")
    started = perf_counter()
    search.fit(X, y)
    elapsed = perf_counter() - started
    return SearchResult(
        method=method,
        estimator=search.best_estimator_,
        best_score=float(search.best_score_),
        best_params=search.best_params_,
        elapsed_seconds=elapsed,
        trials=len(search.cv_results_["params"]),
    )

