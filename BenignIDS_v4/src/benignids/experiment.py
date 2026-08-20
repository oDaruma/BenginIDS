from __future__ import annotations

import platform
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from .data import inject_symmetric_label_noise, load_payload_dataset, split_features_target
from .evaluation import (
    choose_threshold,
    classification_metrics,
    dump_json,
    positive_scores,
    save_comparison,
)
from .models import build_ensembles, model_catalog
from .optimization import optimize_lightgbm
from .preprocessing import build_preprocessor
from .reporting import (
    plot_class_distribution,
    plot_confusion,
    plot_model_comparison,
    plot_optimization_efficiency,
    plot_pr_curves,
    print_table,
    tutorial_step,
    write_tutorial_report,
)


def _split(X, y, test_size, validation_size, random_state):
    X_development, X_test, y_development, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    relative_validation = validation_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_development,
        y_development,
        test_size=relative_validation,
        stratify=y_development,
        random_state=random_state,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_experiment(config: dict, quick: bool = False) -> pd.DataFrame:
    """Run leakage-safe comparison: tune on CV/validation and open the test set once."""
    seed = int(config["project"]["random_state"])
    tutorial = bool(config["project"].get("tutorial_mode", True))
    output = Path(config["project"]["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    tutorial_step(
        1,
        "Load and understand the traffic data",
        "We load an ordered payload-byte prefix and metadata. The label is converted to "
        "0 = benign/normal and 1 = attack/suspicious.",
        tutorial,
    )
    sample_rows = 10000 if quick else config["data"].get("sample_rows")
    frame = load_payload_dataset(
        config["data"]["path"], config["data"].get("payload_prefix_bytes", 256), sample_rows
    )
    X, y = split_features_target(
        frame, config["data"]["target"], config["data"]["benign_labels"]
    )
    class_table = (
        y.map({0: "Benign", 1: "Attack"})
        .value_counts()
        .rename_axis("class")
        .reset_index(name="records")
    )
    class_table["percentage"] = 100 * class_table["records"] / len(y)
    print_table(class_table, "Observed binary class distribution", tutorial)
    plot_class_distribution(y, figures / "01_class_distribution.png")
    tutorial_step(
        2,
        "Create leakage-safe data partitions",
        "Model parameters learn from X_train, threshold tau is selected on X_val, and X_test "
        "is opened only after all choices are fixed.",
        tutorial,
    )
    X_train, X_val, X_test, y_train, y_val, y_test = _split(
        X,
        y,
        config["split"]["test_size"],
        config["split"]["validation_size"],
        seed,
    )
    y_train_observed, noisy_positions = inject_symmetric_label_noise(
        y_train, config["split"].get("label_noise_rate", 0.0), seed
    )
    preprocessor, groups = build_preprocessor(
        X_train,
        config["data"].get("drop_features", []),
        config["data"].get("payload_column"),
        config["data"].get("payload_prefix_bytes", 64),
    )
    dump_json(
        {
            "numerical": groups.numerical,
            "categorical": groups.categorical,
            "payload": groups.payload,
            "dropped": groups.dropped,
        },
        output / "feature_schema.json",
    )
    threshold_config = config["threshold"]
    fitted = []
    rows = []
    curves = {}
    tutorial_step(
        3,
        "Train default comparison models",
        "Logistic regression supplies a linear reference, random forest demonstrates bagging, "
        "and LightGBM supplies the strong tabular reference.",
        tutorial,
    )
    for name in config["models"]["include"]:
        if tutorial:
            print(f"Training {name} ...", flush=True)
        model = model_catalog(seed)[name]
        pipeline = Pipeline([("preprocess", clone(preprocessor)), ("model", model)])
        started = perf_counter()
        pipeline.fit(X_train, y_train_observed)
        fit_seconds = perf_counter() - started
        y_val_score = positive_scores(pipeline, X_val)
        threshold = choose_threshold(
            y_val,
            y_val_score,
            objective=threshold_config.get("objective", "f1"),
            minimum_precision=threshold_config.get("minimum_precision"),
        )
        y_test_score = positive_scores(pipeline, X_test)
        metrics = classification_metrics(y_test, y_test_score, threshold["threshold"])
        metrics.update({"model": name, "method": "default", "fit_seconds": fit_seconds})
        rows.append(metrics)
        curves[f"{name}/default"] = (y_test, y_test_score)
        fitted.append((name, pipeline))
        joblib.dump(pipeline, output / f"{name}.joblib")

    # Compare optimization algorithms on the same LightGBM pipeline and CV objective.
    base_pipeline = Pipeline(
        [("preprocess", clone(preprocessor)), ("model", model_catalog(seed)["lightgbm"])]
    )
    tutorial_step(
        4,
        "Compare hyperparameter optimization methods",
        "Grid search enumerates a fixed design, random search samples broadly, and Bayesian "
        "optimization uses a probabilistic surrogate to balance exploration and exploitation. "
        "Every method is scored with stratified-CV average precision.",
        tutorial,
    )
    for method in config["optimization"]["methods"]:
        if tutorial:
            print(f"Optimizing LightGBM with {method} search ...", flush=True)
        result = optimize_lightgbm(
            clone(base_pipeline),
            X_train,
            y_train_observed,
            method,
            n_iter=5 if quick else config["optimization"]["n_iter"],
            cv_folds=3 if quick else config["split"]["cv_folds"],
            scoring=config["optimization"]["scoring"],
            n_jobs=config["optimization"]["n_jobs"],
            random_state=seed,
        )
        y_val_score = positive_scores(result.estimator, X_val)
        threshold = choose_threshold(
            y_val,
            y_val_score,
            objective=threshold_config.get("objective", "f1"),
            minimum_precision=threshold_config.get("minimum_precision"),
        )
        metrics = classification_metrics(
            y_test, positive_scores(result.estimator, X_test), threshold["threshold"]
        )
        metrics.update(
            {
                "model": "lightgbm",
                "method": method,
                "fit_seconds": result.elapsed_seconds,
                "cv_pr_auc": result.best_score,
                "trials": result.trials,
            }
        )
        rows.append(metrics)
        curves[f"lightgbm/{method}"] = (y_test, positive_scores(result.estimator, X_test))
        dump_json(result.best_params, output / f"best_params_{method}.json")
        joblib.dump(result.estimator, output / f"lightgbm_{method}.joblib")

    # Ensembles show variance reduction and meta-learning on common preprocessing pipelines.
    selected = [(name, clone(pipeline)) for name, pipeline in fitted if name != "logistic_regression"]
    if len(selected) >= 2:
        tutorial_step(
            5,
            "Train ensemble models",
            "Soft voting averages attack probabilities; stacking learns how to combine base "
            "model outputs using a logistic meta-learner.",
            tutorial,
        )
        X_train_val = pd.concat([X_train, X_val])
        y_train_val = pd.concat([y_train_observed, y_val])
        for name, ensemble in build_ensembles(selected, seed).items():
            started = perf_counter()
            ensemble.fit(X_train_val, y_train_val)
            metrics = classification_metrics(y_test, positive_scores(ensemble, X_test), 0.5)
            metrics.update(
                {"model": name, "method": "ensemble", "fit_seconds": perf_counter() - started}
            )
            rows.append(metrics)
            curves[name] = (y_test, positive_scores(ensemble, X_test))
            joblib.dump(ensemble, output / f"{name}.joblib")

    table = save_comparison(rows, output / "model_comparison.csv")
    champion = table.iloc[0].to_dict()
    tutorial_step(
        6,
        "Compare and interpret results",
        "PR-AUC ranks rare-event detection quality across thresholds. Precision, recall and F1 "
        "describe performance at tau selected from validation data.",
        tutorial,
    )
    printable = table[
        ["model", "method", "pr_auc", "precision", "recall", "f1", "fit_seconds"]
    ].copy()
    print_table(printable.round(4), "Ranked model comparison", tutorial)
    plot_model_comparison(table, figures / "02_model_comparison.png")
    plot_optimization_efficiency(table, figures / "03_optimization_efficiency.png")
    plot_pr_curves(curves, figures / "04_precision_recall_curves.png")
    plot_confusion(
        champion["confusion_matrix"],
        figures / "05_champion_confusion_matrix.png",
        f"Champion confusion matrix: {champion['model']} / {champion['method']}",
    )
    figure_names = [
        "figures/01_class_distribution.png",
        "figures/02_model_comparison.png",
        "figures/03_optimization_efficiency.png",
        "figures/04_precision_recall_curves.png",
        "figures/05_champion_confusion_matrix.png",
    ]
    write_tutorial_report(table, output / "tutorial_report.md", figure_names)
    dump_json(
        {
            "project": "BenignIDS_v4",
            "created_at": datetime.now(UTC).isoformat(),
            "dataset": str(config["data"]["path"]),
            "rows_loaded": len(frame),
            "splits": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
            "label_noise_flips": len(noisy_positions),
            "primary_metric": "PR-AUC / average precision",
            "champion": champion,
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "scikit_learn": sklearn.__version__,
            },
        },
        output / "manifest.json",
    )
    if tutorial:
        print(f"Tutorial report: {output / 'tutorial_report.md'}", flush=True)
    return table
