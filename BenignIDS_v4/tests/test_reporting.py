from pathlib import Path

import pandas as pd

from benignids.reporting import (
    plot_class_distribution,
    plot_confusion,
    plot_model_comparison,
    plot_pr_curves,
    plot_training_history,
    write_transformer_report,
    write_tutorial_report,
)


def test_tutorial_reports_and_charts_are_generated(tmp_path):
    table = pd.DataFrame(
        [
            {
                "model": "lightgbm",
                "method": "bayesian",
                "pr_auc": 0.90,
                "precision": 0.85,
                "recall": 0.80,
                "f1": 0.825,
                "fit_seconds": 2.0,
                "trials": 5,
            },
            {
                "model": "random_forest",
                "method": "default",
                "pr_auc": 0.82,
                "precision": 0.79,
                "recall": 0.75,
                "f1": 0.77,
                "fit_seconds": 1.0,
                "trials": None,
            },
        ]
    )
    plot_class_distribution([0, 0, 1, 1, 1], tmp_path / "classes.png")
    plot_model_comparison(table, tmp_path / "models.png")
    plot_pr_curves({"demo": ([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])}, tmp_path / "pr.png")
    plot_confusion([[2, 0], [1, 2]], tmp_path / "confusion.png", "Demo")
    plot_training_history(
        [{"epoch": 1, "loss": 1.2}],
        [{"epoch": 1, "loss": 0.8, "val_pr_auc": 0.75}],
        tmp_path / "history.png",
    )
    write_tutorial_report(table, tmp_path / "tutorial.md", ["models.png"])
    metrics = {
        "accuracy": 0.9,
        "macro_f1": 0.8,
        "weighted_f1": 0.85,
        "input_mode": "csv_records",
    }
    write_transformer_report(metrics, tmp_path / "transformer.md", ["history.png"])
    expected = {
        "classes.png",
        "models.png",
        "pr.png",
        "confusion.png",
        "history.png",
        "tutorial.md",
        "transformer.md",
    }
    assert expected.issubset({path.name for path in Path(tmp_path).iterdir()})
