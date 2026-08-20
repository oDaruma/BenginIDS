from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def tutorial_step(number: int, title: str, explanation: str, enabled: bool = True) -> None:
    if not enabled:
        return
    rule = "=" * 78
    print(f"\n{rule}\nSTEP {number}: {title}\n{rule}\n{explanation}\n", flush=True)


def print_table(table: pd.DataFrame, title: str, enabled: bool = True) -> None:
    if enabled:
        print(f"\n{title}\n{'-' * len(title)}\n{table.to_string(index=False)}\n", flush=True)


def _prepare(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")
    return path


def plot_class_distribution(y, path: str | Path) -> None:
    path = _prepare(path)
    counts = pd.Series(y).map({0: "Benign", 1: "Attack"}).value_counts()
    figure, axis = plt.subplots(figsize=(7, 4))
    sns.barplot(x=counts.index, y=counts.values, hue=counts.index, legend=False, ax=axis)
    axis.set(title="Binary class distribution", xlabel="Traffic class", ylabel="Records")
    for index, value in enumerate(counts.values):
        axis.text(index, value, f"{value:,}", ha="center", va="bottom")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_model_comparison(table: pd.DataFrame, path: str | Path) -> None:
    path = _prepare(path)
    view = table.copy()
    view["candidate"] = view["model"] + " / " + view["method"]
    view = view.sort_values("pr_auc", ascending=True)
    figure, axis = plt.subplots(figsize=(10, max(5, len(view) * 0.55)))
    sns.barplot(data=view, x="pr_auc", y="candidate", hue="method", dodge=False, ax=axis)
    axis.set(title="Model comparison on the held-out test set", xlabel="PR-AUC", ylabel="")
    axis.set_xlim(0, 1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_optimization_efficiency(table: pd.DataFrame, path: str | Path) -> None:
    path = _prepare(path)
    view = table[table["method"].isin(["grid", "random", "bayesian"])].copy()
    if view.empty:
        return
    figure, axis = plt.subplots(figsize=(7, 5))
    sns.scatterplot(
        data=view, x="fit_seconds", y="pr_auc", hue="method", size="trials", sizes=(80, 240), ax=axis
    )
    axis.set(
        title="Optimization quality versus computation",
        xlabel="Search time (seconds)",
        ylabel="Held-out PR-AUC",
    )
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_pr_curves(curves: dict[str, tuple], path: str | Path) -> None:
    path = _prepare(path)
    figure, axis = plt.subplots(figsize=(8, 6))
    for name, (y_true, y_score) in curves.items():
        PrecisionRecallDisplay.from_predictions(y_true, y_score, name=name, ax=axis)
    axis.set_title("Precision–recall curves")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_confusion(matrix, path: str | Path, title: str) -> None:
    path = _prepare(path)
    figure, axis = plt.subplots(figsize=(5, 5))
    display = ConfusionMatrixDisplay(np.asarray(matrix), display_labels=["Benign", "Attack"])
    display.plot(ax=axis, cmap="Blues", colorbar=False)
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_training_history(pretraining: list[dict], fine_tuning: list[dict], path: str | Path) -> None:
    path = _prepare(path)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    pre = pd.DataFrame(pretraining)
    fine = pd.DataFrame(fine_tuning)
    axes[0].plot(pre["epoch"], pre["loss"], marker="o")
    axes[0].set(title="Masked-token pretraining", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(fine["epoch"], fine["val_pr_auc"], marker="o", color="tab:green")
    axes[1].set(title="Supervised fine-tuning", xlabel="Epoch", ylabel="Validation PR-AUC", ylim=(0, 1.02))
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_tutorial_report(table: pd.DataFrame, output: str | Path, figures: list[str]) -> None:
    output = Path(output)
    champion = table.sort_values("pr_auc", ascending=False).iloc[0]
    columns = ["model", "method", "pr_auc", "precision", "recall", "f1", "fit_seconds"]
    markdown_table = table[columns].sort_values("pr_auc", ascending=False).to_markdown(index=False)
    images = "\n".join(f"![{Path(figure).stem}]({figure})" for figure in figures)
    output.write_text(
        "# BenignIDS v4 tutorial results\n\n"
        "PR-AUC is the primary ranking metric because attacks and benign traffic are imbalanced. "
        "Threshold-dependent precision, recall and F1 explain the operational trade-off.\n\n"
        f"## Ranked results\n\n{markdown_table}\n\n"
        f"## Interpretation\n\nThe current champion is **{champion['model']} / "
        f"{champion['method']}** with PR-AUC **{champion['pr_auc']:.4f}**. "
        "This is evidence for this run and split, not a universal performance claim.\n\n"
        f"## Figures\n\n{images}\n",
        encoding="utf-8",
    )


def write_transformer_report(metrics: dict, output: str | Path, figures: list[str]) -> None:
    output = Path(output)
    images = "\n".join(f"![{Path(figure).stem}]({figure})" for figure in figures)
    output.write_text(
        "# Traffic-transformer tutorial results\n\n"
        "The encoder first learned by predicting masked traffic tokens, then learned the binary "
        "IDS target. The decision threshold was selected on validation data.\n\n"
        "## Test metrics\n\n"
        f"- PR-AUC: **{metrics['pr_auc']:.4f}**\n"
        f"- Precision: **{metrics['precision']:.4f}**\n"
        f"- Recall: **{metrics['recall']:.4f}**\n"
        f"- F1: **{metrics['f1']:.4f}**\n"
        f"- Threshold tau: **{metrics['threshold']:.4f}**\n"
        f"- Input mode: **{metrics['input_mode']}**\n\n"
        "These values describe this split and run; they are not a universal deployment claim.\n\n"
        f"## Figures\n\n{images}\n",
        encoding="utf-8",
    )
