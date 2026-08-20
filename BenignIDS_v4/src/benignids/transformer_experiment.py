from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .data import load_payload_dataset, make_binary_target
from .evaluation import choose_threshold, classification_metrics, dump_json
from .reporting import (
    plot_class_distribution,
    plot_confusion,
    plot_pr_curves,
    plot_training_history,
    print_table,
    tutorial_step,
    write_transformer_report,
)
from .tokenization import TokenizerConfig, TrafficTokenizer, mask_tokens
from .transformer import (
    TransformerConfig,
    build_traffic_transformer,
    fine_tune_classifier,
    predict_scores,
    pretrain_masked_language_model,
    save_checkpoint,
)


def _indices(y, test_size, validation_size, seed):
    all_indices = list(range(len(y)))
    development, test = train_test_split(
        all_indices, test_size=test_size, stratify=y, random_state=seed
    )
    relative_validation = validation_size / (1.0 - test_size)
    train, validation = train_test_split(
        development,
        test_size=relative_validation,
        stratify=y.iloc[development],
        random_state=seed,
    )
    return train, validation, test


def run_transformer_experiment(config: dict, quick: bool = False) -> dict:
    seed = int(config["project"]["random_state"])
    tutorial = bool(config["project"].get("tutorial_mode", True))
    output = Path(config["project"]["output_dir"]).resolve() / "transformer"
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    tutorial_step(
        1,
        "Load packet-derived records",
        "The loader keeps packet-byte order and maps benign/normal to 0 and every attack "
        "category to 1. Labels are excluded from transformer tokens.",
        tutorial,
    )
    frame = load_payload_dataset(
        config["data"]["path"],
        config["data"].get("payload_prefix_bytes", 256),
        5000 if quick else config["data"].get("sample_rows"),
    )
    target = config["data"]["target"]
    if target not in frame:
        raise KeyError(f"Target column '{target}' is absent from the traffic records")
    y = make_binary_target(frame[target], config["data"]["benign_labels"])
    X = frame.drop(columns=[target, "attack_cat"], errors="ignore")
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
        "Tokenize and split traffic",
        "Metadata becomes deterministic hashed tokens, payload bytes keep their numeric order, "
        "and the train/validation/test partitions remain isolated.",
        tutorial,
    )
    train, validation, test = _indices(
        y, config["split"]["test_size"], config["split"]["validation_size"], seed
    )
    settings = config["transformer"]
    tokenizer_config = TokenizerConfig(
        vocab_size=settings["vocab_size"],
        max_length=settings["max_length"],
        payload_prefix_bytes=config["data"].get("payload_prefix_bytes", 64),
    )
    tokenizer = TrafficTokenizer(tokenizer_config)
    input_ids, attention_mask = tokenizer.encode_frame(X)
    masked, token_labels = mask_tokens(
        input_ids[train],
        attention_mask[train],
        random_state=seed,
        vocab_size=settings["vocab_size"],
    )
    model_config = TransformerConfig(
        vocab_size=settings["vocab_size"],
        max_length=settings["max_length"],
        d_model=settings["d_model"],
        nhead=settings["nhead"],
        layers=settings["layers"],
        feedforward=settings["feedforward"],
        dropout=settings["dropout"],
    )
    model = build_traffic_transformer(model_config)
    tutorial_step(
        3,
        "Self-supervised masked-token pretraining",
        "Random traffic tokens are hidden and the transformer learns to reconstruct them. This "
        "uses X_train without attack labels.",
        tutorial,
    )
    pretraining = pretrain_masked_language_model(
        model,
        masked,
        attention_mask[train],
        token_labels,
        epochs=1 if quick else settings["pretrain_epochs"],
        batch_size=settings["batch_size"],
        seed=seed,
    )
    tutorial_step(
        4,
        "Supervised IDS fine-tuning",
        "The pretrained [CLS] representation is fine-tuned to estimate P(y=1|X). Class-weighted "
        "loss and validation PR-AUC address imbalance and overfitting.",
        tutorial,
    )
    fine_tuning = fine_tune_classifier(
        model,
        input_ids[train],
        attention_mask[train],
        y.iloc[train],
        input_ids[validation],
        attention_mask[validation],
        y.iloc[validation],
        epochs=2 if quick else settings["finetune_epochs"],
        batch_size=settings["batch_size"],
        patience=settings["patience"],
        seed=seed,
    )
    y_val_score = predict_scores(model, input_ids[validation], attention_mask[validation])
    threshold = choose_threshold(
        y.iloc[validation],
        y_val_score,
        config["threshold"].get("objective", "f1"),
        config["threshold"].get("minimum_precision"),
    )
    y_test_score = predict_scores(model, input_ids[test], attention_mask[test])
    metrics = classification_metrics(y.iloc[test], y_test_score, threshold["threshold"])
    metrics.update(
        {
            "model": "traffic_transformer",
            "training": "masked_language_model_then_binary_finetuning",
            "input_mode": "pcap_flows" if "capture_id" in X else "unsw_csv_records",
        }
    )
    tutorial_step(
        5,
        "Select tau and interpret held-out results",
        "The decision threshold tau is selected from validation scores, then the test set is "
        "opened once for final PR-AUC, precision, recall, F1 and calibration metrics.",
        tutorial,
    )
    print_table(
        pd.DataFrame(
            [
                {
                    key: metrics[key]
                    for key in ["pr_auc", "roc_auc", "precision", "recall", "f1", "brier", "threshold"]
                }
            ]
        ).round(4),
        "Traffic-transformer test metrics",
        tutorial,
    )
    plot_training_history(pretraining, fine_tuning, figures / "02_training_history.png")
    plot_pr_curves(
        {"traffic_transformer": (y.iloc[test], y_test_score)},
        figures / "03_precision_recall_curve.png",
    )
    plot_confusion(
        metrics["confusion_matrix"],
        figures / "04_confusion_matrix.png",
        "Traffic-transformer confusion matrix",
    )
    tokenizer.save(output / "tokenizer.json")
    save_checkpoint(model, model_config, output / "traffic_transformer.pt")
    pd.DataFrame(pretraining).to_csv(output / "pretraining_history.csv", index=False)
    pd.DataFrame(fine_tuning).to_csv(output / "finetuning_history.csv", index=False)
    dump_json(metrics, output / "metrics.json")
    dump_json(
        {
            "input_mode": metrics["input_mode"],
            "rows": len(frame),
            "splits": {"train": len(train), "validation": len(validation), "test": len(test)},
            "target": target,
            "label_semantics": {"0": "benign/normal", "1": "attack/suspicious"},
            "test_opened_once": True,
        },
        output / "manifest.json",
    )
    write_transformer_report(
        metrics,
        output / "tutorial_report.md",
        [
            "figures/01_class_distribution.png",
            "figures/02_training_history.png",
            "figures/03_precision_recall_curve.png",
            "figures/04_confusion_matrix.png",
        ],
    )
    if tutorial:
        print(f"Tutorial report: {output / 'tutorial_report.md'}", flush=True)
    return metrics
