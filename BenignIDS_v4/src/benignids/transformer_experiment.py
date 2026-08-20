from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .behavior import prepare_behavior_labels
from .data import load_payload_dataset
from .evaluation import dump_json
from .reporting import (
    plot_class_distribution,
    plot_confusion,
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
    predict_probabilities,
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


def run_transformer_experiment(
    config: dict, quick: bool = False, model_name: str | None = None
) -> dict:
    seed = int(config["project"]["random_state"])
    tutorial = bool(config["project"].get("tutorial_mode", True))
    output_root = Path(config["project"]["output_dir"]).resolve()
    output = output_root / "models" / model_name if model_name else output_root / "transformer"
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    tutorial_step(
        1,
        "Load packet-derived records",
        "The loader keeps packet-byte order and prepares supplied or explicitly marked "
        "pseudo-labels. Labels are excluded from transformer tokens.",
        tutorial,
    )
    frame = load_payload_dataset(
        config["data"]["path"],
        config["data"].get("payload_prefix_bytes", 256),
        5000 if quick else config["data"].get("sample_rows"),
    )
    target = config["data"].get("behavior_target", "behavior_label")
    labels, label_audit = prepare_behavior_labels(frame, target)
    label_audit.to_csv(output / "training_labels.csv", index=False)
    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(labels), index=frame.index, name="behavior_class")
    if len(encoder.classes_) < 2:
        raise ValueError("Behavior training requires at least two classes after pseudo-labelling")
    minimum_class_rows = max(
        math.ceil(1 / config["split"]["test_size"]),
        math.ceil((1 - config["split"]["test_size"]) / config["split"]["validation_size"]),
    )
    if labels.value_counts().min() < minimum_class_rows:
        raise ValueError(
            f"Every behavior class needs at least {minimum_class_rows} records for the "
            "configured stratified splits"
        )
    X = frame.drop(columns=[target, "attack_cat"], errors="ignore")
    class_table = (
        labels
        .value_counts()
        .rename_axis("class")
        .reset_index(name="records")
    )
    class_table["percentage"] = 100 * class_table["records"] / len(y)
    print_table(class_table, "Observed behavior class distribution", tutorial)
    plot_class_distribution(labels, figures / "01_class_distribution.png")
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
        classes=len(encoder.classes_),
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
        "The pretrained [CLS] representation is fine-tuned to estimate a probability for each "
        "behavior class. Class-weighted loss reduces the effect of imbalance.",
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
    test_probabilities = predict_probabilities(model, input_ids[test], attention_mask[test])
    test_predictions = test_probabilities.argmax(axis=1)
    metrics = {
        "accuracy": float(accuracy_score(y.iloc[test], test_predictions)),
        "macro_f1": float(f1_score(y.iloc[test], test_predictions, average="macro")),
        "weighted_f1": float(f1_score(y.iloc[test], test_predictions, average="weighted")),
        "confusion_matrix": confusion_matrix(
            y.iloc[test], test_predictions, labels=range(len(encoder.classes_))
        ).tolist(),
        "classification_report": classification_report(
            y.iloc[test], test_predictions, labels=range(len(encoder.classes_)),
            target_names=encoder.classes_, output_dict=True,
            zero_division=0,
        ),
        "model": "traffic_transformer",
        "training": "masked_language_model_then_multiclass_behavior_finetuning",
        "input_mode": "pcap_flows" if "capture_id" in X else "csv_records",
    }
    tutorial_step(
        5,
        "Interpret held-out behavior results",
        "The test set is opened once for final multiclass accuracy, macro F1, weighted F1, and "
        "per-class metrics.",
        tutorial,
    )
    print_table(
        pd.DataFrame(
            [
                {
                    key: metrics[key]
                    for key in ["accuracy", "macro_f1", "weighted_f1"]
                }
            ]
        ).round(4),
        "Traffic-transformer test metrics",
        tutorial,
    )
    plot_training_history(pretraining, fine_tuning, figures / "02_training_history.png")
    plot_confusion(
        metrics["confusion_matrix"],
        figures / "03_confusion_matrix.png",
        "Traffic-transformer confusion matrix",
        encoder.classes_.tolist(),
    )
    tokenizer.save(output / "tokenizer.json")
    checkpoint_name = "model.pt" if model_name else "traffic_transformer.pt"
    save_checkpoint(model, model_config, output / checkpoint_name)
    pd.DataFrame(pretraining).to_csv(output / "pretraining_history.csv", index=False)
    pd.DataFrame(fine_tuning).to_csv(output / "finetuning_history.csv", index=False)
    dump_json(metrics, output / "metrics.json")
    dump_json(
        {
            "model_name": model_name,
            "input_mode": metrics["input_mode"],
            "rows": len(frame),
            "splits": {"train": len(train), "validation": len(validation), "test": len(test)},
            "target": target,
            "behavior_classes": encoder.classes_.tolist(),
            "pseudo_label_count": int(label_audit["pseudo_label"].sum()),
            "pseudo_labels_marked_with": "*",
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
            "figures/03_confusion_matrix.png",
        ],
    )
    if tutorial:
        print(f"Tutorial report: {output / 'tutorial_report.md'}", flush=True)
    return metrics
