from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from .pcap import pcap_to_flows
from .tokenization import TrafficTokenizer
from .transformer import load_checkpoint, predict_probabilities, predict_scores

MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def model_bundle_path(output_dir: str | Path, model_name: str) -> Path:
    """Resolve a friendly model name without allowing path traversal."""
    if not MODEL_NAME.fullmatch(model_name):
        raise ValueError(
            "Model names must start with a letter or number and contain only letters, "
            "numbers, dots, underscores, or hyphens"
        )
    return Path(output_dir).resolve() / "models" / model_name


def load_threshold(path: str | Path) -> float:
    with Path(path).open(encoding="utf-8") as handle:
        metrics = json.load(handle)
    threshold = float(metrics.get("threshold", 0.5))
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Stored threshold must be between 0 and 1, got {threshold}")
    return threshold


def classify_pcap(
    pcap_path: str | Path, output_dir: str | Path, model_name: str
) -> tuple[pd.DataFrame, float]:
    """Classify bidirectional flows from a PCAP using a named transformer bundle."""
    pcap_path = Path(pcap_path)
    if pcap_path.suffix.lower() not in {".pcap", ".pcapng"}:
        raise ValueError("--run accepts only .pcap or .pcapng files")
    if not pcap_path.is_file():
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    bundle = model_bundle_path(output_dir, model_name)
    required = {
        "model checkpoint": bundle / "model.pt",
        "tokenizer": bundle / "tokenizer.json",
        "metrics": bundle / "metrics.json",
    }
    missing = [f"{label}: {path}" for label, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Incomplete model bundle:\n" + "\n".join(missing))

    flows = pcap_to_flows(pcap_path)
    if flows.empty:
        raise ValueError(f"No IPv4 or IPv6 flows found in PCAP: {pcap_path}")

    tokenizer = TrafficTokenizer.load(required["tokenizer"])
    model, model_config = load_checkpoint(required["model checkpoint"])
    if tokenizer.config.max_length != model_config.max_length:
        raise ValueError("Tokenizer and model max_length values do not match")
    input_ids, attention_mask = tokenizer.encode_frame(flows)
    result = flows.drop(columns=["payload"], errors="ignore").copy()
    manifest_path = bundle / "manifest.json"
    if manifest_path.is_file():
        with manifest_path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
    else:
        manifest = {}
    classes = manifest.get("behavior_classes")
    if classes and len(classes) == model_config.classes:
        probabilities = predict_probabilities(model, input_ids, attention_mask)
        order = probabilities.argsort(axis=1)
        winners = order[:, -1]
        alternatives = order[:, -2]
        result["behavior"] = [classes[index] for index in winners]
        result["confidence"] = probabilities.max(axis=1)
        result["alternative"] = [classes[index] for index in alternatives]
        result["alternative_confidence"] = [
            probabilities[row, index] for row, index in enumerate(alternatives)
        ]
        result["evidence"] = result.apply(_flow_evidence, axis=1)
        result["disposition"] = result["behavior"].map(
            lambda label: "BASELINE_MATCH" if label == "BENIGN" else "BEHAVIOR_HYPOTHESIS"
        )
        return result, 0.0

    scores = predict_scores(model, input_ids, attention_mask)
    threshold = load_threshold(required["metrics"])
    result["malicious_probability"] = scores
    result["threshold"] = threshold
    result["result"] = ["MALICIOUS" if score >= threshold else "BENIGN" for score in scores]
    return result, threshold


def _flow_evidence(row: pd.Series) -> str:
    """Describe only facts visible in the reconstructed flow."""
    return (
        f"{int(row['packets'])} packets, {int(row['total_len'])} bytes, "
        f"{float(row['duration']):.3f}s, {row['protocol']} ports "
        f"{int(row['port_a'])}<->{int(row['port_b'])}"
    )


def print_pcap_results(results: pd.DataFrame, threshold: float) -> None:
    """Write flow classifications and a compact summary to stdout."""
    printable = results.copy()
    if "behavior" in printable:
        printable["confidence"] = printable["confidence"].map(lambda value: f"{value:.6f}")
        printable["alternative_confidence"] = printable["alternative_confidence"].map(
            lambda value: f"{value:.6f}"
        )
        print(printable.to_string(index=False))
        counts = results["behavior"].value_counts().sort_index()
        summary = " ".join(f"{label}={count}" for label, count in counts.items())
        print(f"Summary: flows={len(results)} {summary}")
        print("Note: behavior labels are model hypotheses, not statements of human intent.")
        return
    printable["malicious_probability"] = printable["malicious_probability"].map(lambda value: f"{value:.6f}")
    printable["threshold"] = printable["threshold"].map(lambda value: f"{value:.6f}")
    print(printable.to_string(index=False))
    malicious = int((results["result"] == "MALICIOUS").sum())
    benign = int((results["result"] == "BENIGN").sum())
    print(
        f"Summary: flows={len(results)} benign={benign} malicious={malicious} "
        f"threshold={threshold:.6f}"
    )
