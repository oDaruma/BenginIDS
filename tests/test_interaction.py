import numpy as np
import pandas as pd
import pytest

from benignids.interaction import (
    INTERACTION_FEATURES,
    classify_interaction_sessions,
    normalize_interaction_label,
    train_interaction_model,
)


def _sessions(rows=40):
    labels = ["HUMAN_INTERACTIVE", "MACHINE_AUTOMATED", "MIXED", "UNKNOWN"]
    records = []
    for index in range(rows):
        label_index = index % len(labels)
        record = {
            name: float(label_index * 10 + index % 5 + position / 100)
            for position, name in enumerate(INTERACTION_FEATURES)
        }
        record.update(
            {
                "interaction_label": labels[label_index],
                "capture_id": f"capture-{index % 8}",
                "endpoint_a": f"10.0.0.{index % 8}",
            }
        )
        records.append(record)
    return pd.DataFrame(records)


def test_normalizes_interaction_aliases_and_rejects_unsupported_labels():
    assert normalize_interaction_label("human") == "HUMAN_INTERACTIVE"
    assert normalize_interaction_label("m2m") == "MACHINE_AUTOMATED"
    with pytest.raises(ValueError, match="Unsupported"):
        normalize_interaction_label("probably a person")


def test_trains_persists_and_classifies_interaction_model(tmp_path):
    sessions = _sessions()
    manifest = train_interaction_model(sessions, tmp_path, "interaction-v1")

    assert set(manifest["supported_taxonomy"]) == {
        "HUMAN_INTERACTIVE",
        "MACHINE_AUTOMATED",
        "MIXED",
        "UNKNOWN",
    }
    assert (tmp_path / "interaction_models" / "interaction-v1" / "model.joblib").is_file()

    results = classify_interaction_sessions(sessions.iloc[:4], tmp_path, "interaction-v1", 0.0)
    assert set(results["interaction"]).issubset(set(manifest["interaction_classes"]))
    assert np.all((results["confidence"] >= 0) & (results["confidence"] <= 1))
    assert results["interaction_evidence"].str.contains("regularity=").all()


def test_low_confidence_abstains_to_unknown(tmp_path):
    sessions = _sessions()
    train_interaction_model(sessions, tmp_path, "interaction-v1")
    results = classify_interaction_sessions(sessions.iloc[:2], tmp_path, "interaction-v1", 1.0)
    assert results["interaction"].eq("UNKNOWN").all()


def test_training_requires_session_features(tmp_path):
    with pytest.raises(ValueError, match="missing interaction features"):
        train_interaction_model(
            pd.DataFrame({"interaction_label": ["human", "machine"]}), tmp_path, "bad"
        )
