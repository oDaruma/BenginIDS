import pandas as pd

from benignids.behavior import normalize_behavior_label, prepare_behavior_labels


def test_normalizes_supported_behavior_aliases():
    assert normalize_behavior_label("scan") == "RECONNAISSANCE"
    assert normalize_behavior_label("C2") == "C2_BEACONING"
    assert normalize_behavior_label("something unsupported") == "UNKNOWN"


def test_missing_labels_are_marked_and_audited():
    frame = pd.DataFrame(
        {
            "behavior_label": ["benign", None, None],
            "attack_cat": [None, "scan", None],
            "packets": [4, 8, 1200],
            "duration": [1.0, 1.0, 10.0],
        }
    )

    labels, audit = prepare_behavior_labels(frame, "behavior_label")

    assert labels.tolist() == ["BENIGN", "RECONNAISSANCE", "DOS"]
    assert audit["behavior_label"].tolist() == ["BENIGN", "RECONNAISSANCE*", "DOS*"]
    assert audit["pseudo_label"].tolist() == [False, True, True]


def test_absent_target_falls_back_to_unknown_star():
    labels, audit = prepare_behavior_labels(pd.DataFrame({"packets": [1]}), "behavior_label")
    assert labels.tolist() == ["UNKNOWN"]
    assert audit.loc[0, "behavior_label"] == "UNKNOWN*"
