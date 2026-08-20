import pandas as pd
import pytest

from benignids.preprocessing import build_preprocessor, infer_feature_groups


def test_empty_numeric_branch_is_skipped_after_v356_drops():
    X = pd.DataFrame(
        {
            "ttl": [1, 2],
            "total_len": [10, 20],
            "t_delta": [0.1, 0.2],
            "protocol": ["tcp", "udp"],
        }
    )
    preprocessor, groups = build_preprocessor(
        X, drop_features=["ttl", "total_len", "t_delta"], payload_column=None
    )
    assert groups.numerical == ()
    assert [name for name, _, _ in preprocessor.transformers] == ["categorical"]
    assert preprocessor.fit_transform(X).shape == (2, 2)


def test_empty_categorical_branch_is_skipped():
    X = pd.DataFrame({"ttl": [1, 2], "total_len": [10, 20]})
    preprocessor, groups = build_preprocessor(X, payload_column=None)
    assert groups.categorical == ()
    assert [name for name, _, _ in preprocessor.transformers] == ["numeric"]


def test_all_features_removed_fails_clearly():
    X = pd.DataFrame({"ttl": [1, 2]})
    with pytest.raises(ValueError, match="No usable features"):
        build_preprocessor(X, drop_features=["ttl"], payload_column=None)


def test_missing_drop_columns_are_harmless():
    groups = infer_feature_groups(pd.DataFrame({"protocol": ["tcp"]}), ["ttl"], None)
    assert groups.dropped == ()

