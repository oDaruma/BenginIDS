import pandas as pd
import pytest

from benignids.data import inject_symmetric_label_noise, make_binary_target


def test_named_attack_labels_become_binary():
    labels = pd.Series(["benign", "Normal", "exploits", "dos"])
    result = make_binary_target(labels, ["benign", "normal"])
    assert result.tolist() == [0, 0, 1, 1]


def test_numeric_binary_labels_are_preserved():
    result = make_binary_target(pd.Series([0, 1, 0, 1]), ["benign"])
    assert result.tolist() == [0, 1, 0, 1]


def test_label_noise_is_reproducible_and_bounded():
    y = pd.Series([0, 1] * 50)
    noisy_a, positions_a = inject_symmetric_label_noise(y, 0.1, 7)
    noisy_b, positions_b = inject_symmetric_label_noise(y, 0.1, 7)
    assert positions_a.tolist() == positions_b.tolist()
    assert int((noisy_a != y).sum()) == 10
    assert noisy_a.equals(noisy_b)


def test_rejects_invalid_noise_rate():
    with pytest.raises(ValueError):
        inject_symmetric_label_noise(pd.Series([0, 1]), 0.5)

