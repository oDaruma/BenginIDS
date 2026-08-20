import numpy as np

from benignids.evaluation import choose_threshold, classification_metrics


def test_f1_threshold_is_selected_from_validation_scores():
    result = choose_threshold([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.9])
    assert 0.4 < result["threshold"] <= 0.6
    assert result["f1"] == 1.0


def test_metrics_include_imbalance_relevant_outputs():
    metrics = classification_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert metrics["pr_auc"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]

