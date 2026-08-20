import pandas as pd
import pytest

from benignids.inference import load_threshold, model_bundle_path, print_pcap_results


@pytest.mark.parametrize("name", ["model/escape", "../escape", "spaces are invalid", ""])
def test_model_name_rejects_path_traversal_and_invalid_characters(tmp_path, name):
    with pytest.raises(ValueError):
        model_bundle_path(tmp_path, name)


def test_model_name_resolves_inside_output_directory(tmp_path):
    assert model_bundle_path(tmp_path, "unsw-v1") == tmp_path.resolve() / "models" / "unsw-v1"


def test_load_threshold_validates_range(tmp_path):
    metrics = tmp_path / "metrics.json"
    metrics.write_text('{"threshold": 1.2}', encoding="utf-8")
    with pytest.raises(ValueError, match="between 0 and 1"):
        load_threshold(metrics)


def test_print_pcap_results_writes_rows_and_summary(capsys):
    results = pd.DataFrame(
        {
            "endpoint_a": ["10.0.0.1", "10.0.0.2"],
            "endpoint_b": ["10.0.0.9", "10.0.0.8"],
            "malicious_probability": [0.1, 0.9],
            "threshold": [0.5, 0.5],
            "result": ["BENIGN", "MALICIOUS"],
        }
    )

    print_pcap_results(results, 0.5)

    output = capsys.readouterr().out
    assert "BENIGN" in output
    assert "MALICIOUS" in output
    assert "Summary: flows=2 benign=1 malicious=1 threshold=0.500000" in output
