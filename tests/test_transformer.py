import numpy as np
import pytest

from benignids.tokenization import mask_tokens
from benignids.transformer import (
    TransformerConfig,
    build_traffic_transformer,
    fine_tune_classifier,
    predict_scores,
    pretrain_masked_language_model,
)


def test_transformer_pretraining_and_finetuning_smoke():
    pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    input_ids = rng.integers(4, 128, size=(12, 8), dtype=np.int64)
    input_ids[:, 0] = 1
    attention_mask = np.ones_like(input_ids)
    y = np.asarray([0, 1] * 6)
    masked, token_labels = mask_tokens(
        input_ids[:8], attention_mask[:8], mask_probability=1.0, vocab_size=128
    )
    config = TransformerConfig(
        vocab_size=128,
        max_length=8,
        d_model=16,
        nhead=4,
        layers=1,
        feedforward=32,
    )
    model = build_traffic_transformer(config)
    pretraining = pretrain_masked_language_model(
        model, masked, attention_mask[:8], token_labels, epochs=1, batch_size=4
    )
    fine_tuning = fine_tune_classifier(
        model,
        input_ids[:8],
        attention_mask[:8],
        y[:8],
        input_ids[8:],
        attention_mask[8:],
        y[8:],
        epochs=1,
        batch_size=4,
    )
    scores = predict_scores(model, input_ids[8:], attention_mask[8:])
    assert len(pretraining) == len(fine_tuning) == 1
    assert scores.shape == (4,)
    assert np.all((scores >= 0) & (scores <= 1))
