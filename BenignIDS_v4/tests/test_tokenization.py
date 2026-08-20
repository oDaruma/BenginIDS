import numpy as np
import pandas as pd

from benignids.tokenization import CLS, PAD, TokenizerConfig, TrafficTokenizer, mask_tokens


def test_payload_bytes_are_encoded_in_numeric_order():
    tokenizer = TrafficTokenizer(TokenizerConfig(vocab_size=1024, max_length=8))
    ids, mask = tokenizer.encode_record(
        {"payload_byte_10": 10, "payload_byte_2": 2, "payload_byte_1": 1, "label": "attack"}
    )
    assert ids[:4] == [CLS, 5, 6, 14]
    assert mask[:4] == [1, 1, 1, 1]
    assert all(value == PAD for value in ids[4:])


def test_label_does_not_change_tokens():
    tokenizer = TrafficTokenizer(TokenizerConfig(vocab_size=1024, max_length=8))
    benign, _ = tokenizer.encode_record({"payload_byte_1": 42, "label": "benign"})
    attack, _ = tokenizer.encode_record({"payload_byte_1": 42, "label": "exploits"})
    assert benign == attack


def test_masking_never_masks_special_or_padding_tokens():
    ids = np.array([[CLS, 10, 11, PAD, PAD]])
    attention = np.array([[1, 1, 1, 0, 0]])
    masked, labels = mask_tokens(ids, attention, mask_probability=1.0, random_state=1, vocab_size=100)
    assert masked[0, 0] == CLS
    assert labels[0, 0] == -100
    assert np.all(labels[0, 3:] == -100)


def test_frame_encoding_has_fixed_shape():
    tokenizer = TrafficTokenizer(TokenizerConfig(vocab_size=1024, max_length=6))
    X = pd.DataFrame({"payload_byte_1": [1, 2], "protocol": ["tcp", "udp"]})
    ids, attention = tokenizer.encode_frame(X)
    assert ids.shape == attention.shape == (2, 6)
