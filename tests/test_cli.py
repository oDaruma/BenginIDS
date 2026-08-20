from benignids.cli import build_parser


def test_parses_named_csv_training_mode():
    args = build_parser().parse_args(
        ["--train", "training.csv", "--model", "unsw-transformer", "--quick"]
    )
    assert args.train == "training.csv"
    assert args.model == "unsw-transformer"
    assert args.quick is True
    assert args.command is None


def test_parses_named_pcap_inference_mode():
    args = build_parser().parse_args(
        ["--run", "capture.pcap", "--model", "unsw-transformer"]
    )
    assert args.run_pcap == "capture.pcap"
    assert args.model == "unsw-transformer"
    assert args.command is None


def test_legacy_subcommands_remain_available():
    args = build_parser().parse_args(["train-transformer", "--quick"])
    assert args.command == "train-transformer"
    assert args.quick is True


def test_parses_interaction_training_and_inference_commands():
    training = build_parser().parse_args(
        ["train-interaction", "--csv", "sessions.csv", "--model", "interaction-v1"]
    )
    assert training.command == "train-interaction"
    assert training.target == "interaction_label"

    inference = build_parser().parse_args(
        ["classify-interaction", "--pcap", "capture.pcap", "--model", "interaction-v1"]
    )
    assert inference.command == "classify-interaction"
    assert inference.minimum_confidence == 0.60
