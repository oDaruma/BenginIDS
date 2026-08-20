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
