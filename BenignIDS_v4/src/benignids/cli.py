from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .experiment import run_experiment
from .pcap import build_flow_dataset
from .transformer_experiment import run_transformer_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BenignIDS v4 experiment runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="train, optimize, compare and stage models")
    run.add_argument("--config", default="configs/default.yaml")
    run.add_argument("--quick", action="store_true", help="use 10k rows and a small search budget")
    transformer = subparsers.add_parser(
        "train-transformer", help="masked-token pretraining followed by IDS fine-tuning"
    )
    transformer.add_argument("--config", default="configs/default.yaml")
    transformer.add_argument("--quick", action="store_true")
    pcap = subparsers.add_parser("prepare-pcap", help="convert labelled PCAP captures to flows")
    pcap.add_argument("--manifest", required=True)
    pcap.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "run":
        config = load_config(Path(args.config))
        table = run_experiment(config, quick=args.quick)
        print(table.to_string(index=False))
    elif args.command == "train-transformer":
        config = load_config(Path(args.config))
        print(run_transformer_experiment(config, quick=args.quick))
    elif args.command == "prepare-pcap":
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        build_flow_dataset(args.manifest).to_parquet(output, index=False)
        print(f"Wrote labelled flow records to {output}")


if __name__ == "__main__":
    main()
