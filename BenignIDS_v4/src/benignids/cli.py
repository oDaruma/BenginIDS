from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

from .config import load_config
from .experiment import run_experiment
from .inference import classify_pcap, model_bundle_path, print_pcap_results
from .pcap import build_flow_dataset
from .transformer_experiment import run_transformer_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BenignIDS v4 experiment runner")
    runtime = parser.add_mutually_exclusive_group()
    runtime.add_argument("--train", metavar="CSV", help="train a named behavior transformer from CSV")
    runtime.add_argument("--run", dest="run_pcap", metavar="PCAP", help="describe flow behavior hypotheses in a PCAP")
    parser.add_argument("--model", help="friendly model name stored under artifacts/models")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--quick", action="store_true", help="use a small training budget")

    subparsers = parser.add_subparsers(dest="command")
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
    parser = build_parser()
    args = parser.parse_args()
    if args.train:
        if not args.model:
            parser.error("--train requires --model")
        training_path = Path(args.train)
        if training_path.suffix.lower() != ".csv":
            parser.error("--train accepts only a .csv file")
        config = deepcopy(load_config(Path(args.config)))
        config["data"]["path"] = str(training_path)
        bundle = model_bundle_path(config["project"]["output_dir"], args.model)
        if (bundle / "model.pt").exists():
            parser.error(f"model '{args.model}' already exists at {bundle}; choose another name")
        metrics = run_transformer_experiment(config, quick=args.quick, model_name=args.model)
        print(f"Stored model bundle: {bundle}")
        print(metrics)
    elif args.run_pcap:
        if not args.model:
            parser.error("--run requires --model")
        config = load_config(Path(args.config))
        results, threshold = classify_pcap(
            args.run_pcap, config["project"]["output_dir"], args.model
        )
        print_pcap_results(results, threshold)
    elif args.command == "run":
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
    else:
        parser.error("choose --train, --run, or one of the legacy subcommands")


if __name__ == "__main__":
    main()
