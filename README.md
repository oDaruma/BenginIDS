# BenignIDS v4

This repository contains the current BenignIDS implementation. The obsolete pre-v4 notebook
pipelines have been removed so the package, tests, teaching notebooks, and documentation share
one canonical layout.

BenignIDS v4 is a teaching-oriented intrusion-detection project for learning from network
payload/flow records derived from PCAP. Its primary model is a compact traffic transformer with
two stages:

1. self-supervised masked-token pretraining on unlabelled traffic; and
2. supervised fine-tuning for a documented network-behavior taxonomy.

A Bayesian-optimized LightGBM pipeline is the principal non-transformer baseline. Grid search,
random search, logistic regression, random forests, a 1D-CNN, soft voting, and stacking show how
different training and optimization methods behave on the same splits.

## Evidence boundary

The legacy `Payload_data_UNSW.csv` is derived from packet captures but is not a raw PCAP stream.
It is not distributed in this repository. When supplied separately, the loader reads the first
configured payload-byte prefix and maps `benign` or `normal` to zero and other named categories to
one. PCAP ingestion is supported only with raw captures and a label manifest supplied by the user.

## Course alignment

The notebooks follow the notation used in the supplied Imperial College learning notebook:
`X`, `y`, `X_train`, `X_val`, `X_test`, `y_score`, and decision threshold `tau` (`τ`). They connect:

- probability and uncertainty to calibrated attack probabilities;
- train/validation/test sets and stratified cross-validation to leakage prevention;
- standardization and PCA to dimensionality reduction;
- bias–variance and class imbalance to model selection and PR-AUC;
- grid, random, and Bayesian search to black-box optimization;
- acquisition functions to exploration versus exploitation;
- neural networks to the 1D-CNN and transformer benchmarks;
- bagging, soft voting, and stacking to variance reduction and meta-learning;
- Monte Carlo masking/noise experiments to robustness under uncertainty.

See [docs/course_mapping.md](docs/course_mapping.md) for the detailed mapping.

## Installation

Use an isolated environment because the host Python installation may contain incompatible NumPy
binary packages.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

Python 3.12 is the supported runtime. The package metadata rejects older Python versions to keep
the scientific and PyTorch dependency stack consistent. On Intel macOS, the available compatible
stack is PyTorch 2.2.x with NumPy 1.26.x, so those versions are constrained in `pyproject.toml`.

## Run

Before running, change `data.path` in `configs/default.yaml` to an accessible CSV or Parquet file.
The example path documents the legacy layout and is not expected to exist in a fresh clone.

### Named transformer training and PCAP inference

Train from a labelled CSV and store an independent, friendly-named model bundle:

```bash
benignids --train data/training.csv --model unsw-transformer
```

Add `--quick` for a small training budget or `--config path/to/config.yaml` to select another
configuration. The behavior target is `behavior_label` by default and is configurable through
`data.behavior_target`. Supported classes are `BENIGN`, `RECONNAISSANCE`, `BRUTE_FORCE`,
`C2_BEACONING`, `EXFILTRATION`, `EXPLOITATION`, `DOS`, and `UNKNOWN`. Bundles are
stored under `artifacts/models/<model-name>/` and contain `model.pt`, `tokenizer.json`,
`metrics.json`, `manifest.json`, and `training_labels.csv`. Existing model names are not
overwritten.

If the target column or an individual value is absent, BenginIDS attempts a conservative
rule-based pseudo-label from observable fields. Generated values carry `*` in
`training_labels.csv`; when no defensible rule matches, the value is `UNKNOWN*`. The classifier
trains on the underlying class without the marker, while the audit file and manifest retain
pseudo-label provenance. Review or replace pseudo-labels before serious evaluation.

Classify an authorized PCAP with a stored model:

```bash
benignids --run captures/example.pcap --model unsw-transformer
```

The command aggregates packets into bidirectional flows and prints the most probable behavior,
confidence, second-ranked alternative, and observable evidence to stdout. This is flow-level
behavior-hypothesis generation, not an independent classification of every packet and not proof
of human intent. `.pcap` and `.pcapng` files are accepted.

Model names must start with a letter or number and may contain letters, numbers, dots, underscores,
and hyphens. Use the same `--config` at training and inference time when it changes `output_dir`.

### Human-interactive versus machine-automated sessions

PCAP ingestion also produces time-bounded bidirectional sessions with timing, direction,
request/response-turn, burst, idle, packet-size, throughput, and TCP lifecycle features. Train a
separate interaction model from labelled session rows:

```bash
benignids train-interaction \
  --csv data/labelled_sessions.csv \
  --model interaction-v1
```

The CSV must contain the session features documented in
[`docs/data_contract.md`](docs/data_contract.md) and an `interaction_label` with one of
`HUMAN_INTERACTIVE`, `MACHINE_AUTOMATED`, `MIXED`, or `UNKNOWN`. Use multiple captures, devices,
users, and time periods; when `capture_id` is present, evaluation holds out complete captures to
reduce leakage.

Classify sessions reconstructed from an authorized capture:

```bash
benignids classify-interaction \
  --pcap captures/example.pcap \
  --model interaction-v1 \
  --minimum-confidence 0.60 \
  --session-timeout 300
```

Predictions at or below the confidence threshold abstain to `UNKNOWN`. These labels estimate the
interaction style visible in a session; they do not establish human identity, intent, or whether
background software was triggered by a person.

### Existing experiment commands

Quick transformer demonstration:

```bash
benignids train-transformer --config configs/default.yaml --quick
```

Model and optimization comparison:

```bash
benignids run --config configs/default.yaml --quick
```

Remove `--quick` for the configured full experiment. The full Bayesian search can be expensive.

True PCAP ingestion requires a CSV manifest like `examples/pcap_labels.csv`:

```bash
benignids prepare-pcap \
  --manifest examples/pcap_labels.csv \
  --output data/labelled_flows.parquet
```

Then point `data.path` at the generated Parquet file. A PCAP does not normally contain its own
ground-truth attack label; the manifest supplies capture-level labels.

The existing experiment subcommands read the configured `data.path`. They remain available for
model-comparison and teaching workflows alongside the named runtime modes above.

## Leakage-safe evaluation

- `X_train` fits model parameters and self-supervised representations.
- Cross-validation inside `X_train` chooses hyperparameters.
- `X_val` chooses `τ` for maximum F1 or a minimum-precision regime.
- `X_test` is evaluated once after all choices are fixed.
- PR-AUC (average precision) is primary because accuracy and ROC-AUC can be misleading under
  severe imbalance.

Artifacts include model files, optimizer parameters, metrics, split counts, feature schemas,
training histories, and a manifest. Generated values are experimental results; this repository
does not hard-code or claim the earlier approximately 0.95 PR-AUC result.

See the [datasheet](docs/data_sheet.md) and [model card](docs/model_card.md) for
provenance requirements, supported uses, and operational limitations.

## Layout

```text
configs/                 experiment configuration
docs/                    course, architecture and data documentation
examples/                PCAP label-manifest example
notebooks/               executable teaching sequence
src/benignids/           reusable package and CLI
tests/                    behavior and regression tests
artifacts/                generated outputs (gitignored)
```

## Safety and operational scope

Live capture is deliberately not automatic: capturing network traffic may require elevated
privileges and authorization. Use only captures you are permitted to inspect. This project is a
research/training IDS, not a drop-in production control.
