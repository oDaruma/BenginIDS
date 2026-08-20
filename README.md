# BenginIDS

BenginIDS is a teaching-oriented intrusion-detection project for experimenting with labelled
network payload and flow data. The current implementation lives in [`BenignIDS_v4`](BenignIDS_v4/)
and provides reproducible tabular baselines, hyperparameter optimization, ensembles, a 1D-CNN,
and a compact traffic transformer.

The project is designed for research and learning. It is not a production IDS and it does not
ship raw captures, large datasets, or trained-model binaries.

## Features

- CSV and Parquet loaders for labelled payload/flow records.
- Labelled PCAP ingestion through a manifest containing `pcap_path`, `label`, and optional
  `attack_cat` fields.
- Leakage-aware train, validation, and test partitions.
- Logistic regression, random forest, LightGBM, 1D-CNN, soft-voting, and stacking comparisons.
- Grid, randomized, and Bayesian hyperparameter searches using average precision.
- A traffic transformer with masked-token pretraining followed by supervised behavior
  classification (`BENIGN`, `RECONNAISSANCE`, `BRUTE_FORCE`, `C2_BEACONING`, `EXFILTRATION`,
  `EXPLOITATION`, `DOS`, or `UNKNOWN`).
- Validation-set decision-threshold selection, PR-AUC, calibration, confusion matrices, and
  reproducible experiment manifests.
- Teaching notebooks connecting the implementation to core machine-learning concepts.

## Repository layout

```text
BenignIDS_v4/       current Python package, CLI, tests, notebooks, and documentation
BenignIDS_pre3/     legacy notebook pipeline retained for historical reference
Label_Trainer*.ipynb
                    earlier standalone notebook experiments
data_sheet.md       dataset provenance, schema, use, and risk notes
model_card.md       supported model families, evaluation, and limitations
```

## Installation

Python 3.12 is supported for the current implementation.

```bash
cd BenignIDS_v4
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
```

Install only the base package if transformer, PCAP, SHAP, and notebook extras are not needed:

```bash
python -m pip install -e .
```

## Provide training data

Large data files are excluded from Git. Set `data.path` in `BenignIDS_v4/configs/default.yaml`
to a CSV or Parquet dataset that follows the [data contract](BenignIDS_v4/docs/data_contract.md).
Named transformer training uses `behavior_label` by default. Missing values are conservatively
pseudo-labelled from observable fields, written to `training_labels.csv` with a trailing `*`, and
fall back to `UNKNOWN*` when there is insufficient evidence. Pseudo-labels are not ground truth.

Raw PCAP files require external labels. Create a manifest such as
[`BenignIDS_v4/examples/pcap_labels.csv`](BenignIDS_v4/examples/pcap_labels.csv), then prepare a
labelled flow dataset:

```bash
cd BenignIDS_v4
benignids prepare-pcap \
  --manifest examples/pcap_labels.csv \
  --output data/labelled_flows.parquet
```

Point `data.path` at the generated Parquet file before training. Capture-level labels are only
appropriate when every relevant flow in a capture has the same ground-truth class.

## Run experiments

Train a friendly-named behavior transformer from CSV, then use it to describe probable behavior
for flows in a PCAP and print results to stdout:

```bash
cd BenignIDS_v4
benignids --train data/training.csv --model unsw-transformer
benignids --run captures/example.pcap --model unsw-transformer
```

Each training run is stored separately under `artifacts/models/<model-name>/`. PCAP inference is
flow-level and reports a probable behavior, confidence, alternative hypothesis, and observable
flow evidence. These are behavior hypotheses—not proof of malicious intent. It requires the
optional PCAP dependencies installed by `.[pcap]` or `.[all]`.

The original experiment subcommands remain available:

```bash
cd BenignIDS_v4

# Fast transformer smoke experiment
benignids train-transformer --config configs/default.yaml --quick

# Fast tabular model and optimization comparison
benignids run --config configs/default.yaml --quick
```

Remove `--quick` for the configured experiment budgets. Full optimization and transformer
training can be computationally expensive.

## Evaluation policy

- Model parameters and representations are learned from the training partition.
- Hyperparameters are selected through cross-validation within training data.
- The validation partition selects the decision threshold `tau` (`τ`).
- The test partition is reserved for final evaluation.
- Average precision/PR-AUC is primary because accuracy can be misleading under class imbalance.

Results depend on the supplied dataset, split, seed, configuration, and software environment.
This repository makes no universal performance claim and does not treat legacy notebook metrics
as validated v4 results.

## Documentation

- [BenignIDS v4 guide](BenignIDS_v4/README.md)
- [Architecture](BenignIDS_v4/docs/architecture.md)
- [Data contract](BenignIDS_v4/docs/data_contract.md)
- [Course mapping](BenignIDS_v4/docs/course_mapping.md)
- [Dataset datasheet](data_sheet.md)
- [Model card](model_card.md)

## Safety and privacy

Use only traffic you are authorized to inspect. Packet captures and payload-derived fields may
contain sensitive information even when this repository does not include them. Review provenance,
licensing, retention, and privacy requirements before collecting, sharing, or training on traffic.
Do not deploy generated models as autonomous security controls without site-specific validation,
monitoring, and human oversight.

## Legacy material

`BenignIDS_pre3` and the root `Label_Trainer*.ipynb` files are retained to preserve the evolution
of the project. They may use older dependency versions, paths, schemas, and experimental claims.
Use `BenignIDS_v4` for current development.
