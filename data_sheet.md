# Datasheet for BenginIDS training data

## Scope and purpose

BenginIDS does not distribute a canonical training dataset. Users supply a labelled CSV or
Parquet table, or generate a labelled flow table from authorized PCAP files and an external label
manifest. Dataset-specific facts such as row count, capture dates, class balance, licensing, and
collection environment must be recorded by the person running the experiment.

The interface supports educational and research experiments in binary network intrusion
detection. It is not intended for identifying people, inspecting traffic without authorization,
or making fully autonomous enforcement decisions.

## Supported sources

Labelled CSV/Parquet records require a target column (`label` by default) and may contain ordered
`payload_byte_1 ... payload_byte_N` fields, numerical or categorical flow metadata, and an
optional `attack_cat`. Named `benign` and `normal` labels map to `0`; other named categories map
to `1`. A target already encoded entirely as `0` and `1` is preserved. See the
[v4 data contract](BenignIDS_v4/docs/data_contract.md).

PCAP files do not normally contain ground truth. The PCAP workflow requires a CSV manifest with
`pcap_path`, `label`, and optional `attack_cat`. It aggregates packets into bidirectional
five-tuple flow records with packet counts, byte counts, duration, payload bytes, protocol data,
and a capture identifier. Capture-level labels are unsafe for mixed-class captures unless
flow-level ground truth is supplied separately.

## Provenance and licensing

Legacy notebooks reference prepared UNSW-NB15/CIC-IDS2017-derived payload tables. Those large
files are excluded from this repository. Users must verify the source, license, redistribution
rights, transformations, and checksum of any supplied dataset.

For each experiment, record the source or collection owner, license, acquisition date, checksum,
capture environment and time range, filtering and labeling steps, row/capture counts, and class
distribution.

## Splitting and leakage

The experiments use stratified train, validation, and test partitions. Random record-level
splitting can still leak correlated packets or flows. For serious evaluation, group by capture,
time window, source host, or collection scenario and document the policy. Fit preprocessing and
models on training data, choose thresholds on validation data, and reserve test data for final
evaluation.

## Sensitive information

Raw captures and payload bytes can expose credentials, identifiers, communications, addresses,
or proprietary protocols. Do not assume packet-derived data is anonymous. Minimize retained
fields, control access, define retention periods, and obtain privacy/security review before
processing real organizational traffic.

## Known limitations

- Public benchmark traffic may not represent a target network.
- Labels can be incomplete, heuristic, or capture-level rather than flow-level.
- Class ratios and attack families change over time.
- Random sampling may hide temporal or host-level dependencies.
- A binary target collapses differences among attack categories.

Dataset files and generated artifacts are intentionally ignored by Git. Experiment manifests
should identify the data path, record count, split sizes, label semantics, and random seed without
embedding sensitive records.
