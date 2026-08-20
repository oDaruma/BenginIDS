# Data contract

Training commands load the path configured at `data.path`. Supported prepared formats are `.csv`
and `.parquet`; raw PCAP must first be converted with `benignids prepare-pcap`.

## Archived payload CSV

Preferred:

- `behavior_label`: one of `BENIGN`, `RECONNAISSANCE`, `BRUTE_FORCE`, `C2_BEACONING`,
  `EXFILTRATION`, `EXPLOITATION`, `DOS`, or `UNKNOWN`.

Supported features:

- ordered `payload_byte_1 ... payload_byte_N` fields;
- metadata such as `ttl`, `total_len`, `protocol`, and `t_delta`;
- optional `attack_cat`, used as pseudo-label evidence when `behavior_label` is missing.

The target name is configurable with `data.behavior_target`. Missing labels are guessed only when
an implemented observable rule matches. `training_labels.csv` appends `*` to generated labels and
records the reason; otherwise the generated label is `UNKNOWN*`. At least two classes are
required. With the default 20/20/60 split, each class needs at least five records.

For CSV payload archives, the loader projects the configured ordered byte prefix to bound memory
and sequence length. In quick mode it samples from the projected table rather than taking only the
first rows, which helps avoid one-class samples from category-ordered archives.

## PCAP mode

PCAP bytes do not supply ground truth. A manifest must contain:

- `pcap_path`;
- `label` (`0` benign, `1` suspicious);
- optional `attack_cat`.

Capture-level labels are a simplification. If a file mixes benign and attack traffic, provide
flow-level ground truth or split it into appropriately labelled captures. Random packet-row splits
can leak near-duplicate flows; production experiments should split by capture/time/source host.

`prepare-pcap` emits time-bounded bidirectional five-tuple session records with endpoints, ports,
protocol, lifecycle state, directional packet and byte counts, timing/burst features, payload bytes,
the supplied labels, and `capture_id`.
PCAP support requires the optional `scapy` dependency; Parquet output requires a compatible pandas
Parquet engine in the runtime environment.

## Interaction-session model

The interaction classifier consumes one row per time-bounded bidirectional session. Training rows
require `interaction_label`; accepted values and aliases normalize to:

- `HUMAN_INTERACTIVE`
- `MACHINE_AUTOMATED`
- `MIXED`
- `UNKNOWN`

Required numeric features are `duration`, `packets`, `total_len`, `forward_packets`,
`reverse_packets`, `forward_bytes`, `reverse_bytes`, `packet_size_mean`, `packet_size_std`,
`interarrival_mean`, `interarrival_std`, `interarrival_cv`, `packets_per_second`,
`bytes_per_second`, `direction_changes`, `request_response_turns`, `burst_count`, `idle_fraction`,
`timing_regularity`, `syn_count`, `fin_count`, and `rst_count`.

Include `capture_id` whenever possible so evaluation can hold out complete captures. Labels should
come from controlled endpoint-side evidence such as originating process telemetry and user-input
records; those sources should create labels but should not be included as network-model features.
Do not label encrypted traffic as human-driven solely from application or port assumptions.

## Validation and provenance

- Supplied behavior labels are normalized to the documented taxonomy; unsupported values become
  `UNKNOWN` rather than silently creating arbitrary classes.
- Pseudo-labels are weak supervision and must not be reported as analyst-confirmed truth.
- Dataset source, license, checksum, transformations, collection period, and split policy should
  accompany every reported result.
- Payloads and addresses may be sensitive even when transformed into flow records.
