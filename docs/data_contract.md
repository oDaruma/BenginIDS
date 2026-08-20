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

`prepare-pcap` emits bidirectional five-tuple flow records with `src`, `dst`, `sport`, `dport`,
`protocol`, packet and byte counts, duration, payload bytes, the supplied labels, and `capture_id`.
PCAP support requires the optional `scapy` dependency; Parquet output requires a compatible pandas
Parquet engine in the runtime environment.

## Validation and provenance

- Supplied behavior labels are normalized to the documented taxonomy; unsupported values become
  `UNKNOWN` rather than silently creating arbitrary classes.
- Pseudo-labels are weak supervision and must not be reported as analyst-confirmed truth.
- Dataset source, license, checksum, transformations, collection period, and split policy should
  accompany every reported result.
- Payloads and addresses may be sensitive even when transformed into flow records.
