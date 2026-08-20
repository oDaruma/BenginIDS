# Data contract

Training commands load the path configured at `data.path`. Supported prepared formats are `.csv`
and `.parquet`; raw PCAP must first be converted with `benignids prepare-pcap`.

## Archived payload CSV

Required:

- `label`: `benign`/`normal`, an attack category, or binary `0`/`1`.

Supported features:

- ordered `payload_byte_1 ... payload_byte_N` fields;
- metadata such as `ttl`, `total_len`, `protocol`, and `t_delta`;
- optional `attack_cat` when the binary label is separate.

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

- The configured target column must exist.
- Named benign labels are configured under `data.benign_labels`; every other named target becomes
  the attack class.
- Numeric targets are accepted directly only when all non-null values are binary `0`/`1`.
- Dataset source, license, checksum, transformations, collection period, and split policy should
  accompany every reported result.
- Payloads and addresses may be sensitive even when transformed into flow records.
