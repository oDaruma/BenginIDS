# Data contract

## Archived payload CSV

Required:

- `label`: `benign`/`normal`, an attack category, or binary `0`/`1`.

Supported features:

- ordered `payload_byte_1 ... payload_byte_N` fields;
- metadata such as `ttl`, `total_len`, `protocol`, and `t_delta`;
- optional `attack_cat` when the binary label is separate.

The loader reads only the configured byte prefix to bound memory and sequence length.

## PCAP mode

PCAP bytes do not supply ground truth. A manifest must contain:

- `pcap_path`;
- `label` (`0` benign, `1` suspicious);
- optional `attack_cat`.

Capture-level labels are a simplification. If a file mixes benign and attack traffic, provide
flow-level ground truth or split it into appropriately labelled captures. Random packet-row splits
can leak near-duplicate flows; production experiments should split by capture/time/source host.

