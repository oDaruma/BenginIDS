# Legacy pre-v4 data scheme

> This schema is preserved for the archived notebook pipeline. Use the current
> [v4 data contract](../../BenignIDS_v4/docs/data_contract.md) for new work.

```yaml
DATA_SCHEME:
  required_any_of: [["label", "label_str", "attack_cat"]]
  target:
    name: TARGET_COL  # typically "label"
    type: binary_int
    domain: [0, 1]
  target_alternatives:
    label_str:
      benign: 0
      normal: 0
      noattack: 0
      false: 0
      neg: 0
      malicious: 1
      attack: 1
      true: 1
      pos: 1
    attack_cat:
      benign: 0
      "*": 1  # everything else (dos, exploits, fuzzers, etc.)
  payload:
    pattern: "^payload_byte_(\\d+)$"
    expected_min: 16
    expected_max: 1498
    into: payload
  protocols:
    column: proto
    allowed_values: ["tcp", "udp", "icmp"]  # others tolerated via OHE ignore
  special_optional: ["attack_cat", "label_str"]
  exclude_from_features: [TARGET_COL, "payload", "attack_cat", "label_str"]
  validation: "Assert schema in Section 0.3 using pandera or Great Expectations."
```
