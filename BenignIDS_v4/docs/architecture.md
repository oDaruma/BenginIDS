# Architecture

```text
Payload_data_UNSW.csv                 raw .pcap + label manifest
          |                                      |
 ordered byte/metadata loader              five-tuple flow builder
          +----------------------+---------------+
                                 |
                     train / validation / test
                                 |
              +------------------+------------------+
              |                                     |
       TrafficTokenizer                    tabular preprocessing
              |                                     |
     masked-token pretraining                LightGBM + BO
              |                                     |
     supervised fine-tuning              grid/random/ensembles
              +------------------+------------------+
                                 |
                  PR-AUC, threshold, calibration,
                    runtime, robustness, manifest
```

The hashed field vocabulary is deterministic and is not fitted on validation/test data. Payload
bytes use dedicated token IDs, so byte order is preserved. `[CLS]` supplies the classification
representation. Masked-token pretraining uses only `X_train`; labels are excluded from tokens.

The v3.5.6 preprocessing fix remains a tested invariant. `ColumnTransformer` branches are appended
only when their resolved column lists are non-empty. Dropping `ttl`, `total_len`, and `t_delta`
therefore cannot produce an empty numerical pipeline failure.

