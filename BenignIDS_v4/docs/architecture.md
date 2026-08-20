# Architecture

```text
labelled CSV/Parquet                  raw .pcap + label manifest
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
                  multiclass F1, per-class metrics,
                    runtime, provenance, manifest
```

The hashed field vocabulary is deterministic and is not fitted on validation/test data. Payload
bytes use dedicated token IDs, so byte order is preserved. `[CLS]` supplies the classification
representation. Masked-token pretraining uses only `X_train`; labels are excluded from tokens.

The PCAP path is a preparation stage: `prepare-pcap` aggregates labelled captures into a Parquet
flow table, and the normal experiment commands then load that table through `data.path`. Raw PCAP
files are not passed directly to a trainer.

The named runtime path uses the same components in two constrained modes. `--train <csv> --model
<name>` trains only from CSV and stores the checkpoint, tokenizer, metrics, manifest, and audited
training labels
under `artifacts/models/<name>/`. `--run <pcap> --model <name>` builds unlabelled flows in memory,
loads that bundle, and writes ranked behavior hypotheses plus observable evidence to stdout.

The behavior classifier predicts a finite taxonomy; it does not infer a person's mental state.
Missing training labels pass through a deterministic rule layer and are marked with `*` in the
audit artifact. `UNKNOWN*` is used when packet/flow evidence cannot support a narrower hypothesis.

The v3.5.6 preprocessing fix remains a tested invariant. `ColumnTransformer` branches are appended
only when their resolved column lists are non-empty. Dropping `ttl`, `total_len`, and `t_delta`
therefore cannot produce an empty numerical pipeline failure.

All partitions are created before model fitting. Training data fits preprocessing and model
parameters, validation data selects `tau`, and test data is reserved for final measurement.
Random row-level partitions do not prevent capture/host leakage, so higher-assurance studies
should introduce group- or time-aware splitting before treating results as operational evidence.
