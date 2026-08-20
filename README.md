# Label_Trainer — Intrusion Detection Model Pipeline

## Non‑technical explanation
This project trains a machine‑learning model to spot suspicious network activity (like lateral movement or zero‑day‑style behavior) from flow/payload statistics. We feed past examples of **benign** vs **attack** traffic to the model so it can learn patterns, then set a **decision threshold (τ)** that balances catching more true attacks with keeping false alarms low for SOC analysts.

## Data
- Original dateset obtained from `UNSW-NB15 and CIC-IDS2017 Labelled PCAP Data` (https://www.kaggle.com/datasets/yasiralifarrukh/unsw-and-cicids2017-labelled-pcap-data/code/data)
- Source file expected at: `archive/Payload_data_UNSW.csv` (tabular flow/payload features with binary labels).
- Features are aggregated statistics per flow/session; no raw packet contents or personal identifiers are included.
- Train/validation/test splits are **stratified** to preserve class balance. Intermediate splits and artifacts are stored under `staging/` for reproducibility.

## Model
- **Primary:** LightGBM classifier (gradient‑boosted trees) chosen for strong tabular performance, speed, and native handling of non‑linear interactions.
- **Secondary (optional):** CNN baseline for derived representations.
- **Decision rule:** predict attack when \( $\hat p \ge \tau$ \); **τ** is chosen to **maximize F1** on the validation set.

## Hyperparameter optimisation
- **Manual grid (parallel):** Exploratory search with `joblib.Parallel` (multi‑threaded).
- **Bayesian optimization:** `skopt` (BayesSearchCV / ask‑tell) with checkpointing and resume.
- Global concurrency via `N_THREADS`; models set `n_jobs=-1` when supported.

## Results



### Champion: LightGBM_BO_Enhanced (stage: `bo_enhanced`)
- **Dataset**: archive/Payload_data_UNSW.csv
- **AUPRC**: 0.9999 | **F1@τ**: 0.9997 | **τ**: 0.050
- **Seed**: 42 | **Updated**: 2025-08-29 07:41:58Z
- **Params**:

```json
{
  "num_leaves": 118,
  "max_depth": 12,
  "min_child_samples": 21,
  "subsample": 0.9012640129099659,
  "colsample_bytree": 0.8659965652378008,
  "learning_rate": 0.050602060132694665,
  "reg_lambda": 0.8973465609044181
}
```


## Profile
https://www.linkedin.com/in/maxchowhk
