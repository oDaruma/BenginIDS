# Model Card — Label_Trainer Intrusion Detector

## Model Description

- **Inputs.** Tabular feature vectors **X** derived from network flows/payload statistics loaded from `archive/Payload_data_UNSW.csv`. Each row corresponds to a session/flow with numeric features and a binary label `y`.
- **Outputs.**
  - **Score:** Predicted probability \( $\hat p = P(y=1 \mid X)$ \).
  - **Class:** Decision \( $\hat y = \mathbb{1}[\hat p \ge \tau]$ \) using a threshold **τ** selected to maximize **F1** on validation.
- **Model architectures.**
  - **Primary:** Gradient-boosted trees (**LightGBM** classifier) with class‑imbalance aware tuning.
  - **Alternative (optional):** CNN‑based classifier for derived representations; used as an experimental baseline.
- **Training setup.** Stratified train/validation/test split; **Bayesian optimization** (skopt) plus parallel grid search; artifacts checkpointed under `staging/` for exact resume.

## Performance


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


## Limitations

- **Domain shift.** Performance may drop on networks unlike the training source; re‑calibration of **τ** is recommended per environment.
- **Label noise.** If labels are derived from heuristics/signatures, minority-class noise can bias thresholding and AUPRC.
- **Interpretability.** Tree ensembles are less interpretable than single trees; use SHAP/feature importance for analyst transparency.

## Trade‑offs

- **Precision vs. coverage.** Increasing **τ** reduces false positives (higher precision) but may miss stealthy attacks (lower recall).
- **Latency vs. accuracy.** Heavier feature pipelines and CNN baselines may improve accuracy but increase compute.
- **Generalization vs. specificity.** Tuning on one subnet/application mix may overfit to that context; prefer cross‑site validation when possible.

