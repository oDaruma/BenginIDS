# Model card for BenginIDS

## Overview

BenginIDS is an experimental framework rather than one pretrained model. It trains and compares
binary baselines and a multiclass behavior transformer on user-supplied network records. No trained model is
distributed as a universal or production-ready champion.

## Supported models

- Logistic regression, random forest, and LightGBM tabular models.
- Optional 1D-CNN experiments.
- Soft-voting and stacking ensembles.
- A compact traffic transformer with masked-token pretraining and multiclass behavior fine-tuning.
- Grid, randomized, and Bayesian hyperparameter searches.

Optional SHAP support can assist analysis but does not establish causal explanations.

## Inputs and outputs

Inputs are CSV/Parquet payload or flow records, including tables prepared from PCAP. Named runtime
training defaults to `behavior_label`. The transformer produces behavior-class probabilities and
reports the highest-ranked class, a second-ranked alternative, confidence, and observable flow
evidence. This is a behavior hypothesis and not evidence of a person's actual intent.

Missing labels may receive deterministic weak-supervision labels. These carry `*` in the training
audit file and fall back to `UNKNOWN*`; they are not equivalent to ground truth and may encode the
heuristics' biases.

Artifacts may include fitted models, optimizer state, metrics, plots, tokenizer/checkpoint files,
and experiment manifests. Generated binaries are excluded from version control.

## Evaluation

- Training data fits preprocessing, representations, and model parameters.
- Cross-validation within training data selects hyperparameters.
- Validation data selects the classification threshold.
- Test data is intended for one final evaluation.
- Average precision/PR-AUC is primary; precision, recall, F1, ROC-AUC, Brier score, calibration,
  confusion matrices, and runtime provide additional context.

Values are meaningful only with their dataset provenance, split policy, seed, configuration, and
dependency versions. Historical legacy-notebook metrics are not validated v4 performance claims.

## Intended and excluded uses

Appropriate uses include education, reproducible IDS model comparison, controlled research on
imbalance and robustness, and prototyping with authorized labelled traffic.

Out-of-scope uses include autonomous blocking or disciplinary decisions, claims of zero-day
detection without prospective evidence, unauthorized surveillance, deployment to an unseen
network without local validation, and training on data with unknown provenance or usage rights.

## Limitations and safeguards

- Domain shift can substantially reduce performance.
- Capture-level labels can mislabel mixed traffic.
- Correlated flows can leak across record-level splits.
- Imbalance can make accuracy and ROC-AUC deceptively strong.
- Payload features can contain sensitive content or dataset-specific shortcuts.
- Probability scores may become miscalibrated after environmental change.
- Encrypted traffic, shared tools, and incomplete session context make intent non-identifiable.
- Pseudo-label errors can be learned and amplified by the classifier.

Validate on representative local data, prefer group/time-based splits, inspect false positives
and negatives, calibrate thresholds, monitor drift, maintain human review and rollback paths, and
treat every generated model as experimental until it passes production security/privacy review.
