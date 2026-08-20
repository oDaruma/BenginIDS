# Datasheet: Label_Trainer IDS Payload Dataset

> This datasheet documents the dataset used by the **Label_Trainer** project to train and evaluate intrusion detection models for adversarial behaviors (e.g., lateral movement, abnormal activity, zero‑day‑like traffic).

## Motivation

- **Purpose.** Support supervised learning experiments for network intrusion detection and alert triage in a SOC setting. The focus is on **high precision under class imbalance** and on selecting a **decision threshold (τ)** that balances coverage (TPR/Recall) versus false alert rate (TNR/Specificity).
- **Creators / funding.** Prepared for the Imperial College PCMLAI coursework as part of the student's defensive cybersecurity track. Curated and formatted by the student for research and education.
- **Intended users.** Researchers and students running the Label_Trainer notebook; SOC analysts interested in reproducible IDS model training and thresholding.

## Composition

- **Instances.** Tabular rows where each instance summarizes a network flow/session or payload-derived feature vector. Columns include numerical/statistical aggregations and a **binary label** (`y ∈ {0,1}`) indicating attack vs. benign.
- **Cardinality.** Depends on the CSV version checked into the project. Typical size: tens to hundreds of thousands of rows.
- **Missing data.** Some columns may contain missing values; preprocessing handles imputation or column drops (see below).
- **Potentially sensitive fields.** No PII is expected. Payload-derived features are **aggregated** statistics (not raw content). Do **not** add raw packet payloads or user identifiers to this dataset without a privacy review.

## Collection process

- **Acquisition.** Loaded from `archive/Payload_data_UNSW.csv` (a prepared CSV of payload/flow features). The file is included in the project’s expected folder layout.
- **Sampling.** Dataset may represent a filtered/stratified subset to emphasize minority attack classes and realistic imbalance. Any additional sampling is documented in the training notebook’s staging manifests.
- **Time frame.** Not time-series aligned in the current release; rows may originate from multiple capture days. If you construct temporal splits, document the period at split time.

## Preprocessing / cleaning / labeling

- **Transforms.** 
  - Column type casting; categorical handling when present.
  - Standardization / scaling where required by model family.
  - Train/validation/test split with **stratification on `y`**.
  - Threshold selection by maximizing **F1** on the validation set to obtain **τ**.
- **Label source.** Binary labels loaded from `label` or `label_str` (converted to `y ∈ {0,1}`).
- **Raw retention.** The original CSV in `archive/` is preserved. Intermediate artifacts and folds are stored in `staging/` to enable exact resume/reproduction.

## Uses

- **Primary tasks.** 
  - Binary classification for intrusion/attack detection.
  - Threshold calibration for SOC deployment.
- **Other plausible tasks.** 
  - Cost-sensitive learning; PU learning; drift checks; feature attribution analyses.
- **Use considerations & risks.**
  - **Imbalance:** Metrics such as **AUPRC** are emphasized; ROC‑AUC alone can be misleading.
  - **Shift risk:** Models may degrade on new networks; validate on site-specific samples before deployment.
  - **Fairness & harm:** Although no protected attributes are included, false positives may incur operational cost; tune **τ** for acceptable alert rates and provide analyst‑in‑the‑loop review.
- **Out‑of‑scope uses.**
  - Identification of individuals; content inspection beyond aggregate statistics; any application involving PII without a separate privacy assessment.

## Distribution

- **Sharing.** Provided for coursework and internal research. Do not redistribute raw captures or any proprietary traffic.
- **Licensing / ToU.** Treat as educational/research use only unless you have rights to redistribute the underlying CSV.

## Maintenance

- **Maintainer.** Student owner of the Label_Trainer project.
- **Contact.** See the project README for contact details (optional).
- **Versioning.** Changes are recorded via notebook `PROJECT_VERSION` and `staging/manifest.json` files per stage.
