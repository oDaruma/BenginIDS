# Notebook Section Structure — Updated v2.4.5

> **Legacy documentation:** These conventions apply only to `BenignIDS_pre3` notebooks. The
> current Python package and teaching notebooks are documented in
> [`BenignIDS_v4`](../../BenignIDS_v4/README.md).

The notebook must follow this ordered section structure.  
Each section has a markdown header (`## Section X — ...`) and a bannered code cell.

---
# BenignIDS Style Guide (v2.4.5)

## Baseline Section Structure

### Section 0 — Setup & Preflight
- 0.1 Sanity Config, Early Imports & Canonical Lock
- 0.2 Load Data
- 0.3 Target Audit
- 0.4 Split & Preprocessing
- 0.6 Save & Restore Staging Training Data
- 0.9 Preflight (Artefact Checks & Lock Arm)

### Section 1 — [Reserved for data audits/feature reports]

### Section 2 — Feature Engineering (Preprocessing)
- 2.1 Payload Sequence (TF-IDF)

### Section 3 — PCA + Feature Weights

### Section 4 — Bayesian Baseline and BO
- 4.1 Baseline Model (LightGBM with staged preprocessor)
- 4.2 Bayesian Optimisation Setup
- 4.3 BO Trials & Evaluation
- 4.4 Persist Best Model + Preprocessor
- 4.5 BO Diagnostics & Visualisations

### Section 5 — Hyperparameter Optimisation (HPO)
- 5.1 Manual Grid Search
- 5.2 Random Search
- 5.3 Automated HPO (Optuna / Hyperopt)
- 5.4 Checkpointing & Resumability
- 5.5 HPO Diagnostics & Visualisations

### Section 6 — Ensembles
- 6.1 Baseline Ensemble
- 6.2 Bagging / Blending
- 6.3 Stacking

### Section 7 — CNN
- 7.1 CNN Training & Validation
- 7.2 CNN Performance Analysis
- 7.3 CNN vs LightGBM Comparison

### Section 8 — Final Report
- 8.1 Analysis & SHAP Validation

### Section 9 — Champion Model Selection & Auto-Wire
- 9.1 Champion Candidate Pool
- 9.2 Champion Evaluation
- 9.3 Champion Persist
- 9.6 Champion Restore
- 9.7 Auto-Wire Baseline
- 9.8 Auto-Wire CNN/Ensemble

### Section 10 — Deployment, Drift, Reporting

## Notes on Structure
- Each section/subsection must start with a markdown cell explaining **what** and **why**.
- Each code cell must start with an ASCII banner and status print:  
  `# Section X — ...` and `print(">>> Section X: ...")`
- All section/subsection must ~~un sequence~~ **in sequence** and no duplicate from top to bottom.
- No placeholder sections.
- **When patching/updating notebooks, show full markdown + code cell pairs in Jupyter format.**
- Main sections **must** ~~to follows~~ **follow**; subsections **may** be ~~change if necessary~~ **changed if necessary** and approved.

## Notes on Reproducibility
- Canonical variables are only defined in Section 0.1. All other sections use them read-only.
- Preflight checks in Section 0.9 must be updated when new artefacts are introduced.
- Default dataset path: `archive/Payload_data_UNSW.csv`.
- Initialise `STAGE_ROOT` once in Sanity Config and reuse everywhere.
- Keep a short CHANGELOG when making non-trivial fixes.
- **Consider DVC or MLflow for advanced artefact tracking if scaling beyond notebooks.**

## Naming (must)
- Follow **Imperial College PCMLAI** conventions for primary variables.
- Splits: `X_train`, `y_train`, `X_val`, `y_val`, `X_test`, `y_test`. Don't use 'Xtr'/'ytr' or 'X_tr'/'y_tr'
- Schema: `feature_names`, `cat_cols`, `num_cols`.
- Models & optimisation: `baseline_model`, `bayes`.
- Config: `RANDOM_STATE`, `DATA_PATH`, `STAGE_ROOT`, `OUT_ROOT`.
- Otherwise, follow **PEP 8**.

## Section Structure (must)
- Keep project-specific **introduction** (from `PROJECT_CHARTER.md`) always at the cell 0.
- For **main sections** use `## Section N — Title` in markdown.
- For **subsections** use `## Section N.M — Title`.
- **Markdown cells** before each code cell must follow:

  ```markdown
  ## Section N(.M) — Title

  **What this does**
  - bullet 1
  - bullet 2

  **Why**
  - rationale 1
  - rationale 2
  ```
## Every code cell must begin with an ASCII banner including bullets and a status print:
  ```python
  # ======================================================
  # Section N(.M) — Title
  #   • bullet 1
  #   • bullet 2
  # ======================================================
  print(">>> Section N(.M): status")
```

## Wrap risky operations (e.g., file I/O, model persistence) in try-except with logging to OUT_ROOT.
