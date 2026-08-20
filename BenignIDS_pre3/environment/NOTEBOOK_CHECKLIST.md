# BenignIDS — Notebook Checklist

**This checklist ensures the notebook conforms to the Style Guide (v2.4.4) baseline.**  
**Baseline Sections Table**  
| Section | Subsections |  
|---------|-------------|  
| 0 — Setup & Preflight | 0.1 Sanity Config, Early Imports & Canonical Lock; 0.2 Load Data; 0.3 Target Audit; 0.4 Split & Preprocessing; 0.6 Save & Restore Staging Training Data; 0.9 Preflight Checks |  
| 1 — Data Audits | (Reserved slot; no current implementation) |  
| 2 — Feature Engineering | 2.1 Payload Sequence (TF-IDF) |  
| 3 — PCA + Feature Weights | - |  
| 4 — Bayesian Baseline + BO | 4.1 Baseline LGBM; 4.2 BO Setup; 4.3 BO Trials; 4.4 Persist Model + Preprocessor; 4.5 BO Diagnostics |  
| 5 — HPO | 5.1 Manual Grid Search; 5.2 Random Search; 5.3 Automated HPO; 5.4 Checkpointing; 5.5 HPO Diagnostics |  
| 6 — Ensembles | 6.1 Baseline Ensemble; 6.2 Bagging / Blending; 6.3 Stacking |  
| 7 — CNN | 7.1 CNN Training; 7.2 CNN Analysis; 7.3 CNN vs LGBM |  
| 8 — Final Report | 8.1 Analysis & SHAP Validation |  
| 9 — Champion Model Selection & Auto-Wire | 9.1 Candidate Pool; 9.2 Evaluation; 9.3 Persist; 9.6 Restore; 9.7 Auto-Wire Baseline; 9.8 Auto-Wire CNN/Ensemble |  
| 10 — Deployment & Reporting | - |

## Structure (must)
- [ ] Project introduction is the **first cell**.
- [ ] **Main sections** numbered as "Section x", one digit numbering.
- [ ] **Subsections** numbered as "Section x.x", two digit numbering.
- [ ] **Section 0** reserved for configuration & preprocessing only.
- [ ] Exactly **one** markdown cell and  **one** banner + print line at the top of each code cell.
- [ ] All the **Main sections** and their **Subsections** follows the **Baseline sections** table.
- [ ] All code cells should assigned with a section/subsection number
- [ ] All section/subsection number should be in sequence from top to bottom
- [ ] No duplicate section/subsection numbers

## Canonical Variables (must)
- [ ] Declared once in **Sanity Config**: `RANDOM_STATE`, `DATA_PATH`, `TARGET_COL`, `STAGE_ROOT`, `OUT_ROOT`.
- [ ] Declared once in **Derive & Refresh Preprocessor**: `feature_names`, `cat_cols`, `num_cols`, `preprocessor`.
- [ ] Use the following annotation of variable: X_train/y_train, X_val/y_val, X_test/y_test. don't use Xtr/ytr or X_tr/y_tr
- [ ] use only feature 'payload' in all training, never use 'payload_byte*'

## Data & Paths (must)
- [ ] Default `DATA_PATH = "archive/Payload_data_UNSW.csv"`.
- [ ] `STAGE_ROOT = "staging"` initialised once; only store staging data.
- [ ] `OUT_ROOT = "out"` initialised once; store outputs other than staging data.
- [ ] Data Load asserts file exists; folds `payload_byte_*` → `payload` (array-per-row).
- [ ] All the variable name in Section 0.1 are in upper case and the same in the whole notebook.

## Preprocessing (should)
- [ ] `feature_names` excludes `TARGET_COL` and special `payload` column.
- [ ] `cat_cols` and `num_cols` derived by dtype; `payload` excluded.
- [ ] `preprocessor = ColumnTransformer([num: StandardScaler, cat: OneHotEncoder])` with version-safe OHE.

## Validation (should)
- [ ] Schema Assertions pass before training.
- [ ] Train/Validation split uses `random_state=RANDOM_STATE` and `stratify=y`.
- [ ] Clean up redundancy code and keep the code clean.

## Reporting (may)
- [ ] Include SHAP insights; summarise top features and stability.
- [ ] Record PR-AUC and tuned F1 per fold; include confusion matrices.
- [ ] **Log drift metrics (e.g., KS statistic) in OUT_ROOT.**

## Version Control (must)
- [ ] **Commit hash logged in notebook metadata.**
- [ ] **CHANGELOG.md updated for non-trivial changes.**
