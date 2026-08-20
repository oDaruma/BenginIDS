# **Final Project — PCMLAI, Imperial College London**

> **Legacy documentation:** This charter records the goals and claims of the pre-v4 notebook
> project. It is retained for history and is not a current BenginIDS performance claim. See the
> current [model card](../../model_card.md) and [`BenignIDS_v4`](../../BenignIDS_v4/README.md).

This project develops a resilient Intrusion Detection System (IDS) designed to detect **unknown, evolving, and imbalanced cyber threats** in network traffic using the **UNSW-NB15 dataset**. The system distinguishes **benign traffic** from various attack categories under challenging conditions: **class imbalance** and **label noise**. 

The system uses **Bayesian Optimization (BO)** to fine-tune a **LightGBM** classifier, achieving ~0.95 **PR-AUC** for reliable detection of rare, high-risk attacks. A **1D Convolutional Neural Network (CNN)** was benchmarked but underperformed compared to tree-based methods on structured network data. The pipeline incorporates **interpretability (SHAP)** and **ensemble methods** (bagging, soft voting, stacking) to enhance accuracy and trust. Exper
iments are versioned in a **staging pipeline** with manifest tracking for reproducibility and operational deployment.

---

## Objectives
- Provide a baseline benchmark (LightGBM with Bayesian Optimisation).
- Compare against advanced methods (Hyperparameter Optimisation, CNN, Ensembles).
- Ensure reproducibility, artefact persistence, and resilience under drift.
- Deliver tooling for explainability (PCA, SHAP) and deployment readiness.

## Scope
- Dataset: UNSW-NB15 payload features.
- Preprocessing: TF-IDF, PCA, feature weighting.
- Models: LightGBM, CNN, Ensembles.
- Optimisation: Bayesian Optimisation, HPO (grid, random, Optuna/Hyperopt).
- Outputs: Models, predictions, metrics, diagnostics, final champion artefact.

## Deliverables
- Jupyter Notebook `BenignIDS_Trainer_v2.4.4.ipynb`
- Artefact directories: `staging/` and `out/`
- Documentation:
  - STYLE_GUIDE_v2.4.4.md
  - NOTEBOOK_CHECKLIST_v2.4.4.md
  - PROJECT_CHARTER_v2.4.4.md

## Constraints
- Must run top-to-bottom without redefinition of canonical variables (Section 0).
- Must persist intermediate artefacts for resume-safe execution.
- Must avoid global edits unless explicitly required.

## Success Criteria
- Accurate, resilient IDS benchmark on UNSW-NB15.
- Artefacts for CNN, Ensembles, Baseline, and HPO available for comparison (Sections 6–8).
- Champion model persisted and auto-wired for downstream use (Section 9).
- Deployment and drift monitoring hooks available (Section 10).

## Data Scheme
```
DATA_SCHEME = {
    "required_any_of": [["label", "label_str", "attack_cat"]],
    "target": {
        "name": TARGET_COL,               # typically "label"
        "type": "binary_int",
        "domain": [0, 1],
    },
    "target_alternatives": {
        "label_str": {
            "benign": 0, "normal": 0, "noattack": 0, "false": 0, "neg": 0,
            "malicious": 1, "attack": 1, "true": 1, "pos": 1,
        },
        "attack_cat": {
            "benign": 0,   # only benign counts as 0
            "*": 1         # everything else (dos, exploits, fuzzers, etc.)
        }
    },
    "payload": {
        "pattern": r"^payload_byte_(\d+)$",
        "expected_min": 16,
        "expected_max": 1498,
        "into": "payload",
    },
    "protocols": {
        "column": "proto",
        "allowed_values": ["tcp", "udp", "icmp"],   # others tolerated via OHE ignore
    },
    "special_optional": ["attack_cat", "label_str"],
    "exclude_from_features": [TARGET_COL, "payload", "attack_cat", "label_str"],
}

``` 
