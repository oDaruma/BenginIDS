# IDS Final Report

This directory collects reports and trained model artefacts from the IDS training pipeline.

## Reports
- Evaluation assets (e.g., PR curves, confusion matrices), exported as JSON/CSV/PNG/PDF/HTML.

## Models
- Serialized estimators for deployment or inspection (e.g., LightGBM/XGBoost `.pkl`/`.joblib`, Keras `.keras`, ONNX `.onnx`).

## Metrics
- `metrics/best_tree.json`: best tree baseline/ensemble validation accuracy (from 04/06).
- `metrics/cnn.json` (optional): CNN validation accuracy (from 07).
- `manifest.json`: machine-readable index of artefacts + headline metrics.

