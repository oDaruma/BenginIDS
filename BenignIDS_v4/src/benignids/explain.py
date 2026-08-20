from __future__ import annotations

from pathlib import Path

import numpy as np


def save_shap_summary(pipeline, X_sample, output_path: str | Path, max_rows: int = 1000):
    """Create a SHAP summary for a fitted tree pipeline without making SHAP mandatory."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError as exc:
        raise RuntimeError("Install explanation support with: pip install -e '.[explain]'") from exc
    transformed = pipeline.named_steps["preprocess"].transform(X_sample.iloc[:max_rows])
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    explainer = shap.TreeExplainer(pipeline.named_steps["model"])
    values = explainer.shap_values(transformed)
    if isinstance(values, list):
        values = values[-1]
    shap.summary_plot(np.asarray(values), transformed, feature_names=names, show=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close()

