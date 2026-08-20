from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(path)
    data_path = Path(config["data"]["path"])
    if not data_path.is_absolute():
        config["data"]["path"] = str((path.parent / data_path).resolve())
    output_path = Path(config["project"]["output_dir"])
    if not output_path.is_absolute():
        config["project"]["output_dir"] = str((path.parent.parent / output_path).resolve())
    return config
