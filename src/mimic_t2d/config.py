"""Configuration loading and stable project paths."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a study configuration and resolve project-relative directories."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    root = config_path.parent.parent
    config["project_root"] = root
    config["data_path"] = root / config["data_dir"]
    config["output_path"] = root / config["output_dir"]
    return config
