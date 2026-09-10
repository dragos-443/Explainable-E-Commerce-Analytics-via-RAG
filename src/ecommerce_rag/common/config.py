"""Load an environment profile without embedding storage addresses in code."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(environment: str | None = None) -> dict[str, Any]:
    """Return the selected YAML environment configuration."""
    selected_environment = environment or os.getenv("APP_ENV", "local")
    configured_directory = os.getenv("ECOMMERCE_RAG_CONFIG_DIR")
    config_directory = (
        Path(configured_directory)
        if configured_directory
        else PROJECT_ROOT / "config" / "environments"
    )
    config_path = config_directory / f"{selected_environment}.yml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration profile not found: {config_path}")

    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if config.get("environment") != selected_environment:
        raise ValueError(
            f"Profile {config_path} declares environment={config.get('environment')!r}"
        )
    return config
