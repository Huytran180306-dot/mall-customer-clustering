"""
Central configuration loader for Mall Customer Clustering.
All project modules should read configuration through this file.
"""

from pathlib import Path
from typing import Any

import yaml


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

CONFIG_PATH = ROOT_DIR / "config.yaml"


# ============================================================
# 2. LOAD YAML CONFIGURATION
# ============================================================

def load_config(config_path: Path = CONFIG_PATH) -> dict:
    """Load and validate the YAML configuration file."""

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            "Invalid configuration: YAML root must be a dictionary."
        )

    return config


# Load configuration once when the module is imported.
CONFIG = load_config()


# ============================================================
# 3. GET CONFIGURATION VALUES
# ============================================================

def get_config(*keys: str, default: Any = None) -> Any:
    """
    Retrieve a nested configuration value.

    Example:
        get_config("final_model", "n_clusters")
    """

    value = CONFIG

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default

        value = value[key]

    return value


# ============================================================
# 4. RESOLVE PROJECT PATHS
# ============================================================

def resolve_path(path_value: str | Path) -> Path:
    """Convert a project-relative path into an absolute path."""

    path = Path(path_value)

    if path.is_absolute():
        return path

    return (ROOT_DIR / path).resolve()


def get_path(*keys: str) -> Path:
    """
    Retrieve and resolve a path from config.yaml.

    Example:
        get_path("data", "raw_path")
    """

    value = get_config(*keys)

    if not isinstance(value, (str, Path)):
        raise KeyError(
            f"Invalid or missing path configuration: {keys}"
        )

    return resolve_path(value)