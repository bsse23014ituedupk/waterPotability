"""
Centralized configuration loader for the Water Potability Prediction System.

Loads config/config.yaml and exposes all parameters as typed namespace objects.
No hardcoded values should exist anywhere else in the codebase.
"""

import os
import yaml
from types import SimpleNamespace


def _dict_to_namespace(d: dict) -> SimpleNamespace:
    """Recursively convert a dict to a SimpleNamespace for attribute access."""
    ns = SimpleNamespace()
    for key, value in d.items():
        if isinstance(value, dict):
            setattr(ns, key, _dict_to_namespace(value))
        else:
            setattr(ns, key, value)
    return ns


def load_config(config_path: str = None) -> SimpleNamespace:
    """
    Load and parse the YAML configuration file.

    Args:
        config_path: Path to config.yaml. Defaults to config/config.yaml
                     relative to project root.

    Returns:
        SimpleNamespace with nested attribute access (e.g. config.data.filepath).

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    if config_path is None:
        # Resolve relative to this file's grandparent (project root)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", "config.yaml")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _dict_to_namespace(raw)


# Module-level singleton — import and use directly:  from src.config import CONFIG
CONFIG = load_config()
