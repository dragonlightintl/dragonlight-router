"""Configuration loader — YAML file → RouterConfig model.

Resolution order:
1. Canonical config path (if provided)
2. Environment variable DRAGONLIGHT_ROUTER_CONFIG
3. Default empty config (all defaults)
"""
from __future__ import annotations

import os
from pathlib import Path

import structlog
import yaml

from dragonlight_router.config.schema import RouterConfig

logger = structlog.get_logger()


def load_config(config_path: Path | None = None) -> RouterConfig:
    """Load and validate router configuration.

    Falls back to defaults if no config file is found.
    """
    # Resolution: explicit path → env var → defaults
    if config_path is None:
        env_path = os.environ.get("DRAGONLIGHT_ROUTER_CONFIG")
        if env_path:
            config_path = Path(env_path)

    if config_path is not None and config_path.exists():
        return _load_from_yaml(config_path)

    if config_path is not None and not config_path.exists():
        logger.warning("config_file_not_found", path=str(config_path))

    return RouterConfig()


def _load_from_yaml(path: Path) -> RouterConfig:
    """Parse a YAML config file into a validated RouterConfig."""
    try:
        text = path.read_text()
        data = yaml.safe_load(text) or {}
        return RouterConfig(**data)
    except (yaml.YAMLError, OSError) as exc:
        logger.error("config_load_failed", path=str(path), error=str(exc))
        raise