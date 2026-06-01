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
from dragonlight_router.core.errors import RouterConfigError
from dragonlight_router.result import Err, Ok, Result

logger = structlog.get_logger()


def load_config(config_path: Path | None = None) -> Result[RouterConfig, RouterConfigError]:
    """Load and validate router configuration.

    Falls back to defaults if no config file is found.
    """
    # Resolution: explicit path → env var → defaults
    if config_path is None:
        env_path = os.environ.get("DRAGONLIGHT_ROUTER_CONFIG")
        if env_path:
            config_path = Path(env_path)

    if config_path is not None:
        if config_path.exists():
            # Try to load the YAML file
            try:
                return Ok(_load_from_yaml(config_path))
            except (yaml.YAMLError, OSError) as exc:
                logger.error("config_load_failed", path=str(config_path), error=str(exc))
                return Err(RouterConfigError(
                    message=f"Failed to load config from {config_path}: {exc}",
                    config_path=str(config_path)
                ))
        else:
            # File doesn't exist - fall back to defaults (original behavior)
            logger.warning("config_file_not_found", path=str(config_path))
            return Ok(RouterConfig())

    # No config path provided - use defaults
    return Ok(RouterConfig())


def _load_from_yaml(path: Path) -> RouterConfig:
    """Parse a YAML config file into a validated RouterConfig."""
    try:
        text = path.read_text()
        data = yaml.safe_load(text) or {}
        return RouterConfig(**data)
    except (yaml.YAMLError, OSError) as exc:
        logger.error("config_load_failed", path=str(path), error=str(exc))
        raise
