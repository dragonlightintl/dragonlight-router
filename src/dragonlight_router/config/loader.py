"""Configuration loader -- YAML file to RouterConfig model.

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
from dragonlight_router.core.validation import validate_provider_url
from dragonlight_router.result import Err, Ok, Result

logger = structlog.get_logger()


def _resolve_config_path(config_path: Path | None) -> Path | None:
    """Resolve config path from explicit argument or environment variable."""
    if config_path is not None:
        return config_path
    env_path = os.environ.get("DRAGONLIGHT_ROUTER_CONFIG")
    if env_path:
        return Path(env_path)
    return None


def load_config(config_path: Path | None = None) -> Result[RouterConfig, RouterConfigError]:
    """Load and validate router configuration.

    Falls back to defaults if no config file is found.
    """
    resolved = _resolve_config_path(config_path)

    if resolved is None:
        return Ok(RouterConfig())

    if not resolved.exists():
        logger.warning("config_file_not_found", path=str(resolved))
        return Ok(RouterConfig())

    return _load_yaml_config(resolved)


def _load_yaml_config(path: Path) -> Result[RouterConfig, RouterConfigError]:
    """Attempt to load config from a YAML file, returning Result."""
    assert path.exists(), f"config file must exist: {path}"
    try:
        config = _load_from_yaml(path)
        assert isinstance(config, RouterConfig), "loaded config must be RouterConfig"
        return Ok(config)
    except (yaml.YAMLError, OSError) as exc:
        logger.error("config_load_failed", path=str(path), error=str(exc))
        return Err(
            RouterConfigError(
                message=f"Failed to load config from {path}: {exc}", config_path=str(path)
            )
        )


def _validate_provider_urls(config: RouterConfig) -> None:
    """SEC-003: Validate all provider URLs against SSRF rules.

    Raises ValueError if any provider URL targets a private IP or
    uses an insecure scheme for non-localhost URLs.
    """
    for provider in config.providers:
        try:
            validate_provider_url(provider.base_url)
        except ValueError as exc:
            logger.warning(
                "provider_url_ssrf_rejected",
                provider=provider.name,
                url=provider.base_url,
                error=str(exc),
            )
            raise

        if provider.catalog_url:
            try:
                validate_provider_url(provider.catalog_url)
            except ValueError as exc:
                logger.warning(
                    "provider_catalog_url_ssrf_rejected",
                    provider=provider.name,
                    url=provider.catalog_url,
                    error=str(exc),
                )
                raise


def _load_from_yaml(path: Path) -> RouterConfig:
    """Parse a YAML config file into a validated RouterConfig."""
    try:
        text = path.read_text()
        data = yaml.safe_load(text) or {}
        config = RouterConfig(**data)
        _validate_provider_urls(config)
        return config
    except (yaml.YAMLError, OSError) as exc:
        logger.error("config_load_failed", path=str(path), error=str(exc))
        raise
