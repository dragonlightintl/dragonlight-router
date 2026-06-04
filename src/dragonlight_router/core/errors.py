"""Typed errors for the routing system."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RouterConfigError:
    """Configuration error — invalid YAML, missing required fields, etc."""

    message: str
    config_path: str | None = None


@dataclass(frozen=True)
class CatalogRefreshError:
    """Failed to refresh provider catalog."""

    provider: str
    message: str
    http_status: int | None = None


@dataclass(frozen=True)
class StatePersistenceError:
    """Failed to read or write state file."""

    path: str
    message: str
    operation: str  # "read" | "write"


@dataclass(frozen=True)
class ProviderNotFoundError:
    """Requested provider is not configured or not available."""

    provider: str
    message: str = "Provider not found"


@dataclass(frozen=True)
class ModelNotFoundError:
    """Requested model is not available in the catalog for the given provider."""

    provider: str
    model: str
    message: str = "Model not found"


@dataclass(frozen=True)
class StaleCatalogError:
    """Catalog data is stale and needs refreshing."""

    provider: str
    max_age_seconds: int
    age_seconds: int
    message: str = "Catalog is stale"