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
