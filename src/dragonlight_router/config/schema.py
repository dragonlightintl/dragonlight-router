"""Pydantic configuration schema for the router.

All configuration is validated through these models.
RouterConfig is the top-level model loaded from YAML.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class RateLimitSchema(BaseModel):
    """Rate limit configuration for a provider."""

    rpm: int
    rpd: int | None = None
    tpm: int | None = None


class ProviderSchema(BaseModel):
    """Provider-level configuration from YAML."""

    name: str
    base_url: str
    catalog_url: str | None = None
    env_key: str | None = None
    model_prefix: str
    rate_limits: RateLimitSchema


class RouterConfig(BaseModel):
    """Top-level router configuration."""

    state_dir: Path = Path("./router_state")
    catalog_ttl_hours: int = 24
    budget_flush_interval_s: int = 5
    default_top_n: int = 12
    max_consecutive_same_provider: int = 2
    providers: list[ProviderSchema] = Field(default_factory=list)
