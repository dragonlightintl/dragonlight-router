"""Pydantic configuration schema for the router.

All configuration is validated through these models.
RouterConfig is the top-level model loaded from YAML.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class RateLimitSchema(BaseModel):
    """Rate limit configuration for a provider."""

    model_config = ConfigDict(frozen=True)

    rpm: int
    rpd: int | None = None
    tpm: int | None = None
    daily_token_cap: int | None = None


class ProviderSchema(BaseModel):
    """Provider-level configuration from YAML."""

    model_config = ConfigDict(frozen=True)

    name: str
    base_url: str
    catalog_url: str | None = None
    env_key: str | None = None
    model_prefix: str
    rate_limits: RateLimitSchema


class IntentClassificationConfig(BaseModel):
    """IBR intent classification configuration.

    Controls the Intent Based Router (IBR) subsystem.  When disabled
    (the default), the pipeline behaves identically to v0.3.0.
    See IBR spec v0.1.0 section 6.1 for field semantics.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    timeout_ms: int = 100
    cache_ttl_s: int = 300
    cache_max_entries: int = 5000
    confidence_threshold: float = 0.6
    profile_confidence_threshold: float = 0.3
    flavor_match_weight: float = 0.15
    flavor_match_weight_governor: float = 0.05


class PinnedDispatchConfig(BaseModel):
    """Configuration for pinned (direct model) dispatch.

    Controls operational guardrails when a caller pins a specific backend
    via DispatchOrder.model, bypassing the cascade pipeline.
    """

    model_config = ConfigDict(frozen=True)

    honor_health: bool = True


class RouterConfig(BaseModel):
    """Top-level router configuration."""

    model_config = ConfigDict(frozen=True)

    state_dir: Path = Path("./router_state")
    catalog_ttl_hours: int = 24
    budget_flush_interval_s: int = 5
    default_top_n: int = 12
    max_consecutive_same_provider: int = 2
    providers: list[ProviderSchema] = Field(default_factory=list)
    # HAZ-011: Bearer token for admin endpoints (retire, reinstate, catalog/refresh).
    # When set, admin endpoints require Authorization: Bearer <token>.
    # When empty/None, admin endpoints are open (backward compatible).
    admin_api_key: str | None = None
    # IBR: Intent classification subsystem (opt-in, disabled by default).
    intent_classification: IntentClassificationConfig = Field(
        default_factory=IntentClassificationConfig,
    )
    # Model pinning: operational guardrails for direct-dispatch bypass.
    pinned_dispatch: PinnedDispatchConfig = Field(
        default_factory=PinnedDispatchConfig,
    )
