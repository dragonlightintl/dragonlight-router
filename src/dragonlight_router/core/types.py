"""Frozen data types for the routing system.

All configuration and request/response types are frozen dataclasses.
Mutable runtime state lives in state.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import AsyncIterator, Protocol, runtime_checkable


@unique
class BackendTier(Enum):
    """Capability tiers — abstract, not provider-specific."""

    LOCAL = "local"
    HAIKU = "haiku"
    SONNET = "sonnet"
    OPUS = "opus"


@unique
class BackendStatus(Enum):
    """Runtime health state of a single backend."""

    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    DAILY_CAP_HIT = "daily_cap_hit"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"
    OFFLINE = "offline"


@dataclass(frozen=True)
class BackendCapabilities:
    """Immutable capability declaration for a backend."""

    max_context_tokens: int
    supports_tool_use: bool
    supports_streaming: bool
    supports_json_mode: bool
    supports_system_prompts: bool


@dataclass(frozen=True)
class BackendCostProfile:
    """Per-token cost structure. All values in USD per million tokens."""

    input_per_mtok: float
    output_per_mtok: float
    cache_read_per_mtok: float = 0.0
    cache_write_per_mtok: float = 0.0


@dataclass(frozen=True)
class BackendRateLimits:
    """Provider-imposed rate limits."""

    rpm: int
    rpd: int
    tpm: int
    daily_token_cap: int


@dataclass(frozen=True)
class BackendConfig:
    """Complete, immutable configuration for a single backend.

    Frozen dataclass — constructed once at boot, never mutated.
    Runtime state lives in BackendState (separate, mutable).
    """

    name: str
    provider: str
    model: str
    tier: BackendTier
    base_url: str
    env_key: str | None
    capabilities: BackendCapabilities
    cost: BackendCostProfile
    rate_limits: BackendRateLimits
    priority: int = 0


@dataclass(frozen=True)
class ProviderConfig:
    """Provider-level configuration (config-driven)."""

    name: str
    base_url: str
    catalog_url: str | None
    env_key: str | None
    model_prefix: str
    rpm_limit: int
    rpd_limit: int | None
    tpm_limit: int | None


@dataclass(frozen=True)
class DispatchOrder:
    """Immutable request from server to cascade router."""

    intent_category: str
    specific_intent: str
    operator_message: str
    system_prompt: str
    context_tokens: int
    requires_tool_use: bool = False
    requires_long_context: bool = False
    persona: str | None = None
    request_id: int | None = None
    stream_id: str | None = None


@dataclass(frozen=True)
class EngineResponse:
    """Immutable response from cascade router to server."""

    content: str
    backend_used: str
    backend_tier: BackendTier
    tokens_in: int
    tokens_out: int
    estimated_cost_usd: float
    latency_ms: float
    was_fallback: bool
    fallback_chain: list[str]


@dataclass(frozen=True)
class DispatchFailure:
    """Returned when all backends in the cascade are exhausted."""

    message: str
    attempted_backends: list[str]
    error_details: dict[str, str]


@dataclass(frozen=True)
class ComplexityEstimate:
    """Output of the reasoning tier heuristic."""

    tier: BackendTier
    confidence: float
    signals: list[str]


@dataclass(frozen=True)
class BackendError:
    """Base error from a backend dispatch attempt."""

    backend_name: str
    error_type: str
    message: str
    http_status: int | None = None
    retryable: bool = False


@dataclass(frozen=True)
class ModelScore:
    """Composite score for a model candidate."""

    model_id: str
    provider: str
    rank: int
    budget_score: float
    health_score: float
    composite: float


@dataclass(frozen=True)
class CatalogEntry:
    """One model from a provider's catalog."""

    model_id: str
    provider: str
    created: int | None = None


@runtime_checkable
class GenerativeBackend(Protocol):
    """Protocol that every backend adapter must implement."""

    @property
    def config(self) -> BackendConfig: ...

    @property
    def status(self) -> BackendStatus: ...

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]: ...

    async def health_check(self) -> bool: ...

    def record_usage(self, tokens_in: int, tokens_out: int) -> None: ...
