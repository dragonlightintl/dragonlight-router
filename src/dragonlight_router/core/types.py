"""Frozen data types for the routing system.
All configuration and request/response types are frozen dataclasses.
Mutable runtime state lives in state.py.
Canonical Result type for fallible operations.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum, unique
from typing import Generic, NoReturn, Protocol, TypeVar, runtime_checkable

T = TypeVar("T")
E = TypeVar("E")


@dataclass(frozen=True)
class Ok(Generic[T]):
    """Successful result containing a value."""

    value: T

    def is_ok(self) -> bool:
        """Return True because this is an Ok."""
        return True

    def is_err(self) -> bool:
        """Return False because this is not an Err."""
        return False

    def unwrap(self: Ok[T]) -> T:
        """Return the contained value."""
        return self.value

    def unwrap_err(self: Ok[T]) -> NoReturn:
        """Raise AssertionError because this is an Ok value."""
        raise AssertionError("Called unwrap_err on Ok value")


@dataclass(frozen=True)
class Err(Generic[E]):
    """Failed result containing an error."""

    error: E

    def is_ok(self) -> bool:
        """Return False because this is not an Ok."""
        return False

    def is_err(self) -> bool:
        """Return True because this is an Err."""
        return True

    def unwrap(self: Err[E]) -> NoReturn:
        """Raise AssertionError because this is an Err value."""
        raise AssertionError("Called unwrap on Err value")

    def unwrap_err(self: Err[E]) -> E:
        """Return the contained error."""
        return self.error


Result = Ok[T] | Err[E]


@unique
class BackendTier(Enum):
    """Capability tiers — abstract, not provider-specific."""

    LOCAL = "local"
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


@unique
class BackendStatus(Enum):
    """Runtime health state of a single backend."""

    AVAILABLE = "available"
    RATE_LIMITED = "rate_limited"
    DAILY_CAP_HIT = "daily_cap_hit"
    ERROR = "error"
    CIRCUIT_OPEN = "circuit_open"
    OFFLINE = "offline"
    DEGRADED = "degraded"
    RETIRED = "retired"
    KEY_INVALID = "key_invalid"


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
class LatencySLO:
    """Latency Service Level Objective for health checking."""

    latency_ms: float
    description: str = ""


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
    daily_token_cap: int | None


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
    context_trust_tier: str | None = None
    # HAZ-004: Fallback policy — controls cascade behavior on primary failure.
    #   "allow" (default): fall back to next candidate in ranked list
    #   "deny": fail immediately if primary candidate fails (no fallback)
    #   "same_tier": only fall back to candidates at the same BackendTier
    fallback_policy: str = "allow"
    # Model pinning: when set, bypass cascade and dispatch directly to
    # this backend.  Value is the backend name as registered in
    # BackendRegistry (e.g. "anthropic/claude-sonnet-4-20250514").
    model: str | None = None
    # Tool-use support: OpenAI-format tool definitions and structured messages.
    # When tools is set, the adapter layer includes them in the API request body.
    tools: tuple[dict, ...] | None = None
    tool_choice: str | dict | None = None
    # Structured message list for multi-turn tool-use conversations.
    # When set, takes precedence over operator_message/system_prompt for
    # building the API request body. Each dict follows the OpenAI message
    # format: {"role": "...", "content": "...", ...} including tool/assistant
    # messages with tool_calls and tool_call_id.
    messages: tuple[dict, ...] | None = None


# ---------------------------------------------------------------------------
# IBR (Intent Based Router) types — frozen dataclasses for intent
# classification and model flavor profiling.  See IBR spec v0.1.0.
# ---------------------------------------------------------------------------

# Validation constants — canonical allowed values for ClassifiedIntent fields.
IBR_TASK_TYPES: frozenset[str] = frozenset(
    {
        "generation",
        "analysis",
        "refactoring",
        "summarization",
        "creative",
        "reasoning",
        "lookup",
        "translation",
    }
)
IBR_DOMAINS: frozenset[str] = frozenset(
    {
        "code",
        "technical",
        "legal",
        "business",
        "creative_writing",
        "general",
    }
)
IBR_QUALITY_SPEED: frozenset[str] = frozenset({"quality", "balanced", "speed"})


@dataclass(frozen=True)
class ClassifiedIntent:
    """Output of IBR intent classification.

    Represents three orthogonal dimensions of a request: what it needs done
    (task_type), the subject matter (domain), and the latency-quality
    tradeoff preference (quality_speed).
    """

    task_type: str  # One of IBR_TASK_TYPES
    domain: str  # One of IBR_DOMAINS
    quality_speed: str  # One of IBR_QUALITY_SPEED
    confidence: float  # 0.0-1.0 classifier self-reported confidence
    latency_ms: float  # Wall-clock time for classification
    from_cache: bool  # Whether this was a cache hit


@dataclass(frozen=True)
class SpectrographScore:
    """Single-dimension score within a model spectrograph profile.

    score is the model's relative strength on this dimension (0.0-1.0).
    confidence indicates how much data backs the score (0.0-1.0).
    sample_count is the number of observations behind the score.
    """

    score: float  # 0.0-1.0 relative strength
    confidence: float  # 0.0-1.0 how much data backs this score
    sample_count: int  # Number of observations behind this score


# Neutral default for missing spectrograph dimensions.
IBR_NEUTRAL_SPECTROGRAPH: SpectrographScore = SpectrographScore(
    score=0.5, confidence=0.0, sample_count=0
)


@dataclass(frozen=True)
class ModelSpectrographProfile:
    """Full spectrograph profile for a single model.

    Maps a model_id to its relative strengths across IBR intent dimensions.
    Missing dimensions default to IBR_NEUTRAL_SPECTROGRAPH.
    """

    model_id: str
    version: int  # Profile schema version (for migration)
    updated_at: str  # ISO-8601 timestamp of last update
    task_scores: dict[str, SpectrographScore]  # task_type -> SpectrographScore
    domain_scores: dict[str, SpectrographScore]  # domain -> SpectrographScore
    qs_scores: dict[str, SpectrographScore]  # quality_speed -> SpectrographScore


@dataclass(frozen=True)
class IBRScoringContext:
    """Context passed from IBR into CBR scoring.

    Carries the classification result, loaded spectrograph profiles, and the
    effective spectrograph_match weight so CBR can incorporate the signal.
    """

    classified_intent: ClassifiedIntent | None
    spectrograph_profiles: dict[str, ModelSpectrographProfile]
    spectrograph_match_weight: float


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
    classified_intent: ClassifiedIntent | None = None
    spectrograph_match_score: float | None = None
    ibr_active: bool = False
    dispatch_mode: str = "cascade"
    # Tool-use support: raw tool_call objects from the LLM API response.
    # Present when the model returns tool_calls instead of (or alongside) text content.
    tool_calls: list[dict] | None = None
    # Finish reason from the API: "stop", "tool_calls", "length", etc.
    finish_reason: str | None = None


@dataclass(frozen=True)
class DispatchFailure:
    """Returned when all backends in the cascade are exhausted."""

    message: str
    attempted_backends: list[str]
    error_details: dict[str, str]


# ---------------------------------------------------------------------------
# Pinned dispatch error types (model-pinning v0.1.0 spec section 2.4).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelNotFoundError:
    """Pinned model not found in registry."""

    model: str
    message: str


@dataclass(frozen=True)
class ModelUnhealthyError:
    """Pinned model is circuit-open or retired."""

    model: str
    status: str  # "circuit_open" | "retired"
    message: str


@dataclass(frozen=True)
class BudgetExhaustedError:
    """Pinned model's provider has exhausted its budget or rate limits."""

    model: str
    provider: str
    message: str


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
class ScoredCandidate:
    """A BackendConfig paired with its composite selection score.

    Carries the CBR composite score alongside the config so that
    downstream stages (LBR, final selection) can use the score for
    weighted random selection and observability without re-computing it.
    """

    config: BackendConfig
    score: float


@dataclass(frozen=True)
class CatalogEntry:
    """One model from a provider's catalog."""

    model_id: str
    provider: str
    created: int | None = None


@dataclass(frozen=True)
class StreamChunk:
    """A single chunk from a streaming dispatch response.

    event_type is one of:
        "token"    — a content token from the LLM
        "metadata" — final metadata after generation completes
        "error"    — an error during generation
    """

    event_type: str
    content: str = ""
    backend_used: str = ""
    backend_tier: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    estimated_cost_usd: float = 0.0
    latency_ms: float = 0.0
    was_fallback: bool = False
    fallback_chain: list[str] | None = None
    error_message: str = ""
    classified_intent_json: str = ""
    spectrograph_match_score: float | None = None
    ibr_active: bool = False
    dispatch_mode: str = "cascade"
    # Tool-use support in streaming metadata chunks.
    tool_calls: list[dict] | None = None
    finish_reason: str | None = None


@dataclass(frozen=True)
class RequestOutcome:
    """Immutable record of a request outcome for budget/health tracking."""

    provider: str
    model_id: str
    success: bool
    tokens_used: int = 0
    latency_ms: float = 0.0
    quality_rating: int | None = None


@runtime_checkable
class GenerativeBackend(Protocol):
    """Protocol that every backend adapter must implement."""

    @property
    def config(self) -> BackendConfig: ...
    @property
    def status(self) -> BackendStatus: ...
    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        stream: bool = True,
    ) -> AsyncIterator[str]: ...
    async def health_check(self) -> bool: ...
    def record_usage(self, tokens_in: int, tokens_out: int) -> None: ...
