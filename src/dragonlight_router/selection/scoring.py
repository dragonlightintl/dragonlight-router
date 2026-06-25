"""Composite scoring functions for model selection.

Scores combine rank (role-matrix position), budget availability,
and health state into a single comparable float.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import BackendConfig, DispatchOrder
from dragonlight_router.health.tracker import HealthTracker


def compute_budget_score(
    rpm_remaining: int,
    rpm_limit: int,
    rpd_remaining: int | None,
    rpd_limit: int | None,
) -> float:
    """Score budget availability on 0-100 scale.

    Returns the minimum of RPM and RPD utilization ratios, scaled to 0-100.
    None RPD limits are treated as unlimited (100% available).
    """
    assert rpm_limit > 0, f"rpm_limit must be > 0, got {rpm_limit}"
    assert 0 <= rpm_remaining <= rpm_limit, (
        f"rpm_remaining must be in [0, rpm_limit], got {rpm_remaining}"
    )

    rpm_ratio = rpm_remaining / rpm_limit * 100.0
    rpd_ratio = _compute_rpd_ratio(rpd_remaining, rpd_limit)

    result = min(rpm_ratio, rpd_ratio)
    assert 0.0 <= result <= 100.0, f"budget score out of bounds: {result}"
    return result


def _compute_rpd_ratio(rpd_remaining: int | None, rpd_limit: int | None) -> float:
    """Compute RPD utilization ratio, treating None as unlimited."""
    if rpd_remaining is None or rpd_limit is None:
        return 100.0

    assert rpd_limit > 0, f"rpd_limit must be > 0, got {rpd_limit}"
    assert 0 <= rpd_remaining <= rpd_limit, (
        f"rpd_remaining must be in [0, rpd_limit], got {rpd_remaining}"
    )
    return rpd_remaining / rpd_limit * 100.0


def compute_health_score(
    error_count: int,
    circuit_open: bool,
    last_success_age_s: float,
) -> float:
    """Score backend health on 0-100 scale.

    - Circuit open -> always 0.
    - 0 errors -> 100.
    - 1-2 errors -> 70.
    - 3+ errors -> 30.
    """
    assert error_count >= 0, f"error_count must be >= 0, got {error_count}"
    assert last_success_age_s >= 0, f"last_success_age_s must be >= 0, got {last_success_age_s}"

    if circuit_open:
        return 0.0

    if error_count == 0:
        return 100.0
    elif error_count <= 2:
        return 70.0
    else:
        return 30.0


def compute_composite_score(rank: int, budget_score: float, health_score: float) -> float:
    """Weighted composite: rank 60%, budget 25%, health 15%.

    All inputs should be on 0-100 scale. Output is 0-100.
    """
    assert 0 <= rank <= 100, f"rank must be between 0 and 100, got {rank}"
    assert 0 <= budget_score <= 100, f"budget_score must be between 0 and 100, got {budget_score}"
    assert 0 <= health_score <= 100, f"health_score must be between 0 and 100, got {health_score}"

    result = rank * 0.6 + budget_score * 0.25 + health_score * 0.15

    assert 0 <= result <= 100, f"computed score {result} must be between 0 and 100"
    return result


class ScoringWeights(Enum):
    """Legacy canonical scoring weights for the dispatch path (MBR->CBR->LBR cascade).

    NOTE: ScoringWeightsConfig dataclass is the source of truth for weights.
    This enum is retained for backward compatibility. With IBR activation,
    the 6-dimension ScoringWeightsConfig (including spectrograph_match) is
    the authoritative weight vector.

    Default values (IBR-active, 6-dimension):
    - cost: 0.20
    - latency: 0.25
    - priority: 0.20
    - queue: 0.10
    - health: 0.10
    - spectrograph_match: 0.15

    Note: HEALTH and QUEUE both have value 0.10, which makes HEALTH an alias.
    """

    COST = 0.20
    LATENCY = 0.25
    PRIORITY = 0.20
    QUEUE = 0.10
    HEALTH = 0.10
    SPECTROGRAPH_MATCH = 0.15


assert (
    abs(
        sum(
            [
                ScoringWeights.COST.value,
                ScoringWeights.LATENCY.value,
                ScoringWeights.PRIORITY.value,
                ScoringWeights.QUEUE.value,
                ScoringWeights.HEALTH.value,
                ScoringWeights.SPECTROGRAPH_MATCH.value,
            ]
        )
        - 1.0
    )
    < 1e-9
), "ScoringWeights must sum to 1.0"


@dataclass(frozen=True)
class ScoringWeightsConfig:
    """Configuration object for scoring weights.

    When IBR is active, spectrograph_match carries a non-zero weight and the
    other weights are adjusted so the total remains 1.0.  When IBR is
    disabled (spectrograph_match=0.0), behavior is identical to v0.3.0.
    """

    cost: float = 0.20
    latency: float = 0.25
    priority: float = 0.20
    queue: float = 0.10
    health: float = 0.10
    spectrograph_match: float = 0.15

    def __post_init__(self) -> None:
        """Validate that weights sum to 1.0."""
        total = (
            self.cost
            + self.latency
            + self.priority
            + self.queue
            + self.health
            + self.spectrograph_match
        )
        assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"


@dataclass(frozen=True)
class ScoringContext:
    """Grouped context for candidate scoring (QA-004 compliance)."""

    config: BackendConfig
    order: DispatchOrder
    weights: ScoringWeightsConfig
    budget_tracker: BudgetTracker
    health_tracker: HealthTracker | None


def normalize_rank(rank: int) -> float:
    """Normalize rank to [0.0, 1.0] where 1.0 is best rank.

    Args:
        rank: Rank position (1 = best, higher numbers = worse)

    Returns:
        Normalized score in [0.0, 1.0]
    """
    assert rank >= 1, f"Rank must be >= 1, got {rank}"
    normalized: float = max(0.0, min(1.0, 2.0 ** (1 - rank / 10.0)))
    assert 0.0 <= normalized <= 1.0, f"Normalized rank out of bounds: {normalized}"
    return normalized


def normalize_budget_score(budget_score: float) -> float:
    """Normalize budget score to [0.0, 1.0].

    Args:
        budget_score: Budget availability score (0-100)

    Returns:
        Normalized score in [0.0, 1.0]
    """
    assert 0.0 <= budget_score <= 100.0, f"Budget score must be 0-100, got {budget_score}"
    normalized = budget_score / 100.0
    assert 0.0 <= normalized <= 1.0, f"Normalized budget score out of bounds: {normalized}"
    return normalized


def normalize_latency_score(latency_score: float) -> float:
    """Normalize latency score to [0.0, 1.0] where 1.0 is best latency.

    Args:
        latency_score: Latency score (0-100, higher = better)

    Returns:
        Normalized score in [0.0, 1.0]
    """
    assert 0.0 <= latency_score <= 100.0, f"Latency score must be 0-100, got {latency_score}"
    normalized = latency_score / 100.0
    assert 0.0 <= normalized <= 1.0, f"Normalized latency score out of bounds: {normalized}"
    return normalized


def normalize_priority_score(priority: int) -> float:
    """Normalize priority to [0.0, 1.0] where higher priority = better.

    Args:
        priority: Priority value (typically 0-100)

    Returns:
        Normalized score in [0.0, 1.0]
    """
    assert priority >= 0, f"Priority must be >= 0, got {priority}"
    capped_priority = min(priority, 100)
    normalized = capped_priority / 100.0
    assert 0.0 <= normalized <= 1.0, f"Normalized priority out of bounds: {normalized}"
    return normalized


def normalize_queue_score(queue_depth: int, max_queue_depth: int = 100) -> float:
    """Normalize queue depth to [0.0, 1.0] where 1.0 is best (empty queue).

    Args:
        queue_depth: Current queue depth
        max_queue_depth: Maximum expected queue depth for normalization

    Returns:
        Normalized score in [0.0, 1.0] (1.0 = no queue, 0.0 = max queue)
    """
    assert queue_depth >= 0, f"Queue depth must be >= 0, got {queue_depth}"
    assert max_queue_depth > 0, f"Max queue depth must be > 0, got {max_queue_depth}"
    normalized = max(0.0, min(1.0, 1.0 - (queue_depth / max_queue_depth)))
    assert 0.0 <= normalized <= 1.0, f"Normalized queue score out of bounds: {normalized}"
    return normalized


def normalize_health_score(health_score: float) -> float:
    """Normalize health score to [0.0, 1.0].

    Args:
        health_score: Health score (0-100, higher = healthier)

    Returns:
        Normalized score in [0.0, 1.0]
    """
    assert 0.0 <= health_score <= 100.0, f"Health score must be 0-100, got {health_score}"
    normalized = health_score / 100.0
    assert 0.0 <= normalized <= 1.0, f"Normalized health score out of bounds: {normalized}"
    return normalized


# DEVIATION CS-PARAM-001: score_candidate takes 6 params — dataclass grouping would break API.
def score_candidate(
    config: BackendConfig,
    order: DispatchOrder,
    weights: ScoringWeightsConfig,
    budget_tracker: BudgetTracker,
    health_tracker: HealthTracker | None,
    spectrograph_match: float = 0.0,
) -> float:
    """Score a single candidate using canonical ScoringWeights.

    Args:
        config: Backend configuration to score
        order: Dispatch order for context
        weights: Scoring weights to apply
        budget_tracker: Budget tracker for budget/latency scores
        health_tracker: Health tracker for health/priority scores
        spectrograph_match: Spectrograph match score in [0.0, 1.0] from IBR stage

    Returns:
        Composite score in [0.0, 1.0]
    """
    assert isinstance(config, BackendConfig), "config must be BackendConfig"
    assert isinstance(weights, ScoringWeightsConfig), "weights must be ScoringWeightsConfig"
    assert 0.0 <= spectrograph_match <= 1.0, (
        f"spectrograph_match must be in [0.0, 1.0], got {spectrograph_match}"
    )

    raw = _extract_raw_scores(config, budget_tracker, health_tracker)
    normalized = _normalize_all_dimensions(raw, spectrograph_match=spectrograph_match)
    composite = _apply_weights(normalized, weights)

    assert 0.0 <= composite <= 1.0, f"Composite score out of bounds: {composite}"
    return composite


@dataclass(frozen=True)
class _RawScores:
    """Raw scores extracted from trackers before normalization."""

    budget: float
    health: float
    rank: float
    latency: float
    priority: float
    queue: float


@dataclass(frozen=True)
class _NormalizedScores:
    """Scores normalized to [0.0, 1.0] range."""

    rank: float
    budget: float
    latency: float
    priority: float
    queue: float
    health: float
    spectrograph_match: float = 0.0


def _extract_raw_scores(
    config: BackendConfig,
    budget_tracker: BudgetTracker,
    health_tracker: HealthTracker | None,
) -> _RawScores:
    """Extract raw scores from trackers for a candidate."""
    assert isinstance(config, BackendConfig), "config must be BackendConfig"
    assert isinstance(budget_tracker, BudgetTracker), "budget_tracker must be BudgetTracker"

    budget_result = budget_tracker.score(config.provider)
    budget_score = budget_result.value if hasattr(budget_result, "value") else 50.0

    health_score = 50.0
    if health_tracker is not None:
        health_result = health_tracker.score(config.model)
        health_score = health_result.value if hasattr(health_result, "value") else 50.0

    avg_cost = (config.cost.input_per_mtok + config.cost.output_per_mtok) / 2.0
    rank_score = min(100.0 / (avg_cost + 1.0), 100.0) if avg_cost >= 0 else 50.0

    assert 0.0 <= budget_score <= 100.0, f"budget_score out of range: {budget_score}"
    return _RawScores(
        budget=budget_score,
        health=health_score,
        rank=rank_score,
        latency=health_score,
        priority=config.priority,
        queue=budget_score,
    )


def _normalize_cost_score(cost_score: float) -> float:
    """Normalize cost-efficiency score to [0.0, 1.0].

    The raw cost score is on a 0-100 scale where higher = cheaper/better.
    This is a direct score, not a rank position, so we simply divide by 100.

    Args:
        cost_score: Cost-efficiency score (0-100, higher = cheaper)

    Returns:
        Normalized score in [0.0, 1.0]
    """
    assert 0.0 <= cost_score <= 100.0, f"cost_score must be 0-100, got {cost_score}"
    normalized = cost_score / 100.0
    assert 0.0 <= normalized <= 1.0, f"Normalized cost score out of bounds: {normalized}"
    return normalized


def _normalize_all_dimensions(
    raw: _RawScores, spectrograph_match: float = 0.0
) -> _NormalizedScores:
    """Normalize all raw scores to [0.0, 1.0]."""
    assert isinstance(raw, _RawScores), "raw must be _RawScores"
    assert raw.budget >= 0, "budget score must be non-negative"
    assert 0.0 <= spectrograph_match <= 1.0, (
        f"spectrograph_match must be in [0.0, 1.0], got {spectrograph_match}"
    )

    return _NormalizedScores(
        rank=_normalize_cost_score(raw.rank),
        budget=normalize_budget_score(raw.budget),
        latency=normalize_latency_score(raw.latency),
        priority=normalize_priority_score(int(raw.priority)),
        queue=normalize_queue_score(int(100 - raw.budget)),
        health=normalize_health_score(raw.health),
        spectrograph_match=spectrograph_match,
    )


def _apply_weights(normalized: _NormalizedScores, weights: ScoringWeightsConfig) -> float:
    """Compute weighted composite from normalized scores."""
    assert isinstance(weights, ScoringWeightsConfig), "weights must be ScoringWeightsConfig"

    composite = (
        normalized.rank * weights.cost
        + normalized.latency * weights.latency
        + normalized.priority * weights.priority
        + normalized.queue * weights.queue
        + normalized.health * weights.health
        + normalized.spectrograph_match * weights.spectrograph_match
    )

    assert 0.0 <= composite <= 1.0, f"Composite out of bounds: {composite}"
    return composite


def cost_governor_active(
    daily_spend: float,
    monthly_spend: float,
    config: dict[str, Any],
) -> bool:
    """Check if cost governor should be active.

    Args:
        daily_spend: Current daily spend in USD
        monthly_spend: Current monthly spend in USD
        config: Router configuration containing thresholds

    Returns:
        True if cost governor should override weights
    """
    assert isinstance(daily_spend, (int, float)), "daily_spend must be numeric"
    assert isinstance(monthly_spend, (int, float)), "monthly_spend must be numeric"

    daily_threshold: float = float(config.get("cost_down_threshold_daily", 100.0))
    monthly_threshold: float = float(config.get("cost_down_threshold_monthly", 1000.0))

    return daily_spend >= daily_threshold or monthly_spend >= monthly_threshold


def cost_adjusted_weights(
    base_weights: ScoringWeightsConfig,
) -> ScoringWeightsConfig:
    """Adjust weights when cost governor is active.

    Without IBR: cost=0.70, latency=0.10, priority=0.10, queue=0.05, health=0.05
    With IBR:    cost=0.65, latency=0.10, priority=0.10, queue=0.05, health=0.05,
                 spectrograph_match=0.05  (IBR-SCORE-05)
    """
    assert isinstance(base_weights, ScoringWeightsConfig), (
        "base_weights must be ScoringWeightsConfig"
    )
    total = sum(
        [
            base_weights.cost,
            base_weights.latency,
            base_weights.priority,
            base_weights.queue,
            base_weights.health,
            base_weights.spectrograph_match,
        ]
    )
    assert abs(total - 1.0) < 1e-9, "base weights must sum to 1.0"

    if base_weights.spectrograph_match > 0.0:
        # IBR active: cost governor reduces spectrograph_match to 0.05 (IBR-SCORE-05)
        return ScoringWeightsConfig(
            cost=0.65,
            latency=0.10,
            priority=0.10,
            queue=0.05,
            health=0.05,
            spectrograph_match=0.05,
        )
    return ScoringWeightsConfig(
        cost=0.70,
        latency=0.10,
        priority=0.10,
        queue=0.05,
        health=0.05,
    )


# ---------------------------------------------------------------------------
# Per-intent-category CBR weight profiles (IBR weight adaptation)
# ---------------------------------------------------------------------------

# Low-stakes categories: prioritize cost and speed.
_LOW_STAKES_WEIGHTS = ScoringWeightsConfig(
    cost=0.35,
    latency=0.30,
    priority=0.10,
    spectrograph_match=0.10,
    queue=0.10,
    health=0.05,
)

# High-stakes categories: prioritize capability and quality.
_HIGH_STAKES_WEIGHTS = ScoringWeightsConfig(
    cost=0.10,
    latency=0.10,
    priority=0.30,
    spectrograph_match=0.25,
    queue=0.10,
    health=0.15,
)

# Categories classified as low-stakes — fast, cheap models preferred.
_LOW_STAKES_INTENTS: frozenset[str] = frozenset({
    "test_generation",
    "test_property",
    "audit",
    "data_analysis",
    "summarization",
})

# Categories classified as high-stakes — capability and precision preferred.
_HIGH_STAKES_INTENTS: frozenset[str] = frozenset({
    "implementation",
    "implementation_complex",
    "coherence_merge",
    "complex_reasoning",
    "strategic_planning",
    "architecture",
})


def intent_weights_for_category(intent_category: str) -> ScoringWeightsConfig:
    """Return per-intent-category CBR weight profile.

    Low-stakes intents (test_generation, audit, etc.) shift weight toward
    cost and speed. High-stakes intents (implementation, coherence_merge,
    etc.) shift weight toward capability and spectrograph match quality.

    Unrecognized categories receive the default ScoringWeightsConfig.

    Args:
        intent_category: The intent category from the DispatchOrder.

    Returns:
        ScoringWeightsConfig tuned for the intent category.
    """
    assert isinstance(intent_category, str), "intent_category must be a string"

    if intent_category in _LOW_STAKES_INTENTS:
        return _LOW_STAKES_WEIGHTS
    if intent_category in _HIGH_STAKES_INTENTS:
        return _HIGH_STAKES_WEIGHTS
    return ScoringWeightsConfig()
