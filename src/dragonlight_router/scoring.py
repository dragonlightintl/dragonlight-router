"""Composite scoring functions for model selection.

Scores combine rank (role-matrix position), budget availability,
and health state into a single comparable float.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from dragonlight_router.core.types import BackendConfig


def compute_composite_score(rank: int, budget_score: float, health_score: float) -> float:
    """Weighted composite: rank 60%, budget 25%, health 15%.

    All inputs should be on 0-100 scale. Output is 0-100.
    """
    # Precondition assertions
    assert 0 <= rank <= 100, f'rank must be between 0 and 100, got {rank}'
    assert 0.0 <= budget_score <= 100.0, f'budget_score must be between 0 and 100, got {budget_score}'
    assert 0.0 <= health_score <= 100.0, f'health_score must be between 0 and 100, got {health_score}'

    result = rank * 0.6 + budget_score * 0.25 + health_score * 0.15

    # Postcondition assertion
    assert 0.0 <= result <= 100.0, f'computed composite score out of bounds: {result}'
    return result


def compute_budget_score(
    rpm_remaining: int,
    rpm_limit: int,
    rpd_remaining: int | None,
    rpd_limit: int | None,
) -> float:
    """Budget availability score (0-100).

    Returns min(rpm_ratio, rpd_ratio) * 100.
    None limits are treated as unlimited (ratio = 1.0).
    """
    # Precondition assertions
    assert rpm_remaining >= 0, f'rpm_remaining must be non-negative, got {rpm_remaining}'
    assert rpm_limit >= 0, f'rpm_limit must be non-negative, got {rpm_limit}'
    if rpd_remaining is not None:
        assert rpd_remaining >= 0, f'rpd_remaining must be non-negative when not None, got {rpd_remaining}'
    if rpd_limit is not None:
        assert rpd_limit >= 0, f'rpd_limit must be non-negative when not None, got {rpd_limit}'

    rpm_ratio = rpm_remaining / rpm_limit if rpm_limit > 0 else 1.0

    if rpd_remaining is None or rpd_limit is None or rpd_limit == 0:
        rpd_ratio = 1.0
    else:
        rpd_ratio = rpd_remaining / rpd_limit

    result = min(rpm_ratio, rpd_ratio) * 100.0
    # Postcondition assertion
    assert 0.0 <= result <= 100.0, f'computed budget score out of bounds: {result}'
    return result


def compute_health_score(
    error_count: int,
    circuit_open: bool,
    last_success_age_s: float,
) -> float:
    """Health score (0-100).

    - circuit_open → 0
    - 3+ errors → 30
    - 1-2 errors → 70
    - 0 errors → 100
    """
    # Precondition assertions
    assert error_count >= 0, f'error_count must be non-negative, got {error_count}'
    assert last_success_age_s >= 0.0, f'last_success_age_s must be non-negative, got {last_success_age_s}'
    # circuit_open is bool, no need to assert type

    if circuit_open:
        result = 0.0
    elif error_count >= 3:
        result = 30.0
    elif error_count >= 1:
        result = 70.0
    else:
        result = 100.0

    # Postcondition assertion
    assert 0.0 <= result <= 100.0, f'computed health score out of bounds: {result}'
    return result


# ===== Canonical ScoringWeights + CostGovernor (RT-018) =====


@dataclass(frozen=True)
class ScoringWeights:
    """Canonical weights for scoring dimensions.

    All weights are non-negative and sum to 1.0.
    """

    rank: float
    budget: float
    health: float
    cost: float
    latency: float
    queue_depth: float

    def __post_init__(self) -> None:
        """Validate that weights are non-negative and sum to 1.0."""
        total = self.rank + self.budget + self.health + self.cost + self.latency + self.queue_depth
        assert all(
            w >= 0 for w in (self.rank, self.budget, self.health, self.cost, self.latency, self.queue_depth)
        ), "All weights must be non-negative"
        assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"


def cost_adjusted_weights(base_weights: ScoringWeights, daily_spend: float, daily_budget: float) -> ScoringWeights:
    """Adjust weights based on daily spend vs budget.

    When over budget, increase cost weight and decrease rank weight.
    When under budget, decrease cost weight and increase rank weight.
    Other weights remain proportional.

    Args:
        base_weights: The canonical weights to adjust.
        daily_spend: Amount spent so far today.
        daily_budget: The daily budget limit.

    Returns:
        A new ScoringWeights instance with adjusted cost and rank weights.
    """
    assert isinstance(base_weights, ScoringWeights), "base_weights must be ScoringWeights instance"
    assert daily_spend >= 0, f'daily_spend must be non-negative, got {daily_spend}'
    assert daily_budget > 0, f'daily_budget must be positive, got {daily_budget}'

    # Compute ratio of spend to budget
    ratio = daily_spend / daily_budget if daily_budget > 0 else float('inf')

    # We want to adjust cost and rank weights inversely based on ratio.
    # If ratio >= 1 (over budget), we increase cost weight and decrease rank weight.
    # If ratio < 1 (under budget), we decrease cost weight and increase rank weight.
    # We'll keep the sum of cost + rank constant, and adjust proportionally.
    # Let shift = (ratio - 1) * sensitivity, clamped to [-0.5, 0.5] to avoid extreme weights.
    sensitivity = 0.5  # maximum shift of 0.5 (50% of weight)
    shift = max(-sensitivity, min(sensitivity, ratio - 1.0))

    new_cost = base_weights.cost + shift
    new_rank = base_weights.rank - shift
    # Ensure non-negative
    new_cost = max(0.0, new_cost)
    new_rank = max(0.0, new_rank)

    # Renormalize cost + rank to keep their sum constant
    cost_rank_sum = base_weights.cost + base_weights.rank
    if cost_rank_sum > 0:
        new_cost = new_cost * cost_rank_sum / (new_cost + new_rank)
        new_rank = new_rank * cost_rank_sum / (new_cost + new_rank)
    else:
        # If both were zero, keep them zero
        new_cost = 0.0
        new_rank = 0.0

    # Keep other weights the same
    return ScoringWeights(
        rank=new_rank,
        budget=base_weights.budget,
        health=base_weights.health,
        cost=new_cost,
        latency=base_weights.latency,
        queue_depth=base_weights.queue_depth,
    )


@dataclass(frozen=True)
class CostGovernorConfig:
    """Configuration for the cost governor.

    Attributes:
        daily_budget_usd: The daily budget in USD.
        warning_threshold: Fraction of budget at which to warn (0.0-1.0).
        critical_threshold: Fraction of budget at which to trigger cost governor (0.0-1.0).
        weight_shift_sensitivity: How much to shift weights when over/under budget (0.0-0.5).
    """

    daily_budget_usd: float
    warning_threshold: float = 0.75
    critical_threshold: float = 0.9
    weight_shift_sensitivity: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration."""
        assert self.daily_budget_usd > 0, f'Daily budget must be positive, got {self.daily_budget_usd}'
        assert 0.0 <= self.warning_threshold <= 1.0, f'Warning threshold must be between 0 and 1, got {self.warning_threshold}'
        assert 0.0 <= self.critical_threshold <= 1.0, f'Critical threshold must be between 0 and 1, got {self.critical_threshold}'
        assert 0.0 <= self.weight_shift_sensitivity <= 0.5, f'Weight shift sensitivity must be between 0 and 0.5, got {self.weight_shift_sensitivity}'
        assert self.warning_threshold <= self.critical_threshold, f'Warning threshold ({self.warning_threshold}) must be <= critical threshold ({self.critical_threshold})'


def cost_governor_active(config: CostGovernorConfig, daily_spend: float) -> bool:
    """Return True if the cost governor should be active (i.e., we've exceeded critical threshold).

    Args:
        config: The CostGovernorConfig.
        daily_spend: Amount spent so far today.

    Returns:
        True if daily_spend >= critical_threshold * daily_budget_usd.
    """
    assert isinstance(config, CostGovernorConfig), "config must be CostGovernorConfig instance"
    assert daily_spend >= 0, f'daily_spend must be non-negative, got {daily_spend}'

    return daily_spend >= config.critical_threshold * config.daily_budget_usd


def score_candidate(
    rank: float,
    budget_score: float,
    health_score: float,
    cost_score: float,
    latency_score: float,
    queue_depth_score: float,
    weights: ScoringWeights,
) -> float:
    """Score a candidate using the given ScoringWeights.

    All input scores should be on 0-100 scale.
    Returns a weighted sum on 0-100 scale.

    Args:
        rank: Rank score (0-100, higher is better).
        budget_score: Budget score (0-100, higher is better).
        health_score: Health score (0-100, higher is better).
        cost_score: Cost score (0-100, higher is better, i.e., lower cost).
        latency_score: Latency score (0-100, higher is better, i.e., lower latency).
        queue_depth_score: Queue depth score (0-100, higher is better, i.e., shorter queue).
        weights: The ScoringWeights to apply.

    Returns:
        Weighted composite score (0-100).
    """
    # Precondition assertions
    assert 0.0 <= rank <= 100.0, f'rank must be between 0 and 100, got {rank}'
    assert 0.0 <= budget_score <= 100.0, f'budget_score must be between 0 and 100, got {budget_score}'
    assert 0.0 <= health_score <= 100.0, f'health_score must be between 0 and 100, got {health_score}'
    assert 0.0 <= cost_score <= 100.0, f'cost_score must be between 0 and 100, got {cost_score}'
    assert 0.0 <= latency_score <= 100.0, f'latency_score must be between 0 and 100, got {latency_score}'
    assert 0.0 <= queue_depth_score <= 100.0, f'queue_depth_score must be between 0 and 100, got {queue_depth_score}'
    assert isinstance(weights, ScoringWeights), f'weights must be ScoringWeights instance, got {type(weights)}'

    result = (
        rank * weights.rank
        + budget_score * weights.budget
        + health_score * weights.health
        + cost_score * weights.cost
        + latency_score * weights.latency
        + queue_depth_score * weights.queue_depth
    )

    # Postcondition assertion
    assert 0.0 <= result <= 100.0, f'computed score out of bounds: {result}'
    return result