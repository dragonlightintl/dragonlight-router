"""Cost Balancing (CBR) stage for the cascade router.

Filters model candidates based on cost-effectiveness and budget constraints.
"""
from __future__ import annotations

import statistics
import structlog
from dataclasses import dataclass

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import BackendConfig, BackendCostProfile, DispatchOrder
from dragonlight_router.result import Err, Ok, Result
from dragonlight_router.selection.scoring import (
    ScoringWeightsConfig,
    cost_governor_active,
    cost_adjusted_weights,
    score_candidate,
)

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class CostFilterParams:
    """Grouped parameters for cost filtering beyond core (candidates, order, budget_tracker)."""
    daily_spend: float = 0.0
    monthly_spend: float = 0.0
    config: dict | None = None
    health_tracker: object | None = None


def filter_by_cost(
    candidates: list[BackendConfig],
    order: DispatchOrder,
    budget_tracker: BudgetTracker,
    *,
    daily_spend: float = 0.0,
    monthly_spend: float = 0.0,
    config: dict | None = None,
    health_tracker: object | None = None,
) -> list[BackendConfig]:
    """Filter candidates based on cost effectiveness given available budget.

    Implements hard budget filtering and scoring with cost governor support.
    Returns candidates sorted by score (best first), keeping at least one.

    Args:
        candidates: List of backend configurations to evaluate.
        order: The dispatch order with context for scoring.
        budget_tracker: Tracks per-provider budget availability.
        daily_spend: Total daily spend in USD across all providers (for cost governor).
        monthly_spend: Total monthly spend in USD across all providers (for cost governor).
        config: Router configuration dictionary with cost governor thresholds.
        health_tracker: Optional health tracker for candidate scoring.
    """
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(
        isinstance(c, BackendConfig) for c in candidates
    ), "all candidates must be BackendConfig instances"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(budget_tracker, BudgetTracker), "budget_tracker must be BudgetTracker instance"

    params = CostFilterParams(
        daily_spend=daily_spend,
        monthly_spend=monthly_spend,
        config=config,
        health_tracker=health_tracker,
    )

    return _filter_by_cost_impl(candidates, order, budget_tracker, params)


def _filter_by_cost_impl(
    candidates: list[BackendConfig],
    order: DispatchOrder,
    budget_tracker: BudgetTracker,
    params: CostFilterParams,
) -> list[BackendConfig]:
    """Core cost filtering: hard budget filter then score-based ranking."""
    assert isinstance(params, CostFilterParams), "params must be CostFilterParams"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder"

    logger.debug(
        "filtering candidates by cost",
        candidate_count=len(candidates),
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
        daily_spend=params.daily_spend,
        monthly_spend=params.monthly_spend,
    )

    if not candidates:
        logger.debug("no candidates, returning as-is")
        return candidates

    filtered = _hard_budget_filter(candidates, budget_tracker)
    if not filtered:
        return filtered

    return _score_and_rank(filtered, order, budget_tracker, params)


def _hard_budget_filter(
    candidates: list[BackendConfig],
    budget_tracker: BudgetTracker,
) -> list[BackendConfig]:
    """Exclude providers with spent_usd >= budget_usd (budget score of 0.0)."""
    assert len(candidates) > 0, "candidates must not be empty"

    filtered: list[BackendConfig] = []
    budget_scores: dict[str, float] = {}

    for candidate in candidates:
        provider = candidate.provider
        budget_result = budget_tracker.score(provider)
        budget_score = budget_result.value if hasattr(budget_result, 'value') else 50.0
        budget_scores[provider] = budget_score
        if budget_score > 0.0:
            filtered.append(candidate)

    logger.debug(
        "budget filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered),
        budget_scores=budget_scores,
    )

    assert len(filtered) <= len(candidates), "filtered count must not exceed original"
    return filtered


def _resolve_weights(params: CostFilterParams) -> ScoringWeightsConfig:
    """Determine scoring weights, activating cost governor if thresholds are met."""
    assert isinstance(params, CostFilterParams), "params must be CostFilterParams"

    governor_config = params.config if params.config is not None else {}
    weights = ScoringWeightsConfig()

    if cost_governor_active(params.daily_spend, params.monthly_spend, governor_config):
        weights = cost_adjusted_weights(weights)
        logger.debug(
            "cost governor active",
            daily_spend=params.daily_spend,
            monthly_spend=params.monthly_spend,
            adjusted_weights=weights.__dict__,
        )

    assert isinstance(weights, ScoringWeightsConfig), "weights must be ScoringWeightsConfig"
    return weights


def _score_and_rank(
    candidates: list[BackendConfig],
    order: DispatchOrder,
    budget_tracker: BudgetTracker,
    params: CostFilterParams,
) -> list[BackendConfig]:
    """Score each candidate and return sorted by score (best first)."""
    assert len(candidates) > 0, "must have candidates to score"

    weights = _resolve_weights(params)

    scored: list[tuple[float, BackendConfig]] = []
    for candidate in candidates:
        score = score_candidate(
            config=candidate,
            order=order,
            weights=weights,
            budget_tracker=budget_tracker,
            health_tracker=params.health_tracker,
        )
        scored.append((score, candidate))

    scored.sort(key=lambda x: x[0], reverse=True)
    result = [candidate for _score, candidate in scored]

    logger.debug(
        "cost scoring complete",
        candidate_count=len(result),
        weights_used=weights.__dict__,
        governor_active=weights.cost > 0.35,
    )

    assert len(result) == len(candidates), "scoring must not lose candidates"
    return result


def filter_by_cost_efficiency(
    candidates: list[BackendConfig],
    budget_scores: dict[str, float],
    order: DispatchOrder,
) -> list[BackendConfig]:
    """Filter candidates by cost-efficiency relative to budget.

    Efficiency = budget_score / avg_cost_per_mtok.

    Keeps candidates whose efficiency is at or above the median efficiency
    of the pool.  If no budget data is available, returns candidates as-is.

    Args:
        candidates: List of backend configurations.
        budget_scores: Mapping of provider name -> budget score (0-100).
        order: Dispatch order (reserved for future use).

    Returns:
        Filtered list of candidates.
    """
    assert isinstance(candidates, list), "candidates must be a list"
    assert isinstance(budget_scores, dict), "budget_scores must be a dict"
    assert isinstance(order, DispatchOrder), "order must be a DispatchOrder"

    if not candidates or not budget_scores:
        return [] if not candidates else candidates

    efficiencies, candidate_efficiencies = _compute_efficiencies(candidates, budget_scores)

    if not efficiencies:
        return candidates

    return _apply_median_filter(candidates, efficiencies, candidate_efficiencies)


def _compute_efficiencies(
    candidates: list[BackendConfig],
    budget_scores: dict[str, float],
) -> tuple[list[float], list[tuple[float | None, BackendConfig]]]:
    """Compute cost-efficiency for each candidate that has budget data."""
    assert isinstance(candidates, list), "candidates must be a list"

    efficiencies: list[float] = []
    candidate_efficiencies: list[tuple[float | None, BackendConfig]] = []

    for candidate in candidates:
        provider_score = budget_scores.get(candidate.provider)
        if provider_score is None:
            candidate_efficiencies.append((None, candidate))
            continue

        eff = _single_efficiency(candidate, provider_score)
        efficiencies.append(eff)
        candidate_efficiencies.append((eff, candidate))

    assert len(candidate_efficiencies) == len(candidates), "must have efficiency entry for each candidate"
    return efficiencies, candidate_efficiencies


def _single_efficiency(candidate: BackendConfig, provider_score: float) -> float:
    """Compute efficiency for a single candidate: budget_score / avg_cost."""
    assert isinstance(candidate, BackendConfig), "candidate must be BackendConfig"
    assert provider_score >= 0, "provider_score must be non-negative"

    avg_cost = (candidate.cost.input_per_mtok + candidate.cost.output_per_mtok) / 2.0
    if avg_cost <= 0.0:
        return float("inf")
    return provider_score / avg_cost


def _apply_median_filter(
    candidates: list[BackendConfig],
    efficiencies: list[float],
    candidate_efficiencies: list[tuple[float | None, BackendConfig]],
) -> list[BackendConfig]:
    """Retain candidates whose efficiency is at or above the median."""
    assert len(efficiencies) > 0, "must have at least one efficiency value"

    median_eff = statistics.median(efficiencies)

    logger.debug(
        "cost efficiency filtering",
        candidate_count=len(candidates),
        median_efficiency=median_eff,
    )

    filtered = [
        candidate
        for eff, candidate in candidate_efficiencies
        if eff is None or eff >= median_eff
    ]

    logger.debug(
        "cost efficiency filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered),
    )

    assert len(filtered) <= len(candidates), "filtered count must not exceed original"
    return filtered


def filter_by_absolute_cost(
    candidates: list[BackendConfig],
    max_cost_per_mtok: float,
) -> list[BackendConfig]:
    """Filter candidates that exceed an absolute cost threshold."""
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(isinstance(c, BackendConfig) for c in candidates), "all candidates must be BackendConfig instances"
    assert isinstance(max_cost_per_mtok, (int, float)) and max_cost_per_mtok >= 0, \
        "max_cost_per_mtok must be a non-negative number"

    logger.debug("filtering by absolute cost", candidate_count=len(candidates), max_cost_per_mtok=max_cost_per_mtok)

    filtered: list[BackendConfig] = []
    for candidate in candidates:
        avg_cost = (candidate.cost.input_per_mtok + candidate.cost.output_per_mtok) / 2.0
        if avg_cost <= max_cost_per_mtok:
            filtered.append(candidate)

    logger.debug("absolute cost filtering complete", original_count=len(candidates), filtered_count=len(filtered))
    assert len(filtered) <= len(candidates), "filtered count must not exceed original"
    return filtered
