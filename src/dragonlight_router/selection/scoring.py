"""Composite scoring functions for model selection.

Scores combine rank (role-matrix position), budget availability,
and health state into a single comparable float.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Optional, Tuple

from dragonlight_router.core.types import BackendConfig, DispatchOrder


def compute_composite_score(rank: int, budget_score: float, health_score: float) -> float:
    """Weighted composite: rank 60%, budget 25%, health 15%.

    All inputs should be on 0-100 scale. Output is 0-100.
    """
    # Precondition assertions
    assert 0 <= rank <= 100, f'rank must be between 0 and 100, got {rank}'
    assert 0 <= budget_score <= 100, f'budget_score must be between 0 and 100, got {budget_score}'
    assert 0 <= health_score <= 100, f'health_score must be between 0 and 100, got {health_score}'

    result = rank * 0.6 + budget_score * 0.25 + health_score * 0.15

    # Postcondition assertion
    assert 0 <= result <= 100, f'computed score {result} must be between 0 and 100'

    return result


@unique
class ScoringWeights(Enum):
    """Canonical scoring weights for the dispatch path (MBR→CBR→LBR cascade).

    Default values per canonical spec:
    - cost: 0.35
    - latency: 0.25
    - priority: 0.20
    - queue: 0.10
    - health: 0.10
    """
    COST = 0.35
    LATENCY = 0.25
    PRIORITY = 0.20
    QUEUE = 0.10
    HEALTH = 0.10

    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        # For Enum, we need to check the sum of all values
        total = sum(member.value for member in ScoringWeights)
        assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"


@dataclass(frozen=True)
class ScoringWeightsConfig:
    """Configuration object for scoring weights."""
    cost: float = 0.35
    latency: float = 0.25
    priority: float = 0.20
    queue: float = 0.10
    health: float = 0.10

    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = self.cost + self.latency + self.priority + self.queue + self.health
        assert abs(total - 1.0) < 1e-9, f"Weights must sum to 1.0, got {total}"


def normalize_rank(rank: int) -> float:
    """Normalize rank to [0.0, 1.0] where 1.0 is best rank.
    
    Args:
        rank: Rank position (1 = best, higher numbers = worse)
        
    Returns:
        Normalized score in [0.0, 1.0]
    """
    assert rank >= 1, f"Rank must be >= 1, got {rank}"
    # Convert to 0-100 scale where 1 = 100, then normalize to 0-1
    # Using exponential decay: better ranks get exponentially higher scores
    normalized = max(0.0, min(1.0, 2.0 ** (1 - rank / 10.0)))
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
    # Cap at 100 for normalization
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
    # Invert so lower queue depth = higher score
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


def score_candidate(
    config: BackendConfig,
    order: DispatchOrder,
    weights: ScoringWeightsConfig,
    budget_tracker,
    health_tracker,
) -> float:
    """Score a single candidate using canonical ScoringWeights.
    
    Args:
        config: Backend configuration to score
        order: Dispatch order for context
        weights: Scoring weights to apply
        budget_tracker: Budget tracker for budget/latency scores
        health_tracker: Health tracker for health/priority scores
        
    Returns:
        Composite score in [0.0, 1.0]
    """
    # Get raw scores from trackers
    budget_result = budget_tracker.score(config.provider)
    budget_score = budget_result.value if hasattr(budget_result, 'value') else 50.0
    
    health_result = health_tracker.score(config.model)
    health_score = health_result.value if hasattr(health_result, 'value') else 50.0
    
    # For latency, we'll use health tracker's latency EMA or approximate from health
    # For priority, we'll use the config's priority field
    # For queue depth, we'll approximate from budget tracker or use a default
    
    # Normalize each dimension to [0.0, 1.0]
    # Rank: we'll use a simple heuristic based on cost (lower cost = better rank)
    # In a full implementation, this would come from role matrix
    avg_cost = (config.cost.input_per_mtok + config.cost.output_per_mtok) / 2.0
    # Lower cost = better rank (rank 1 = best)
    rank_score = 100.0 / (avg_cost + 1.0) if avg_cost >= 0 else 50.0
    rank_score = min(rank_score, 100.0)  # Cap at 100
    
    # Latency: approximate from health or use a default
    latency_score = health_score  # Simplified - in reality would track latency separately
    
    # Priority: from config
    priority_score = config.priority
    
    # Queue depth: approximate from budget utilization (lower utilization = shorter queue)
    # Higher budget score = lower utilization = shorter queue = better score
    queue_score = budget_score  # Simplified
    
    # Health: from health tracker
    
    # Normalize all scores
    norm_rank = normalize_rank(int(rank_score)) if rank_score >= 1 else 0.5
    norm_budget = normalize_budget_score(budget_score)
    norm_latency = normalize_latency_score(latency_score)
    norm_priority = normalize_priority_score(int(priority_score))
    norm_queue = normalize_queue_score(int(100 - budget_score))  # Invert for queue depth
    norm_health = normalize_health_score(health_score)
    
    # Apply weights
    composite = (
        norm_rank * weights.cost +
        norm_latency * weights.latency +
        norm_priority * weights.priority +
        norm_queue * weights.queue +
        norm_health * weights.health
    )
    
    # Ensure result is in [0.0, 1.0]
    assert 0.0 <= composite <= 1.0, f"Composite score out of bounds: {composite}"
    
    return composite


def cost_governor_active(
    daily_spend: float,
    monthly_spend: float,
    config: dict,
) -> bool:
    """Check if cost governor should be active.
    
    Args:
        daily_spend: Current daily spend in USD
        monthly_spend: Current monthly spend in USD
        config: Router configuration containing thresholds
        
    Returns:
        True if cost governor should override weights
    """
    daily_threshold = config.get('cost_down_threshold_daily', 100.0)
    monthly_threshold = config.get('cost_down_threshold_monthly', 1000.0)
    
    return daily_spend >= daily_threshold or monthly_spend >= monthly_threshold


def cost_adjusted_weights(base_weights: ScoringWeightsConfig) -> ScoringWeightsConfig:
    """Adjust weights when cost governor is active.
    
    Shifts to: cost=0.70, latency=0.10, priority=0.10, queue=0.05, health=0.05
    
    Args:
        base_weights: Base scoring weights
        
    Returns:
        Adjusted scoring weights
    """
    return ScoringWeightsConfig(
        cost=0.70,
        latency=0.10,
        priority=0.10,
        queue=0.05,
        health=0.05
    )