"""Composite scoring functions for model selection.

Scores combine rank (role-matrix position), budget availability,
and health state into a single comparable float.
"""

from __future__ import annotations


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