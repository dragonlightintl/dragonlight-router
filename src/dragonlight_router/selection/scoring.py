"""Composite scoring functions for model selection.

Scores combine rank (role-matrix position), budget availability,
and health state into a single comparable float.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import List, Optional, Tuple


def compute_composite_score(rank: int, budget_score: float, health_score: float) -> float:
    """Weighted composite: rank 60%, budget 25%, health 15%.

    All inputs should be on 0-100 scale. Output is 0-100.
    """
    return rank * 0.6 + budget_score * 0.25 + health_score * 0.15


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
    rpm_ratio = rpm_remaining / rpm_limit if rpm_limit > 0 else 1.0

    if rpd_remaining is None or rpd_limit is None:
        rpd_ratio = 1.0
    elif rpd_limit == 0:
        rpd_ratio = 1.0
    else:
        rpd_ratio = rpd_remaining / rpd_limit

    return min(rpm_ratio, rpd_ratio) * 100.0


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
    if circuit_open:
        return 0.0
    if error_count >= 3:
        return 30.0
    if error_count >= 1:
        return 70.0
    return 100.0
