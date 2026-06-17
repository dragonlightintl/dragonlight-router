"""Provider interleaving -- prevents consecutive same-provider runs.

Preserves score ordering as much as possible while ensuring no provider
appears more than max_consecutive times in a row.
"""
from __future__ import annotations

from collections import Counter

from dragonlight_router.core.types import ModelScore


def interleave_providers(
    scored_models: list[ModelScore],
    max_consecutive: int = 2,
) -> list[ModelScore]:
    """Reorder scored models so no provider appears max_consecutive+1 times in a row.

    Preserves score ordering where possible. If only one provider exists,
    returns as-is (can't interleave).
    """
    assert isinstance(scored_models, list), "scored_models must be a list"
    assert all(isinstance(m, ModelScore) for m in scored_models), "all elements must be ModelScore instances"
    assert isinstance(max_consecutive, int) and max_consecutive >= 0, "max_consecutive must be a non-negative integer"

    if len(scored_models) <= 1:
        return list(scored_models)

    providers = {m.provider for m in scored_models}
    if len(providers) <= 1:
        return list(scored_models)

    if not _is_constraint_satisfiable(scored_models, max_consecutive):
        return list(scored_models)

    result = _build_interleaved(scored_models, max_consecutive)

    assert len(result) == len(scored_models), "interleaved result must have same length as input"
    assert all(isinstance(m, ModelScore) for m in result), "all elements in result must be ModelScore instances"
    _verify_consecutive_constraint(result, providers, max_consecutive)

    return result


def _is_constraint_satisfiable(
    scored_models: list[ModelScore],
    max_consecutive: int,
) -> bool:
    """Check if the consecutive constraint can be satisfied for the given distribution."""
    assert len(scored_models) > 0, "scored_models must not be empty"
    assert max_consecutive > 0, "max_consecutive must be positive for satisfiability check"

    counts = Counter(m.provider for m in scored_models)
    n = len(scored_models)

    for provider, count in counts.items():
        others = n - count
        # Provider p can appear at most max_consecutive times for each "gap"
        # created by other items, plus one leading run.
        max_allowed = max_consecutive * (others + 1)
        if count > max_allowed:
            return False

    return True


def _build_interleaved(
    scored_models: list[ModelScore],
    max_consecutive: int,
) -> list[ModelScore]:
    """Greedily build interleaved list, preserving score order where possible.

    Uses a most-frequent-first strategy: when multiple candidates can be
    placed, prefer the one whose provider has the highest remaining count.
    This prevents dominant providers from getting stuck at the end.
    """
    assert len(scored_models) > 0, "scored_models must not be empty"

    result: list[ModelScore] = []
    remaining = list(scored_models)

    while remaining:
        placed = _try_place_best(result, remaining, max_consecutive)
        if not placed:
            result.extend(remaining)
            break

    assert len(result) == len(scored_models), "must preserve all models"
    return result


def _try_place_best(
    result: list[ModelScore],
    remaining: list[ModelScore],
    max_consecutive: int,
) -> bool:
    """Place the best valid candidate, preferring providers with highest remaining count."""
    assert isinstance(remaining, list), "remaining must be a list"
    assert len(remaining) > 0, "remaining must not be empty"

    provider_counts = Counter(m.provider for m in remaining)
    placeable = [
        (i, candidate) for i, candidate in enumerate(remaining)
        if _can_place(result, candidate.provider, max_consecutive)
    ]
    if not placeable:
        return False

    # Among placeable candidates, pick the one whose provider has the most remaining
    # items (break ties by original index to preserve score ordering).
    best_idx, best_candidate = max(
        placeable,
        key=lambda pair: (provider_counts[pair[1].provider], -pair[0]),
    )
    result.append(best_candidate)
    remaining.pop(best_idx)
    return True


def _verify_consecutive_constraint(
    result: list[ModelScore],
    providers: set[str],
    max_consecutive: int,
) -> None:
    """Verify no provider appears more than max_consecutive times consecutively."""
    assert len(result) > 0, "result must not be empty"

    for provider in providers:
        consecutive = 0
        for m in result:
            if m.provider == provider:
                consecutive += 1
                assert consecutive <= max_consecutive, (
                    f"provider {provider} appears more than {max_consecutive} times consecutively"
                )
            else:
                consecutive = 0


def _can_place(result: list[ModelScore], provider: str, max_consecutive: int) -> bool:
    """Check if adding this provider would exceed max_consecutive."""
    assert isinstance(result, list), "result must be a list"
    assert isinstance(provider, str), "provider must be a string"
    assert isinstance(max_consecutive, int) and max_consecutive >= 0, "max_consecutive must be a non-negative integer"

    if not result:
        return True

    consecutive = 0
    for item in reversed(result):
        if item.provider == provider:
            consecutive += 1
        else:
            break

    return consecutive < max_consecutive
