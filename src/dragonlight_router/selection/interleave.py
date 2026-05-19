"""Provider interleaving — prevents consecutive same-provider runs.

Preserves score ordering as much as possible while ensuring no provider
appears more than max_consecutive times in a row.
"""
from __future__ import annotations

from dragonlight_router.core.types import ModelScore


def interleave_providers(
    scored_models: list[ModelScore],
    max_consecutive: int = 2,
) -> list[ModelScore]:
    """Reorder scored models so no provider appears max_consecutive+1 times in a row.

    Preserves score ordering where possible. If only one provider exists,
    returns as-is (can't interleave).
    """
    if len(scored_models) <= 1:
        return list(scored_models)

    providers = set(m.provider for m in scored_models)
    if len(providers) <= 1:
        return list(scored_models)

    result: list[ModelScore] = []
    remaining = list(scored_models)

    while remaining:
        placed = False
        for i, candidate in enumerate(remaining):
            # Check if placing this candidate would violate the constraint
            if _can_place(result, candidate.provider, max_consecutive):
                result.append(candidate)
                remaining.pop(i)
                placed = True
                break

        if not placed:
            # Can't satisfy constraint — just append the rest
            result.extend(remaining)
            break

    return result


def _can_place(result: list[ModelScore], provider: str, max_consecutive: int) -> bool:
    """Check if adding this provider would exceed max_consecutive."""
    if not result:
        return True

    # Look at the tail of result
    consecutive = 0
    for item in reversed(result):
        if item.provider == provider:
            consecutive += 1
        else:
            break

    return consecutive < max_consecutive
