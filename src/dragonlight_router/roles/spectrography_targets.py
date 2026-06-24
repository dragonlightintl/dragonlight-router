"""Spectrography target filtering and prioritization.

Provides :func:`get_spectrography_targets` which filters and prioritizes
models that still need empirical profiling (heuristic-only lifecycle state).

Reads ``matrix_lifecycle.json`` for source metadata and the role matrix for
rank data, then returns a sorted, filtered list of model_ids ready for
the spectrography runner's ``--heuristic-only`` mode.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

_LIFECYCLE_FILENAME = "matrix_lifecycle.json"
_MATRIX_FILENAME = "model_role_matrix.json"


def _load_lifecycle(state_dir: Path) -> dict[str, Any]:
    """Load matrix_lifecycle.json.  Returns empty models dict when missing."""
    assert isinstance(state_dir, Path), "state_dir must be a Path"

    path = state_dir / _LIFECYCLE_FILENAME
    if not path.exists():
        logger.info("lifecycle_not_found", path=str(path))
        return {"models": {}}

    try:
        data: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("lifecycle_load_failed", path=str(path), error=str(exc))
        return {"models": {}}

    if "models" not in data:
        data["models"] = {}

    return data


def _load_matrix_models(state_dir: Path) -> dict[str, int]:
    """Return {model_id: max_rank_across_all_roles} from the role matrix.

    Returns an empty dict when the file is missing or unreadable.
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"

    path = state_dir / _MATRIX_FILENAME
    if not path.exists():
        logger.info("matrix_not_found", path=str(path))
        return {}

    try:
        raw: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("matrix_load_failed", path=str(path), error=str(exc))
        return {}

    max_ranks: dict[str, int] = {}
    roles_data = raw.get("roles", raw)

    if not isinstance(roles_data, dict):
        return {}

    for key, entries in roles_data.items():
        if key in ("version", "default_rank"):
            continue
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                mid = entry.get("model_id")
                rank = entry.get("rank", 0)
                if mid and isinstance(rank, int):
                    max_ranks[mid] = max(max_ranks.get(mid, 0), rank)
        elif isinstance(entries, dict):
            for mid, rank in entries.items():
                if isinstance(rank, int):
                    max_ranks[mid] = max(max_ranks.get(mid, 0), rank)

    return max_ranks


def get_spectrography_targets(
    state_dir: Path,
    *,
    limit: int | None = None,
    min_rank: int = 30,
) -> list[str]:
    """Return model_ids needing spectrography, filtered and prioritized.

    Reads lifecycle state to find models with ``source == "heuristic"``.
    Reads the role matrix to get each model's highest rank across any role.
    Filters out models whose max rank is below *min_rank* (not worth profiling).
    Sorts by highest rank descending (profile the most-used models first).
    Applies *limit* if specified.

    Returns an empty list when ``matrix_lifecycle.json`` does not exist.
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"
    assert limit is None or (isinstance(limit, int) and limit > 0), (
        "limit must be a positive int or None"
    )
    assert isinstance(min_rank, int) and min_rank >= 0, "min_rank must be a non-negative int"

    lifecycle = _load_lifecycle(state_dir)
    models_meta: dict[str, Any] = lifecycle.get("models", {})

    heuristic_ids = {
        mid
        for mid, meta in models_meta.items()
        if isinstance(meta, dict) and meta.get("source") == "heuristic"
    }

    if not heuristic_ids:
        logger.info("no_heuristic_models_found")
        return []

    max_ranks = _load_matrix_models(state_dir)

    # Filter by min_rank
    qualified = [mid for mid in heuristic_ids if max_ranks.get(mid, 0) >= min_rank]

    # Sort by max rank descending, then model_id for deterministic tie-breaking
    qualified.sort(key=lambda mid: (-max_ranks.get(mid, 0), mid))

    if limit is not None:
        qualified = qualified[:limit]

    logger.info(
        "spectrography_targets_resolved",
        heuristic_total=len(heuristic_ids),
        after_min_rank_filter=len(qualified),
        min_rank=min_rank,
        limit=limit,
    )
    return qualified
