"""Matrix lifecycle management module.

Tracks per-model metadata across catalog refreshes: detecting new models,
auto-seeding them into the matrix, decaying ranks for deprecated models, and
identifying which models still need empirical spectrography profiling.

Lifecycle state is persisted in ``{state_dir}/matrix_lifecycle.json``.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import structlog

from dragonlight_router.roles.auto_populate import classify_model, load_catalog

logger = structlog.get_logger()

# All roles the router understands — kept in sync with auto_populate._ROLES.
_ROLES: tuple[str, ...] = ("coding", "testing", "review", "spec", "reasoning")

_LIFECYCLE_FILENAME = "matrix_lifecycle.json"
_MATRIX_FILENAME = "model_role_matrix.json"


# ---------------------------------------------------------------------------
# Return-type dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CatalogDiff:
    """Result of comparing the catalog against the current matrix."""

    new_models: list[str]
    missing_models: list[str]
    unchanged: int


@dataclass(frozen=True)
class SeedResult:
    """Result of auto-seeding new catalog models into the matrix."""

    new_seeded: int
    missing_detected: int
    total_in_matrix: int


@dataclass(frozen=True)
class DecayResult:
    """Result of decaying deprecated-model ranks."""

    decayed: int
    removed: int
    remaining: int


# ---------------------------------------------------------------------------
# Lifecycle state — load / save
# ---------------------------------------------------------------------------


def load_lifecycle_state(state_dir: Path) -> dict[str, Any]:
    """Load ``matrix_lifecycle.json`` from *state_dir*, or create an empty state.

    Returns a dict with a ``"models"`` key mapping model_ids to their metadata.
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"

    lifecycle_path = state_dir / _LIFECYCLE_FILENAME
    if not lifecycle_path.exists():
        logger.info("lifecycle_state_not_found_creating_empty", path=str(lifecycle_path))
        return {"models": {}}

    try:
        text = lifecycle_path.read_text()
        data: dict[str, Any] = json.loads(text)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "lifecycle_state_load_failed",
            path=str(lifecycle_path),
            error=str(exc),
        )
        return {"models": {}}

    if "models" not in data:
        data["models"] = {}

    return data


def save_lifecycle_state(state_dir: Path, state: dict[str, Any]) -> None:
    """Persist *state* to ``{state_dir}/matrix_lifecycle.json`` atomically."""
    assert isinstance(state_dir, Path), "state_dir must be a Path"
    assert isinstance(state, dict), "state must be a dict"

    state_dir.mkdir(parents=True, exist_ok=True)
    lifecycle_path = state_dir / _LIFECYCLE_FILENAME

    fd, tmp_path = tempfile.mkstemp(
        dir=str(state_dir),
        prefix=".lifecycle_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(state, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, str(lifecycle_path))
    except OSError:
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise

    logger.info(
        "lifecycle_state_saved",
        path=str(lifecycle_path),
        models=len(state.get("models", {})),
    )


# ---------------------------------------------------------------------------
# Matrix helpers — read / write
# ---------------------------------------------------------------------------


def _load_matrix_raw(state_dir: Path) -> dict[str, Any]:
    """Read ``model_role_matrix.json`` and return the parsed dict.

    Returns ``{"version": 1, "default_rank": 20, "roles": {}}`` when the file
    is missing or unreadable.
    """
    matrix_path = state_dir / _MATRIX_FILENAME
    if not matrix_path.exists():
        return {"version": 1, "default_rank": 20, "roles": {role: [] for role in _ROLES}}

    try:
        return json.loads(matrix_path.read_text())  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("matrix_load_failed", path=str(matrix_path), error=str(exc))
        return {"version": 1, "default_rank": 20, "roles": {role: [] for role in _ROLES}}


def _save_matrix_raw(state_dir: Path, matrix: dict[str, Any]) -> None:
    """Write *matrix* to ``model_role_matrix.json`` atomically."""
    state_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = state_dir / _MATRIX_FILENAME

    fd, tmp_path = tempfile.mkstemp(
        dir=str(state_dir),
        prefix=".matrix_lifecycle_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(matrix, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.rename(tmp_path, str(matrix_path))
    except OSError:
        import contextlib

        with contextlib.suppress(OSError):
            os.unlink(tmp_path)
        raise


def _get_all_matrix_model_ids(matrix: dict[str, Any]) -> set[str]:
    """Return every model_id that appears in any role of *matrix*."""
    ids: set[str] = set()
    roles = matrix.get("roles", {})
    for entries in roles.values():
        if isinstance(entries, list):
            for entry in entries:
                ids.add(entry["model_id"])
        elif isinstance(entries, dict):
            ids.update(entries.keys())
    return ids


def _get_all_catalog_model_ids(state_dir: Path) -> set[str]:
    """Return every classifiable model_id from the provider catalog."""
    catalog = load_catalog(state_dir)
    ids: set[str] = set()
    for provider_models in catalog.values():
        for entry in provider_models:
            model_id = entry["model_id"]
            # Only track models that would be classifiable (non-excluded)
            if classify_model(model_id):
                ids.add(model_id)
    return ids


# ---------------------------------------------------------------------------
# 1. detect_catalog_changes
# ---------------------------------------------------------------------------


def detect_catalog_changes(state_dir: Path) -> CatalogDiff:
    """Compare the current catalog against the role matrix.

    Returns a :class:`CatalogDiff` with:
    - ``new_models``: in catalog but not yet in matrix
    - ``missing_models``: in matrix but absent from catalog
    - ``unchanged``: count of models present in both
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"

    catalog_ids = _get_all_catalog_model_ids(state_dir)
    matrix = _load_matrix_raw(state_dir)
    matrix_ids = _get_all_matrix_model_ids(matrix)

    new_models = sorted(catalog_ids - matrix_ids)
    missing_models = sorted(matrix_ids - catalog_ids)
    unchanged = len(catalog_ids & matrix_ids)

    logger.info(
        "catalog_diff",
        new=len(new_models),
        missing=len(missing_models),
        unchanged=unchanged,
    )
    return CatalogDiff(
        new_models=new_models,
        missing_models=missing_models,
        unchanged=unchanged,
    )


# ---------------------------------------------------------------------------
# 2. auto_seed_new_models
# ---------------------------------------------------------------------------


def auto_seed_new_models(state_dir: Path) -> SeedResult:
    """Detect catalog changes and seed new models into the matrix.

    Steps:
    1. Call :func:`detect_catalog_changes`.
    2. For each new model: classify it and add it to all roles in the matrix.
    3. For each missing model: increment ``consecutive_misses`` in lifecycle state.
       If a model reappears (consecutive_misses > 0 and back in catalog), reset.
    4. Record new models in lifecycle state as ``source: "heuristic"``.
    5. Save updated matrix and lifecycle state.

    Returns a :class:`SeedResult`.
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"

    today = date.today().isoformat()
    diff = detect_catalog_changes(state_dir)

    matrix = _load_matrix_raw(state_dir)
    lifecycle = load_lifecycle_state(state_dir)
    models_meta: dict[str, Any] = lifecycle.setdefault("models", {})

    # ── Seed new models ────────────────────────────────────────────────────────
    seeded = 0
    for model_id in diff.new_models:
        role_ranks = classify_model(model_id)
        if not role_ranks:
            # classify_model returned {} — excluded model; skip
            continue

        # Add to each role in matrix
        roles = matrix.setdefault("roles", {})
        for role in _ROLES:
            role_list = roles.setdefault(role, [])
            rank = role_ranks.get(role, 20)
            role_list.append({"model_id": model_id, "rank": rank})

        # Record in lifecycle state
        if model_id not in models_meta:
            models_meta[model_id] = {
                "source": "heuristic",
                "first_seen": today,
                "last_catalog_hit": today,
                "consecutive_misses": 0,
                "spectrography_run_id": None,
            }
        else:
            # Model was known but is now new to the matrix — reset misses
            models_meta[model_id]["last_catalog_hit"] = today
            models_meta[model_id]["consecutive_misses"] = 0

        seeded += 1

    # Sort each role by rank descending after inserting
    for role in _ROLES:
        role_list = matrix.get("roles", {}).get(role, [])
        role_list.sort(key=lambda e: e["rank"], reverse=True)

    # ── Track missing models ───────────────────────────────────────────────────
    for model_id in diff.missing_models:
        if model_id not in models_meta:
            models_meta[model_id] = {
                "source": "heuristic",
                "first_seen": today,
                "last_catalog_hit": today,
                "consecutive_misses": 1,
                "spectrography_run_id": None,
            }
        else:
            models_meta[model_id]["consecutive_misses"] = (
                models_meta[model_id].get("consecutive_misses", 0) + 1
            )

    # Update last_catalog_hit for models still present in the catalog
    catalog_ids = _get_all_catalog_model_ids(state_dir)
    for model_id in catalog_ids:
        if model_id in models_meta:
            meta = models_meta[model_id]
            if meta.get("consecutive_misses", 0) > 0:
                # Model reappeared after missing — reset
                meta["consecutive_misses"] = 0
            meta["last_catalog_hit"] = today

    # Persist
    _save_matrix_raw(state_dir, matrix)
    save_lifecycle_state(state_dir, lifecycle)

    total_in_matrix = len(_get_all_matrix_model_ids(matrix))

    logger.info(
        "auto_seed_complete",
        new_seeded=seeded,
        missing_detected=len(diff.missing_models),
        total_in_matrix=total_in_matrix,
    )
    return SeedResult(
        new_seeded=seeded,
        missing_detected=len(diff.missing_models),
        total_in_matrix=total_in_matrix,
    )


# ---------------------------------------------------------------------------
# 3. decay_deprecated_models
# ---------------------------------------------------------------------------


def decay_deprecated_models(
    state_dir: Path,
    *,
    max_misses: int = 3,
    decay_rate: float = 0.5,
) -> DecayResult:
    """Decay or remove models that have been absent from the catalog.

    For models with ``consecutive_misses >= max_misses``:
    - Multiply their rank by *decay_rate* in all roles.
    - If rank falls below 10 after decay, remove the model from the matrix.

    Returns a :class:`DecayResult`.
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"
    assert max_misses >= 1, "max_misses must be >= 1"
    assert 0.0 < decay_rate <= 1.0, "decay_rate must be in (0.0, 1.0]"

    lifecycle = load_lifecycle_state(state_dir)
    models_meta: dict[str, Any] = lifecycle.get("models", {})

    deprecated = {
        mid for mid, meta in models_meta.items() if meta.get("consecutive_misses", 0) >= max_misses
    }

    if not deprecated:
        matrix = _load_matrix_raw(state_dir)
        total_in_matrix = len(_get_all_matrix_model_ids(matrix))
        logger.info("decay_no_deprecated_models", remaining=total_in_matrix)
        return DecayResult(decayed=0, removed=0, remaining=total_in_matrix)

    matrix = _load_matrix_raw(state_dir)
    roles = matrix.get("roles", {})

    decayed_ids: set[str] = set()
    removed_ids: set[str] = set()

    for role in list(roles.keys()):
        entries = roles[role]
        if not isinstance(entries, list):
            continue

        kept: list[dict[str, Any]] = []
        for entry in entries:
            model_id = entry["model_id"]
            if model_id not in deprecated:
                kept.append(entry)
                continue

            new_rank = int(entry["rank"] * decay_rate)
            if new_rank < 10:
                removed_ids.add(model_id)
                logger.info(
                    "model_removed_below_threshold",
                    model_id=model_id,
                    old_rank=entry["rank"],
                    new_rank=new_rank,
                    role=role,
                )
            else:
                entry = dict(entry)  # copy to avoid mutating in-place unexpectedly
                entry["rank"] = new_rank
                kept.append(entry)
                decayed_ids.add(model_id)

        kept.sort(key=lambda e: e["rank"], reverse=True)
        roles[role] = kept

    _save_matrix_raw(state_dir, matrix)

    total_in_matrix = len(_get_all_matrix_model_ids(matrix))

    # Track decayed (not removed) as decayed count
    truly_decayed = decayed_ids - removed_ids

    logger.info(
        "decay_complete",
        decayed=len(truly_decayed),
        removed=len(removed_ids),
        remaining=total_in_matrix,
    )
    return DecayResult(
        decayed=len(truly_decayed),
        removed=len(removed_ids),
        remaining=total_in_matrix,
    )


# ---------------------------------------------------------------------------
# 4. get_models_needing_spectrography
# ---------------------------------------------------------------------------


def get_models_needing_spectrography(state_dir: Path) -> list[str]:
    """Return model_ids with ``source == "heuristic"``, sorted by rank descending.

    These are models that have been auto-seeded from heuristics but have not
    yet been empirically profiled via spectrography.
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"

    lifecycle = load_lifecycle_state(state_dir)
    models_meta: dict[str, Any] = lifecycle.get("models", {})

    heuristic_ids = {mid for mid, meta in models_meta.items() if meta.get("source") == "heuristic"}

    if not heuristic_ids:
        return []

    # Compute max rank across all roles for each model to use as sort key
    matrix = _load_matrix_raw(state_dir)
    roles = matrix.get("roles", {})

    model_max_rank: dict[str, int] = dict.fromkeys(heuristic_ids, 0)
    for entries in roles.values():
        if not isinstance(entries, list):
            continue
        for entry in entries:
            mid = entry["model_id"]
            if mid in model_max_rank:
                model_max_rank[mid] = max(model_max_rank[mid], entry["rank"])

    sorted_ids = sorted(heuristic_ids, key=lambda mid: -model_max_rank.get(mid, 0))

    logger.info(
        "models_needing_spectrography",
        count=len(sorted_ids),
    )
    return sorted_ids


# ---------------------------------------------------------------------------
# 5. mark_spectrography_complete
# ---------------------------------------------------------------------------


def mark_spectrography_complete(
    state_dir: Path,
    model_ids: list[str],
    run_id: str,
) -> None:
    """Update lifecycle state to mark models as empirically profiled.

    Sets ``source`` to ``"empirical"`` and records *run_id* for each model_id
    in *model_ids*.
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"
    assert isinstance(model_ids, list), "model_ids must be a list"
    assert isinstance(run_id, str) and run_id, "run_id must be a non-empty string"

    today = date.today().isoformat()
    lifecycle = load_lifecycle_state(state_dir)
    models_meta: dict[str, Any] = lifecycle.setdefault("models", {})

    for model_id in model_ids:
        if model_id not in models_meta:
            models_meta[model_id] = {
                "source": "empirical",
                "first_seen": today,
                "last_catalog_hit": today,
                "consecutive_misses": 0,
                "spectrography_run_id": run_id,
            }
        else:
            models_meta[model_id]["source"] = "empirical"
            models_meta[model_id]["spectrography_run_id"] = run_id

    save_lifecycle_state(state_dir, lifecycle)

    logger.info(
        "spectrography_marked_complete",
        run_id=run_id,
        model_count=len(model_ids),
    )


# ---------------------------------------------------------------------------
# 6. mark_models_unreachable — inference-level health pruning
# ---------------------------------------------------------------------------


def mark_models_unreachable(
    state_dir: Path,
    model_ids: list[str],
) -> int:
    """Remove models that are listed in the catalog but fail at inference time.

    Unlike catalog-miss decay (which tracks models disappearing from the
    provider's model listing), this handles models that are *listed* but
    return 404/400/empty on ``/v1/chat/completions``.

    Removes the models from all roles in the matrix and sets their lifecycle
    source to ``"unreachable"``.  Returns the number of models removed.
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"
    assert isinstance(model_ids, list), "model_ids must be a list"

    if not model_ids:
        return 0

    dead_set = set(model_ids)
    matrix = _load_matrix_raw(state_dir)
    roles = matrix.get("roles", {})

    removed = 0
    for role in list(roles.keys()):
        entries = roles[role]
        if not isinstance(entries, list):
            continue
        before = len(entries)
        roles[role] = [e for e in entries if e["model_id"] not in dead_set]
        removed += before - len(roles[role])

    _save_matrix_raw(state_dir, matrix)

    lifecycle = load_lifecycle_state(state_dir)
    models_meta: dict[str, Any] = lifecycle.setdefault("models", {})
    today = date.today().isoformat()
    for model_id in model_ids:
        if model_id in models_meta:
            models_meta[model_id]["source"] = "unreachable"
        else:
            models_meta[model_id] = {
                "source": "unreachable",
                "first_seen": today,
                "last_catalog_hit": today,
                "consecutive_misses": 0,
                "spectrography_run_id": None,
            }
    save_lifecycle_state(state_dir, lifecycle)

    logger.info(
        "models_marked_unreachable",
        model_count=len(model_ids),
        matrix_entries_removed=removed,
    )
    return removed
