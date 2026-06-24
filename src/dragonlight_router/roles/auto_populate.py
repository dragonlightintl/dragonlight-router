"""Catalog-to-matrix auto-seeder.

Reads provider_catalog.json from the router state directory and generates
a model_role_matrix.json with intelligent role assignments and rank
estimates based on model name heuristics.

Heuristic ranks are priors that spectrography will later refine. When
merging with an existing matrix, operator-curated ranks are always
preserved.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# All roles the router understands.
_ROLES: tuple[str, ...] = ("coding", "testing", "review", "spec", "reasoning")

# Default rank for models without strong specialization signals.
_DEFAULT_RANK = 20

# ─── Exclusion patterns ────────────────────────────────────────────────────────
# Models matching these substrings are excluded entirely from the matrix.
# Order matters: checked before capability patterns.
_EXCLUDE_PATTERNS: tuple[str, ...] = (
    "embed",
    "whisper",
    "tts",
    "-tts",
    "speech",
    "audio",
    "imagen",
    "veo",
    "lyria",
    "clip",
    "nvclip",
    "diffusion",
    "deplot",
    "vila",
    "neva",
    "kosmos",
    "aqa",
    "ocr",
    "translate",
    "ising-calibration",
    "gliner",
    "parse",
    "reward",
    "live",
    "realtime",
    "robotics",
    "antigravity",
    "deep-research",
    "vibe-cli",
)

# Guard / moderation / safety-classifier models — excluded.
_GUARD_PATTERNS: tuple[str, ...] = (
    "guard",
    "safeguard",
    "moderation",
    "content-safety",
    "safety-guard",
    "nemoguard",
    "prompt-guard",
)

# ─── Rank signal patterns ──────────────────────────────────────────────────────
# Each entry: (substring_pattern, {role: rank_boost})
# Ranks represent absolute estimates (not additive) — the highest-matched
# pattern wins per role.  Patterns are checked in order, first match wins.

_HIGH_CODING: tuple[str, ...] = (
    "coder",
    "codestral",
    "starcoder",
    "codellama",
    "codegemma",
    "granite-34b-code",
    "granite-8b-code",
    "devstral",
    "pareto-code",
    "kimi-k2",
    "qwen3-coder",
    "deepseek-coder",
    "north-mini-code",
    "nv-embedcode",  # only the embedding part excluded; name still has "code"
)

_HIGH_REASONING: tuple[str, ...] = (
    "deepseek-r1",
    "qwen3.5",
    "qwen3.6",
    "qwen3.7",
    "deepseek-v4-pro",
    "nemotron-3-nano-omni",
    "nemotron-3-super",
    "nemotron-3-ultra",
    "nemotron-nano-3",
    "magistral",
    "reasoning",
    "think",
    "-o1",
    "-o3",
    "cosmos-reason",
    "deep-research",  # won't reach here due to exclusion, but defensive
    "antigravity",  # same
)

_LARGE_MODEL_SIGNALS: tuple[str, ...] = (
    "70b",
    "72b",
    "80b",
    "90b",
    "120b",
    "122b",
    "253b",
    "340b",
    "397b",
    "405b",
    "550b",
    "675b",
    "1t",
)

_HIGH_GENERAL: tuple[str, ...] = (
    "pro",
    "ultra",
    "nemotron",
    "maverick",
    "large",
    "max",
    "plus",
    "opus",
    "gpt-5",
    "gpt-4",
    "gemini-3",
    "gemini-2.5-pro",
    "compound",
    "fusion",
    "glm-5",
    "minimax-m3",
    "step-3.7",
)

_MODERATE_SIGNALS: tuple[str, ...] = (
    "instruct",
    "versatile",
    "chat",
    "medium",
    "flash",
    "scout",
    "nano",  # some nano models are capable; re-evaluated below
    "mini",
)

_LOW_SIGNALS: tuple[str, ...] = (
    "mini",
    "small",
    "nano",
    "tiny",
    "lite",
    "micro",
    "1b",
    "2b",
    "3b",
    "4b",
    "7b",
    "8b",
)


def _name_lower(model_id: str) -> str:
    """Return the lowercased model name portion (after the last '/')."""
    # model_id format: provider/org/model-name  OR  provider/model-name
    # We want the full string after the provider prefix for pattern matching,
    # because e.g. "deepseek-ai/deepseek-v4-pro" should match "v4-pro".
    parts = model_id.split("/", 1)
    suffix = parts[1] if len(parts) > 1 else parts[0]
    return suffix.lower()


def _should_exclude(model_id: str) -> bool:
    """Return True if this model should be excluded from the matrix entirely."""
    name = _name_lower(model_id)

    # Guard / moderation models
    if any(pat in name for pat in _GUARD_PATTERNS):
        return True

    # Media / embedding / non-text models
    # "codestral-embed" → embed suffix wins; "devstral" has none, passes through.
    return any(pat in name for pat in _EXCLUDE_PATTERNS)


def classify_model(model_id: str) -> dict[str, int]:
    """Return ``{role: estimated_rank}`` for a model_id based on name heuristics.

    Returns an empty dict for models that should be excluded entirely.
    All roles receive a rank; the heuristic differentiates across roles by
    boosting or penalizing per specialization signal.
    """
    assert isinstance(model_id, str), "model_id must be a string"

    if _should_exclude(model_id):
        return {}

    name = _name_lower(model_id)

    # ── Determine base rank tier ──────────────────────────────────────────────
    is_high_coder = any(pat in name for pat in _HIGH_CODING)
    is_high_reasoner = any(pat in name for pat in _HIGH_REASONING)
    is_large = any(pat in name for pat in _LARGE_MODEL_SIGNALS)
    is_high_general = any(pat in name for pat in _HIGH_GENERAL)
    is_low = any(pat in name for pat in _LOW_SIGNALS)

    # Prevent small-size tokens from dragging large models down.
    # e.g. "mistral-large" has "large" in HIGH_GENERAL AND could match "large"
    # from LOW_SIGNALS via substring — but "large" is not in _LOW_SIGNALS.
    # "gemma-3n-e2b-it" has "2b" which is in LOW_SIGNALS — correctly low.

    # ── Build per-role ranks ──────────────────────────────────────────────────
    ranks: dict[str, int] = {}

    if is_high_coder:
        # Coding specialists: very strong coding, decent everywhere else
        ranks["coding"] = 82
        ranks["testing"] = 70
        ranks["review"] = 62
        ranks["spec"] = 58
        ranks["reasoning"] = 55

    elif is_high_reasoner:
        # Reasoning specialists: strong reasoning, good at review/spec
        ranks["coding"] = 65
        ranks["testing"] = 72
        ranks["review"] = 78
        ranks["spec"] = 80
        ranks["reasoning"] = 85

    elif is_large and is_high_general:
        # Large + branded-strong: capable across all roles
        ranks["coding"] = 60
        ranks["testing"] = 65
        ranks["review"] = 72
        ranks["spec"] = 70
        ranks["reasoning"] = 68

    elif is_large:
        # Large but no strong brand signal: moderate-high general
        ranks["coding"] = 55
        ranks["testing"] = 58
        ranks["review"] = 62
        ranks["spec"] = 60
        ranks["reasoning"] = 58

    elif is_high_general and not is_low:
        # Named "pro/ultra/etc" without large-parameter count — moderate-high
        ranks["coding"] = 50
        ranks["testing"] = 55
        ranks["review"] = 62
        ranks["spec"] = 58
        ranks["reasoning"] = 55

    elif is_low:
        # Explicitly small/lite/mini/nano models
        ranks["coding"] = 28
        ranks["testing"] = 30
        ranks["review"] = 28
        ranks["spec"] = 25
        ranks["reasoning"] = 25

    else:
        # Moderate-signal or unknown: generic instruct/chat/versatile
        ranks["coding"] = 42
        ranks["testing"] = 45
        ranks["review"] = 48
        ranks["spec"] = 45
        ranks["reasoning"] = 42

    assert set(ranks.keys()) == set(_ROLES), "classify_model must return all roles"
    assert all(isinstance(v, int) for v in ranks.values()), "ranks must be ints"
    return ranks


def load_catalog(state_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """Read provider_catalog.json from state_dir.

    Returns the inner ``catalog`` dict keyed by provider name,
    where each value is a list of raw model dicts from the cache file.

    Returns an empty dict if the file is missing or unreadable.
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path instance"

    catalog_path = state_dir / "provider_catalog.json"
    if not catalog_path.exists():
        logger.warning("catalog_not_found", path=str(catalog_path))
        return {}

    try:
        text = catalog_path.read_text()
        data: dict[str, Any] = json.loads(text)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("catalog_read_failed", path=str(catalog_path), error=str(exc))
        return {}

    catalog = data.get("catalog", {})
    assert isinstance(catalog, dict), "catalog must be a dict in provider_catalog.json"

    total = sum(len(v) for v in catalog.values())
    logger.info(
        "catalog_loaded",
        providers=list(catalog.keys()),
        total_models=total,
    )
    return catalog


def auto_populate_matrix(
    state_dir: Path,
    *,
    merge_existing: bool = True,
) -> dict[str, Any]:
    """Read catalog, classify all models, write model_role_matrix.json.

    Parameters
    ----------
    state_dir:
        Directory containing provider_catalog.json and (optionally) an
        existing model_role_matrix.json.
    merge_existing:
        When True (default), load any existing matrix and preserve
        operator-curated ranks for models already present.  New models
        are added with heuristic ranks.  Models in the matrix but absent
        from the catalog receive a ``"catalog_status": "not_in_catalog"``
        field.

    Returns the final matrix dict (also written atomically to disk).
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path instance"

    catalog = load_catalog(state_dir)
    matrix_path = state_dir / "model_role_matrix.json"

    # ── Load existing matrix (if merge requested) ─────────────────────────────
    existing_entries: dict[str, dict[str, Any]] = {}  # model_id → {role: rank, ...}
    if merge_existing and matrix_path.exists():
        try:
            raw = json.loads(matrix_path.read_text())
            if "roles" in raw:
                for role, entries in raw["roles"].items():
                    if isinstance(entries, list):
                        for entry in entries:
                            mid = entry["model_id"]
                            if mid not in existing_entries:
                                existing_entries[mid] = {}
                            existing_entries[mid][role] = entry["rank"]
                    elif isinstance(entries, dict):
                        for mid, rank in entries.items():
                            if mid not in existing_entries:
                                existing_entries[mid] = {}
                            existing_entries[mid][role] = rank
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("existing_matrix_load_failed", error=str(exc))

    logger.info(
        "existing_matrix_loaded",
        curated_models=len(existing_entries),
    )

    # ── Classify all catalog models ───────────────────────────────────────────
    all_catalog_ids: set[str] = set()
    new_model_ranks: dict[str, dict[str, int]] = {}  # model_id → {role: rank}
    excluded_count = 0
    classified_count = 0

    for provider_models in catalog.values():
        for entry in provider_models:
            model_id = entry["model_id"]
            all_catalog_ids.add(model_id)

            if model_id in existing_entries:
                # Curated — don't overwrite
                continue

            role_ranks = classify_model(model_id)
            if not role_ranks:
                excluded_count += 1
                continue

            new_model_ranks[model_id] = role_ranks
            classified_count += 1

    logger.info(
        "catalog_classified",
        new_models=classified_count,
        excluded=excluded_count,
        curated_preserved=len(existing_entries),
    )

    # ── Build role → list[{model_id, rank}] structure ─────────────────────────
    # Start with existing curated entries, then append new heuristic entries.
    roles_out: dict[str, list[dict[str, Any]]] = {role: [] for role in _ROLES}

    # Existing curated entries (preserve all roles present)
    curated_not_in_catalog: set[str] = set()
    for model_id, role_ranks in existing_entries.items():
        if model_id not in all_catalog_ids:
            curated_not_in_catalog.add(model_id)
        for role in _ROLES:
            if role in role_ranks:
                entry: dict[str, Any] = {"model_id": model_id, "rank": role_ranks[role]}
                if model_id in curated_not_in_catalog:
                    entry["catalog_status"] = "not_in_catalog"
                roles_out[role].append(entry)

    if curated_not_in_catalog:
        logger.warning(
            "curated_models_not_in_catalog",
            count=len(curated_not_in_catalog),
            models=sorted(curated_not_in_catalog),
        )

    # New heuristic entries
    for model_id, role_ranks in new_model_ranks.items():
        for role in _ROLES:
            rank = role_ranks.get(role, _DEFAULT_RANK)
            roles_out[role].append({"model_id": model_id, "rank": rank})

    # Sort each role list by rank descending for readability
    for role in _ROLES:
        roles_out[role].sort(key=lambda e: e["rank"], reverse=True)

    matrix: dict[str, Any] = {
        "version": 1,
        "default_rank": _DEFAULT_RANK,
        "roles": roles_out,
    }

    # ── Atomic write ──────────────────────────────────────────────────────────
    state_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(state_dir),
        prefix=".matrix_seed_",
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

    total_entries = sum(len(v) for v in roles_out.values())
    logger.info(
        "matrix_written",
        path=str(matrix_path),
        total_entries=total_entries,
        roles=list(_ROLES),
    )

    return matrix
