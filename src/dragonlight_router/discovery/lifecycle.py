"""Discovery lifecycle management for Model Flavor Discovery.

Profile staleness detection, incremental merge of discovery results,
and fingerprint persistence to config directory.

Spec reference: model-flavor-discovery-v0.1.0-spec.md
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import yaml

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_FLAVOR,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    FlavorScore,
    ModelFlavorProfile,
)

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Decay constants — duplicated from benchmark/runner.py (IBR-FLV-06)
# to avoid circular dependency between discovery and benchmark packages.
# ---------------------------------------------------------------------------

_DECAY_THRESHOLD_DAYS: int = 30
_DECAY_RATE_PER_DAY: float = 0.01
_DECAY_TARGET: float = 0.5


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StaleProfile:
    """Result of a staleness check for a single model profile."""
    model_id: str
    updated_at: str
    age_days: float
    needs_refresh: bool  # True if age_days > threshold_days


# ---------------------------------------------------------------------------
# Staleness detection
# ---------------------------------------------------------------------------

def check_staleness(
    profiles: dict[str, ModelFlavorProfile],
    threshold_days: int = 30,
) -> list[StaleProfile]:
    """Check which profiles are older than threshold_days.

    Returns a StaleProfile entry for each profile in the dict,
    with needs_refresh=True for those exceeding the threshold.
    Logs a warning for each stale profile.
    """
    assert isinstance(profiles, dict), "profiles must be a dict"
    assert threshold_days > 0, "threshold_days must be positive"

    now = datetime.now(UTC)
    results: list[StaleProfile] = []

    for model_id, profile in profiles.items():
        assert isinstance(profile, ModelFlavorProfile), (
            f"profile for {model_id} must be ModelFlavorProfile"
        )

        updated_at = datetime.fromisoformat(profile.updated_at)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)

        age_days = (now - updated_at).total_seconds() / 86400.0
        needs_refresh = age_days > threshold_days

        entry = StaleProfile(
            model_id=model_id,
            updated_at=profile.updated_at,
            age_days=age_days,
            needs_refresh=needs_refresh,
        )
        results.append(entry)

        if needs_refresh:
            logger.warning(
                "stale_profile_detected",
                model_id=model_id,
                age_days=round(age_days, 1),
                threshold_days=threshold_days,
            )

    return results


# ---------------------------------------------------------------------------
# Decay logic (IBR-FLV-06) — re-implemented to avoid circular import
# ---------------------------------------------------------------------------

def apply_discovery_decay(
    profile: ModelFlavorProfile,
    now: datetime | None = None,
) -> ModelFlavorProfile:
    """Apply time-based decay to a discovery profile.

    Profiles older than 30 days decay toward 0.5 at 0.01/day.
    Returns a new profile with adjusted scores but the ORIGINAL
    updated_at timestamp so age tracking remains correct.

    IBR-FLV-06: Discovery profiles older than 30 days MUST decay toward 0.5.
    """
    assert isinstance(profile, ModelFlavorProfile), "profile must be ModelFlavorProfile"

    if now is None:
        now = datetime.now(UTC)

    updated_at = datetime.fromisoformat(profile.updated_at)
    if updated_at.tzinfo is None:
        updated_at = updated_at.replace(tzinfo=UTC)

    age_days = (now - updated_at).total_seconds() / 86400.0

    if age_days <= _DECAY_THRESHOLD_DAYS:
        return profile

    decay_days = age_days - _DECAY_THRESHOLD_DAYS
    assert decay_days > 0, "decay_days must be positive when past threshold"

    task_scores = _decay_dimension(profile.task_scores, decay_days)
    domain_scores = _decay_dimension(profile.domain_scores, decay_days)
    qs_scores = _decay_dimension(profile.qs_scores, decay_days)

    return ModelFlavorProfile(
        model_id=profile.model_id,
        version=profile.version,
        updated_at=profile.updated_at,  # Preserve original timestamp
        task_scores=task_scores,
        domain_scores=domain_scores,
        qs_scores=qs_scores,
    )


def _decay_dimension(
    scores: dict[str, FlavorScore],
    decay_days: float,
) -> dict[str, FlavorScore]:
    """Apply decay to all scores in a dimension dict."""
    assert isinstance(scores, dict), "scores must be a dict"
    assert decay_days > 0, "decay_days must be positive"

    result: dict[str, FlavorScore] = {}
    for key, fs in scores.items():
        decayed_score = _decay_single_score(fs.score, decay_days)
        result[key] = FlavorScore(
            score=decayed_score,
            confidence=fs.confidence,
            sample_count=fs.sample_count,
        )
    return result


def _decay_single_score(score: float, decay_days: float) -> float:
    """Decay a single score toward _DECAY_TARGET.

    Formula: score + (target - score) * min(1.0, decay_days * rate)
    """
    assert 0.0 <= score <= 1.0, f"score must be in [0.0, 1.0], got {score}"
    assert decay_days > 0, "decay_days must be positive"

    decay_factor = min(1.0, decay_days * _DECAY_RATE_PER_DAY)
    result = score + (_DECAY_TARGET - score) * decay_factor

    result = max(0.0, min(1.0, result))
    assert 0.0 <= result <= 1.0, f"decayed score out of range: {result}"
    return result


# ---------------------------------------------------------------------------
# Incremental merge
# ---------------------------------------------------------------------------

def merge_incremental(
    existing: dict[str, ModelFlavorProfile],
    new_results: dict[str, ModelFlavorProfile],
) -> dict[str, ModelFlavorProfile]:
    """Merge new discovery results into an existing profile set.

    Resolution:
    - Models in new_results replace their entry in existing.
    - Models only in existing are preserved unchanged.
    - Models only in new_results are added.

    Returns the merged dict.
    """
    assert isinstance(existing, dict), "existing must be a dict"
    assert isinstance(new_results, dict), "new_results must be a dict"

    merged: dict[str, ModelFlavorProfile] = {}

    # Preserve existing profiles, overwriting with new results where available
    for model_id, profile in existing.items():
        assert isinstance(profile, ModelFlavorProfile), (
            f"existing profile for {model_id} must be ModelFlavorProfile"
        )
        merged[model_id] = new_results.get(model_id, profile)

    # Add models only in new_results
    for model_id, profile in new_results.items():
        assert isinstance(profile, ModelFlavorProfile), (
            f"new profile for {model_id} must be ModelFlavorProfile"
        )
        if model_id not in merged:
            merged[model_id] = profile

    assert len(merged) == len(set(existing) | set(new_results)), (
        "merged dict must contain union of all model IDs"
    )
    return merged


# ---------------------------------------------------------------------------
# Fingerprint I/O
# ---------------------------------------------------------------------------

def write_fingerprints_yaml(yaml_content: str, output_path: Path) -> None:
    """Write YAML fingerprint content to the given path.

    Creates parent directories if needed.
    """
    assert isinstance(yaml_content, str), "yaml_content must be a string"
    assert isinstance(output_path, Path), "output_path must be a Path instance"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml_content)

    logger.info(
        "fingerprints_written",
        path=str(output_path),
        size_bytes=len(yaml_content.encode("utf-8")),
    )


def load_existing_fingerprints(
    fingerprints_path: Path,
) -> dict[str, ModelFlavorProfile]:
    """Load existing fingerprints from a YAML file.

    Returns empty dict if the file doesn't exist or is unparseable.
    Follows the same YAML parsing pattern as selection/flavor.py's
    _parse_profiles().
    """
    assert isinstance(fingerprints_path, Path), "fingerprints_path must be a Path instance"

    if not fingerprints_path.exists():
        logger.info("fingerprints_file_missing", path=str(fingerprints_path))
        return {}

    try:
        text = fingerprints_path.read_text()
        raw: dict[str, Any] = yaml.safe_load(text) or {}
        assert isinstance(raw, dict), "YAML root must be a mapping"
    except (yaml.YAMLError, OSError, AssertionError) as exc:
        logger.warning(
            "fingerprints_load_failed",
            path=str(fingerprints_path),
            error=str(exc),
        )
        return {}

    profiles_raw = raw.get("profiles", {})
    assert isinstance(profiles_raw, dict), "profiles key must be a dict"

    return _parse_profiles(profiles_raw)


def _parse_profiles(
    profiles_raw: dict[str, Any],
) -> dict[str, ModelFlavorProfile]:
    """Parse all profiles from the YAML 'profiles' block."""
    assert isinstance(profiles_raw, dict), "profiles_raw must be a dict"
    result: dict[str, ModelFlavorProfile] = {}
    for model_id, model_raw in profiles_raw.items():
        profile = _parse_single_profile(str(model_id), model_raw or {})
        if profile is not None:
            result[profile.model_id] = profile
    return result


def _parse_single_profile(
    model_id: str, raw: dict[str, Any],
) -> ModelFlavorProfile | None:
    """Parse one model's profile entry. Returns None on bad data."""
    assert isinstance(model_id, str), "model_id must be a string"
    if not isinstance(raw, dict):
        logger.warning("fingerprint_profile_invalid_entry", model_id=model_id)
        return None

    task_scores = _parse_dimension_scores(
        raw.get("task_scores", {}), IBR_TASK_TYPES,
    )
    domain_scores = _parse_dimension_scores(
        raw.get("domain_scores", {}), IBR_DOMAINS,
    )
    qs_scores = _parse_dimension_scores(
        raw.get("qs_scores", {}), IBR_QUALITY_SPEED,
    )

    return ModelFlavorProfile(
        model_id=model_id,
        version=int(raw.get("version", 1)),
        updated_at=str(
            raw.get("updated_at", datetime.now(UTC).isoformat())
        ),
        task_scores=task_scores,
        domain_scores=domain_scores,
        qs_scores=qs_scores,
    )


def _parse_dimension_scores(
    raw_scores: Any,
    allowed_keys: frozenset[str],
) -> dict[str, FlavorScore]:
    """Parse a dimension block into FlavorScore entries.

    Declared values get confidence=1.0 (known but not observed).
    Missing dimensions filled with neutral default.
    """
    assert isinstance(allowed_keys, frozenset), "allowed_keys must be frozenset"

    scores: dict[str, FlavorScore] = {}
    parsed = raw_scores if isinstance(raw_scores, dict) else {}

    for key in allowed_keys:
        if key in parsed:
            value = _clamp_score(float(parsed[key]))
            scores[key] = FlavorScore(
                score=value, confidence=1.0, sample_count=0,
            )
        else:
            scores[key] = IBR_NEUTRAL_FLAVOR

    assert len(scores) == len(allowed_keys), (
        f"Expected {len(allowed_keys)} scores, got {len(scores)}"
    )
    return scores


def _clamp_score(value: float) -> float:
    """Clamp a score to [0.0, 1.0]."""
    clamped = max(0.0, min(1.0, value))
    assert 0.0 <= clamped <= 1.0, f"clamped score out of bounds: {clamped}"
    return clamped


# ---------------------------------------------------------------------------
# Discovery targeting
# ---------------------------------------------------------------------------

def get_models_needing_discovery(
    role_matrix_path: Path,
    existing_profiles: dict[str, ModelFlavorProfile],
    staleness_days: int = 30,
) -> list[str]:
    """Return model IDs that need discovery (missing or stale profiles).

    Reads the role matrix JSON to collect all unique model IDs, then
    returns those that either have no existing profile or have a profile
    older than staleness_days.

    Helps the --models flag default to 'only models that need it'.
    """
    assert isinstance(role_matrix_path, Path), "role_matrix_path must be a Path instance"
    assert isinstance(existing_profiles, dict), "existing_profiles must be a dict"
    assert staleness_days > 0, "staleness_days must be positive"

    all_model_ids = _collect_model_ids_from_matrix(role_matrix_path)
    if not all_model_ids:
        return []

    now = datetime.now(UTC)
    needs_discovery: list[str] = []

    for model_id in sorted(all_model_ids):
        if model_id not in existing_profiles:
            needs_discovery.append(model_id)
            continue

        profile = existing_profiles[model_id]
        updated_at = datetime.fromisoformat(profile.updated_at)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)

        age_days = (now - updated_at).total_seconds() / 86400.0
        if age_days > staleness_days:
            needs_discovery.append(model_id)

    logger.info(
        "models_needing_discovery",
        total_in_matrix=len(all_model_ids),
        needing_discovery=len(needs_discovery),
        staleness_days=staleness_days,
    )
    return needs_discovery


def _collect_model_ids_from_matrix(matrix_path: Path) -> set[str]:
    """Read role matrix JSON and collect all unique model IDs.

    Supports both the full schema (with 'roles' key containing lists of
    {model_id, rank} entries) and the flat dict format.
    """
    assert isinstance(matrix_path, Path), "matrix_path must be a Path instance"

    if not matrix_path.exists():
        logger.warning("role_matrix_missing", path=str(matrix_path))
        return set()

    try:
        text = matrix_path.read_text()
        raw: dict[str, Any] = json.loads(text)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "role_matrix_load_failed",
            path=str(matrix_path),
            error=str(exc),
        )
        return set()

    model_ids: set[str] = set()

    # Full schema: {"version": ..., "roles": {"coding": [{"model_id": "x", "rank": 90}]}}
    roles_data = raw.get("roles", raw)
    assert isinstance(roles_data, dict), "roles data must be a dict"

    for _role, entries in roles_data.items():
        if _role in ("version", "default_rank"):
            continue
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, dict) and "model_id" in entry:
                    model_ids.add(str(entry["model_id"]))
        elif isinstance(entries, dict):
            for model_id in entries:
                model_ids.add(str(model_id))

    return model_ids
