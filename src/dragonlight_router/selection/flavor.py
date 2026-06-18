"""Model flavor profile system for Intent Based Router (IBR).

Loads operator-declared flavor profiles from YAML, computes flavor match
scores against classified intents, and provides confidence gating.

Mirrors the RoleMatrix pattern: load at boot, hot-reload via mtime check.
Missing or malformed YAML yields empty profiles (HAZ-019 mitigation).
"""
from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog
import yaml

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Type imports — parallel development fallback (IBR-DATA-01)
# ---------------------------------------------------------------------------

try:
    from dragonlight_router.core.types import (
        IBR_DOMAINS,
        IBR_NEUTRAL_FLAVOR,
        IBR_QUALITY_SPEED,
        IBR_TASK_TYPES,
        ClassifiedIntent,
        FlavorScore,
        ModelFlavorProfile,
    )
except ImportError:
    from dataclasses import dataclass

    IBR_TASK_TYPES: frozenset[str] = frozenset({  # type: ignore[no-redef]
        "generation", "analysis", "refactoring", "summarization",
        "creative", "reasoning", "lookup", "translation",
    })
    IBR_DOMAINS: frozenset[str] = frozenset({  # type: ignore[no-redef]
        "code", "technical", "legal", "business",
        "creative_writing", "general",
    })
    IBR_QUALITY_SPEED: frozenset[str] = frozenset({  # type: ignore[no-redef]
        "quality", "balanced", "speed",
    })

    @dataclass(frozen=True)
    class FlavorScore:  # type: ignore[no-redef]
        """Single dimension score within a flavor profile."""
        score: float
        confidence: float
        sample_count: int

    @dataclass(frozen=True)
    class ModelFlavorProfile:  # type: ignore[no-redef]
        """Full flavor profile for one model."""
        model_id: str
        version: int
        updated_at: str
        task_scores: dict[str, FlavorScore]
        domain_scores: dict[str, FlavorScore]
        qs_scores: dict[str, FlavorScore]

    @dataclass(frozen=True)
    class ClassifiedIntent:  # type: ignore[no-redef]
        """Classification output from the intent classifier."""
        task_type: str
        domain: str
        quality_speed: str
        confidence: float
        latency_ms: float
        from_cache: bool

    IBR_NEUTRAL_FLAVOR = FlavorScore(score=0.5, confidence=0.0, sample_count=0)


# ---------------------------------------------------------------------------
# Scoring weights (spec section 4.1)
# ---------------------------------------------------------------------------

_TASK_WEIGHT: float = 0.50
_DOMAIN_WEIGHT: float = 0.30
_QS_WEIGHT: float = 0.20

assert abs(_TASK_WEIGHT + _DOMAIN_WEIGHT + _QS_WEIGHT - 1.0) < 1e-9, (
    "Flavor match dimension weights must sum to 1.0"
)


# ---------------------------------------------------------------------------
# FlavorProfileLoader — mirrors RoleMatrix pattern
# ---------------------------------------------------------------------------

class FlavorProfileLoader:
    """Load and hot-reload model flavor profiles from YAML.

    Follows the RoleMatrix pattern: load at boot, mtime-based hot-reload.
    Missing or unparseable files yield empty profiles (HAZ-019).
    """

    def __init__(self, profile_path: Path) -> None:
        assert isinstance(profile_path, Path), "profile_path must be a Path instance"
        self._path = profile_path
        self._mtime: float = 0.0
        self._profiles: dict[str, ModelFlavorProfile] = {}
        self._profiles = self.load()

    def load(self) -> dict[str, ModelFlavorProfile]:
        """Parse YAML and build profiles with defaults.

        Returns empty dict on missing file or parse error (HAZ-019).
        Operator-declared scores get confidence=1.0, sample_count=0.
        Unlisted dimensions default to neutral (0.5, 0.0, 0).
        """
        if not self._path.exists():
            logger.info("flavor_profiles_missing", path=str(self._path))
            self._profiles = {}
            return self._profiles

        raw = self._read_yaml()
        if raw is None:
            self._profiles = {}
            return self._profiles

        profiles_raw = raw.get("profiles", {})
        assert isinstance(profiles_raw, dict), "profiles key must be a dict"

        profiles = _parse_profiles(profiles_raw)
        self._profiles = profiles

        logger.info(
            "flavor_profiles_loaded",
            path=str(self._path),
            model_count=len(profiles),
        )
        return self._profiles

    @property
    def profiles(self) -> dict[str, ModelFlavorProfile]:
        """Return currently loaded profiles."""
        return self._profiles

    def reload_if_changed(self) -> None:
        """Check file mtime and reload if modified (same as RoleMatrix)."""
        if not self._path.exists():
            return

        try:
            current_mtime = os.path.getmtime(self._path)
            if current_mtime > self._mtime:
                self.load()
        except OSError as exc:
            logger.warning("flavor_profile_stat_failed", error=str(exc))

    def _read_yaml(self) -> dict[str, Any] | None:
        """Read and parse YAML. Returns None on failure (HAZ-019)."""
        try:
            text = self._path.read_text()
            raw: dict[str, Any] = yaml.safe_load(text) or {}
            self._mtime = os.path.getmtime(self._path)
            assert isinstance(raw, dict), "YAML root must be a mapping"
            return raw
        except (yaml.YAMLError, OSError, AssertionError) as exc:
            logger.warning(
                "flavor_profile_load_failed",
                path=str(self._path),
                error=str(exc),
            )
            return None


# ---------------------------------------------------------------------------
# YAML parsing helpers
# ---------------------------------------------------------------------------

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
        logger.warning("flavor_profile_invalid_entry", model_id=model_id)
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
    """Clamp a score to [0.0, 1.0] (IBR-FLV-05)."""
    clamped = max(0.0, min(1.0, value))
    assert 0.0 <= clamped <= 1.0, f"clamped score out of bounds: {clamped}"
    return clamped


# ---------------------------------------------------------------------------
# Profile lookup helper
# ---------------------------------------------------------------------------

def get_profile_for_model(
    model_id: str,
    profiles: dict[str, ModelFlavorProfile],
) -> ModelFlavorProfile:
    """Return the profile for a model_id, or a neutral default if missing.

    Neutral profiles have 0.5 score / 0.0 confidence across all dimensions,
    ensuring unknown models neither benefit nor suffer from flavor matching.
    """
    assert isinstance(model_id, str), "model_id must be a string"
    assert isinstance(profiles, dict), "profiles must be a dict"

    if model_id in profiles:
        return profiles[model_id]

    return _build_neutral_profile(model_id)


def _build_neutral_profile(model_id: str) -> ModelFlavorProfile:
    """Build a neutral default profile (all dimensions at 0.5, 0.0 conf)."""
    assert isinstance(model_id, str), "model_id must be a string"

    neutral_task = dict.fromkeys(IBR_TASK_TYPES, IBR_NEUTRAL_FLAVOR)
    neutral_domain = dict.fromkeys(IBR_DOMAINS, IBR_NEUTRAL_FLAVOR)
    neutral_qs = dict.fromkeys(IBR_QUALITY_SPEED, IBR_NEUTRAL_FLAVOR)

    return ModelFlavorProfile(
        model_id=model_id,
        version=1,
        updated_at=datetime.now(UTC).isoformat(),
        task_scores=neutral_task,
        domain_scores=neutral_domain,
        qs_scores=neutral_qs,
    )


# ---------------------------------------------------------------------------
# Flavor match scoring (spec section 4.1)
# ---------------------------------------------------------------------------

def compute_flavor_match(
    intent: ClassifiedIntent,
    profile: ModelFlavorProfile,
) -> float:
    """Compute weighted flavor match score for a single model.

    Returns:
        Weighted sum: 0.50 * task + 0.30 * domain + 0.20 * quality_speed.
        Always in [0.0, 1.0]. Missing dimensions use neutral default (0.5).
    """
    assert isinstance(intent, ClassifiedIntent), "intent must be ClassifiedIntent"
    assert isinstance(profile, ModelFlavorProfile), "profile must be ModelFlavorProfile"

    task_fs = profile.task_scores.get(intent.task_type, IBR_NEUTRAL_FLAVOR)
    domain_fs = profile.domain_scores.get(intent.domain, IBR_NEUTRAL_FLAVOR)
    qs_fs = profile.qs_scores.get(intent.quality_speed, IBR_NEUTRAL_FLAVOR)

    result = (
        _TASK_WEIGHT * task_fs.score
        + _DOMAIN_WEIGHT * domain_fs.score
        + _QS_WEIGHT * qs_fs.score
    )

    result = max(0.0, min(1.0, result))
    assert 0.0 <= result <= 1.0, f"flavor_match out of bounds: {result}"
    return result


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------

def compute_flavor_scores(
    intent: ClassifiedIntent | None,
    profiles: dict[str, ModelFlavorProfile],
    candidate_ids: list[str],
) -> dict[str, float]:
    """Compute flavor match scores for all candidates.

    Returns model_id -> flavor_match_score for each candidate.
    If intent is None, returns empty dict (IBR inactive for this request).
    """
    assert isinstance(profiles, dict), "profiles must be a dict"
    assert isinstance(candidate_ids, list), "candidate_ids must be a list"

    if intent is None:
        return {}

    scores: dict[str, float] = {}
    for model_id in candidate_ids:
        profile = get_profile_for_model(model_id, profiles)
        scores[model_id] = compute_flavor_match(intent, profile)

    assert len(scores) == len(candidate_ids), (
        f"Expected {len(candidate_ids)} scores, got {len(scores)}"
    )
    return scores


# ---------------------------------------------------------------------------
# Confidence gating (spec section 4.1)
# ---------------------------------------------------------------------------

def should_apply_flavor_match(
    intent: ClassifiedIntent | None,
    profile: ModelFlavorProfile,
    confidence_threshold: float = 0.6,
    profile_confidence_threshold: float = 0.3,
) -> bool:
    """Determine whether flavor match should be applied for this request.

    Returns False if:
    - intent is None (classification failed/skipped)
    - intent.confidence < confidence_threshold
    - average profile confidence across matched dimensions < profile_confidence_threshold

    This prevents low-confidence classifications from distorting routing.
    """
    assert isinstance(profile, ModelFlavorProfile), "profile must be ModelFlavorProfile"
    assert 0.0 <= confidence_threshold <= 1.0, "confidence_threshold must be in [0.0, 1.0]"
    assert 0.0 <= profile_confidence_threshold <= 1.0, (
        "profile_confidence_threshold must be in [0.0, 1.0]"
    )

    if intent is None:
        return False

    if intent.confidence < confidence_threshold:
        return False

    avg_confidence = _average_matched_confidence(intent, profile)
    return avg_confidence >= profile_confidence_threshold


def _average_matched_confidence(
    intent: ClassifiedIntent,
    profile: ModelFlavorProfile,
) -> float:
    """Compute average confidence across the three matched dimensions."""
    assert isinstance(intent, ClassifiedIntent), "intent must be ClassifiedIntent"
    assert isinstance(profile, ModelFlavorProfile), "profile must be ModelFlavorProfile"

    task_fs = profile.task_scores.get(intent.task_type, IBR_NEUTRAL_FLAVOR)
    domain_fs = profile.domain_scores.get(intent.domain, IBR_NEUTRAL_FLAVOR)
    qs_fs = profile.qs_scores.get(intent.quality_speed, IBR_NEUTRAL_FLAVOR)

    avg = (task_fs.confidence + domain_fs.confidence + qs_fs.confidence) / 3.0
    assert 0.0 <= avg <= 1.0, f"average confidence out of bounds: {avg}"
    return avg
