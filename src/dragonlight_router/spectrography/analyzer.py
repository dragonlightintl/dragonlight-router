"""Flavor fingerprint analyzer for Model Spectrography.

Computes flavor fingerprints from raw spectrography probe results, performs
cross-model rank normalization, and calculates calibration deltas against
operator-declared profiles.

Spec reference: model-spectrography-v0.1.0-spec.md
"""

from __future__ import annotations

import math
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
# Data structures — probe results and intermediate representations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProbeResult:
    """Single probe evaluation result from a spectrography run."""

    model_id: str
    probe_id: str
    task_type: str
    domain: str
    quality_speed: str
    normalized_score: float  # 0.0-1.0 from judge
    judge_scores: dict[str, int] | None
    is_self_eval: bool = False
    error: str | None = None


@dataclass(frozen=True)
class DimensionStats:
    """Aggregated statistics for a single (model, dimension) pair."""

    mean: float
    stddev: float
    count: int


@dataclass(frozen=True)
class RawFingerprint:
    """Raw aggregated fingerprint for one model before rank normalization."""

    model_id: str
    task_scores: dict[str, DimensionStats]
    domain_scores: dict[str, DimensionStats]
    qs_scores: dict[str, DimensionStats]


@dataclass(frozen=True)
class CalibrationDelta:
    """Delta between empirical and operator-declared scores for one dimension."""

    dimension: str  # e.g. "task/generation", "domain/code", "qs/speed"
    declared: float
    empirical: float
    delta: float  # abs(empirical - declared)
    recommendation: str  # "confirm" | "review" | "update"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_CONFIRM_THRESHOLD: float = 0.05
_REVIEW_THRESHOLD: float = 0.15
_PROFILE_SCHEMA_VERSION: int = 1


# ---------------------------------------------------------------------------
# 1. aggregate_scores — group probes into raw fingerprints
# ---------------------------------------------------------------------------


def aggregate_scores(
    results: list[ProbeResult],
) -> dict[str, RawFingerprint]:
    """Group probe results by model, then by dimension, and compute stats.

    Filters out probes with errors. For each (model, dimension_type,
    dimension_value) computes mean, stddev, and count of normalized_scores.

    Returns dict of model_id -> RawFingerprint.
    """
    assert isinstance(results, list), "results must be a list"

    # Filter out errored probes
    valid = [r for r in results if r.error is None]

    if not valid:
        logger.warning("aggregate_scores_no_valid_results", total=len(results))
        return {}

    # Group by model
    by_model: dict[str, list[ProbeResult]] = {}
    for r in valid:
        assert isinstance(r, ProbeResult), "each result must be a ProbeResult"
        assert 0.0 <= r.normalized_score <= 1.0, (
            f"normalized_score out of range: {r.normalized_score}"
        )
        by_model.setdefault(r.model_id, []).append(r)

    fingerprints: dict[str, RawFingerprint] = {}
    for model_id, probes in by_model.items():
        fingerprints[model_id] = _build_raw_fingerprint(model_id, probes)

    logger.info(
        "aggregate_scores_complete",
        models=len(fingerprints),
        probes_valid=len(valid),
        probes_errored=len(results) - len(valid),
    )
    return fingerprints


def _build_raw_fingerprint(
    model_id: str,
    probes: list[ProbeResult],
) -> RawFingerprint:
    """Build a RawFingerprint from a single model's probe results."""
    assert isinstance(model_id, str) and model_id, "model_id must be non-empty"
    assert len(probes) > 0, "probes must not be empty"

    task_accum: dict[str, list[float]] = {}
    domain_accum: dict[str, list[float]] = {}
    qs_accum: dict[str, list[float]] = {}

    for p in probes:
        task_accum.setdefault(p.task_type, []).append(p.normalized_score)
        domain_accum.setdefault(p.domain, []).append(p.normalized_score)
        qs_accum.setdefault(p.quality_speed, []).append(p.normalized_score)

    return RawFingerprint(
        model_id=model_id,
        task_scores=_compute_dimension_stats(task_accum),
        domain_scores=_compute_dimension_stats(domain_accum),
        qs_scores=_compute_dimension_stats(qs_accum),
    )


def _compute_dimension_stats(
    accum: dict[str, list[float]],
) -> dict[str, DimensionStats]:
    """Compute mean, stddev, count for each dimension value."""
    assert isinstance(accum, dict), "accum must be a dict"

    result: dict[str, DimensionStats] = {}
    for key, values in accum.items():
        assert len(values) > 0, f"empty values list for dimension {key}"
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stddev = math.sqrt(variance)
        result[key] = DimensionStats(mean=mean, stddev=stddev, count=len(values))
    return result


# ---------------------------------------------------------------------------
# 2. rank_normalize — cross-model rank normalization
# ---------------------------------------------------------------------------


# DEVIATION DCS-FUNC-LEN — rank_normalize is 56 lines.
# Justification: cross-model normalization pipeline iterating over all dimension
# categories; splitting would scatter the normalization contract.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
def rank_normalize(
    raw: dict[str, RawFingerprint],
) -> dict[str, ModelFlavorProfile]:
    """Rank-normalize raw fingerprints across models per dimension.

    Within each dimension value, best model gets 1.0, worst gets 0.0.
    When only one model exists, score is 0.5. Missing dimensions get
    IBR_NEUTRAL_FLAVOR.

    Maps to FlavorScore: score=rank_normalized_mean, confidence=1.0-stddev,
    sample_count=count.
    """
    assert isinstance(raw, dict), "raw must be a dict"

    if not raw:
        return {}

    model_ids = sorted(raw.keys())
    now_iso = datetime.now(UTC).isoformat()

    # Collect all dimension values across all models for each dimension type
    all_task_dims = _collect_all_dimension_keys(raw, "task_scores")
    all_domain_dims = _collect_all_dimension_keys(raw, "domain_scores")
    all_qs_dims = _collect_all_dimension_keys(raw, "qs_scores")

    # Rank-normalize within each dimension
    task_ranked = _rank_normalize_dimension(raw, "task_scores", all_task_dims)
    domain_ranked = _rank_normalize_dimension(raw, "domain_scores", all_domain_dims)
    qs_ranked = _rank_normalize_dimension(raw, "qs_scores", all_qs_dims)

    # Build ModelFlavorProfile for each model
    profiles: dict[str, ModelFlavorProfile] = {}
    for model_id in model_ids:
        fp = raw[model_id]

        task_scores = _build_flavor_scores_from_ranked(
            fp.task_scores,
            task_ranked,
            model_id,
            IBR_TASK_TYPES,
        )
        domain_scores = _build_flavor_scores_from_ranked(
            fp.domain_scores,
            domain_ranked,
            model_id,
            IBR_DOMAINS,
        )
        qs_scores = _build_flavor_scores_from_ranked(
            fp.qs_scores,
            qs_ranked,
            model_id,
            IBR_QUALITY_SPEED,
        )

        profiles[model_id] = ModelFlavorProfile(
            model_id=model_id,
            version=_PROFILE_SCHEMA_VERSION,
            updated_at=now_iso,
            task_scores=task_scores,
            domain_scores=domain_scores,
            qs_scores=qs_scores,
        )

    logger.info("rank_normalize_complete", models=len(profiles))
    return profiles


def _collect_all_dimension_keys(
    raw: dict[str, RawFingerprint],
    attr: str,
) -> set[str]:
    """Collect all dimension keys across all models for a given score attr."""
    keys: set[str] = set()
    for fp in raw.values():
        keys.update(getattr(fp, attr).keys())
    return keys


# DEVIATION DCS-FUNC-LEN — _rank_normalize_dimension is 41 lines.
# Justification: per-dimension rank normalization with scoring and tie-handling;
# tightly coupled logic. Approved by: architect. Scope: this function.
def _rank_normalize_dimension(
    raw: dict[str, RawFingerprint],
    attr: str,
    all_keys: set[str],
) -> dict[str, dict[str, float]]:
    """Rank-normalize one dimension type across all models.

    Returns dim_value -> {model_id: normalized_score}.
    Best gets 1.0, worst gets 0.0. Single model gets 0.5.
    """
    result: dict[str, dict[str, float]] = {}

    for dim_key in all_keys:
        # Collect (model_id, mean) for all models that have this dimension
        model_means: list[tuple[str, float]] = []
        for model_id, fp in raw.items():
            scores = getattr(fp, attr)
            if dim_key in scores:
                model_means.append((model_id, scores[dim_key].mean))

        if not model_means:
            result[dim_key] = {}
            continue

        if len(model_means) == 1:
            # Single model: assign 0.5
            result[dim_key] = {model_means[0][0]: 0.5}
            continue

        # Sort by mean ascending for rank assignment
        sorted_means = sorted(model_means, key=lambda x: x[1])
        n = len(sorted_means)

        normalized: dict[str, float] = {}
        for rank, (mid, _mean) in enumerate(sorted_means):
            # rank 0 (worst) -> 0.0, rank n-1 (best) -> 1.0
            normalized[mid] = rank / (n - 1)

        result[dim_key] = normalized

    return result


def _build_flavor_scores_from_ranked(
    raw_stats: dict[str, DimensionStats],
    ranked: dict[str, dict[str, float]],
    model_id: str,
    allowed_keys: frozenset[str],
) -> dict[str, FlavorScore]:
    """Build FlavorScore dict from rank-normalized values.

    Missing dimensions (not in ranked or raw_stats) default to IBR_NEUTRAL_FLAVOR.
    """
    assert isinstance(allowed_keys, frozenset), "allowed_keys must be frozenset"

    scores: dict[str, FlavorScore] = {}
    for key in allowed_keys:
        if key in raw_stats and key in ranked and model_id in ranked[key]:
            stats = raw_stats[key]
            norm_score = ranked[key][model_id]
            confidence = max(0.0, min(1.0, 1.0 - stats.stddev))
            scores[key] = FlavorScore(
                score=max(0.0, min(1.0, norm_score)),
                confidence=confidence,
                sample_count=stats.count,
            )
        else:
            scores[key] = IBR_NEUTRAL_FLAVOR

    assert len(scores) == len(allowed_keys), (
        f"Expected {len(allowed_keys)} scores, got {len(scores)}"
    )
    return scores


# ---------------------------------------------------------------------------
# 3. compute_calibration_deltas — compare empirical vs declared
# ---------------------------------------------------------------------------


# DEVIATION DCS-FUNC-LEN — compute_calibration_deltas is 57 lines.
# Justification: loads declared profiles, iterates all dimensions, and computes
# per-dimension deltas; splitting would scatter the comparison contract.
# Approved by: architect. Scope: this function. Expiration: revisit 2026-09-01.
def compute_calibration_deltas(
    empirical: dict[str, ModelFlavorProfile],
    declared_path: Path,
) -> dict[str, dict[str, CalibrationDelta]]:
    """Load operator-declared profiles and compute deltas against empirical.

    For each model and dimension, computes absolute delta and assigns a
    recommendation: confirm (<=0.05), review (0.05-0.15), update (>0.15).

    Returns model_id -> {dimension_key: CalibrationDelta}.
    """
    assert isinstance(empirical, dict), "empirical must be a dict"
    assert isinstance(declared_path, Path), "declared_path must be a Path"

    declared_profiles = _load_declared_profiles(declared_path)
    if not declared_profiles:
        logger.warning(
            "calibration_no_declared_profiles",
            path=str(declared_path),
        )
        return {}

    deltas: dict[str, dict[str, CalibrationDelta]] = {}

    for model_id, emp_profile in empirical.items():
        if model_id not in declared_profiles:
            logger.info(
                "calibration_model_not_declared",
                model_id=model_id,
            )
            continue

        decl_profile = declared_profiles[model_id]
        model_deltas: dict[str, CalibrationDelta] = {}

        _compute_dimension_deltas(
            model_deltas,
            "task",
            emp_profile.task_scores,
            decl_profile.task_scores,
        )
        _compute_dimension_deltas(
            model_deltas,
            "domain",
            emp_profile.domain_scores,
            decl_profile.domain_scores,
        )
        _compute_dimension_deltas(
            model_deltas,
            "qs",
            emp_profile.qs_scores,
            decl_profile.qs_scores,
        )

        if model_deltas:
            deltas[model_id] = model_deltas

    logger.info(
        "calibration_deltas_computed",
        models_compared=len(deltas),
        total_deltas=sum(len(d) for d in deltas.values()),
    )
    return deltas


def _load_declared_profiles(
    path: Path,
) -> dict[str, ModelFlavorProfile]:
    """Load operator-declared profiles from YAML file.

    Returns empty dict on missing file or parse error.
    """
    assert isinstance(path, Path), "path must be a Path"

    if not path.exists():
        logger.info("declared_profiles_missing", path=str(path))
        return {}

    try:
        text = path.read_text()
        raw: dict[str, Any] = yaml.safe_load(text) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.warning(
            "declared_profiles_load_failed",
            path=str(path),
            error=str(exc),
        )
        return {}

    assert isinstance(raw, dict), "YAML root must be a mapping"

    profiles_raw = raw.get("profiles", {})
    if not isinstance(profiles_raw, dict):
        logger.warning("declared_profiles_invalid_format", path=str(path))
        return {}

    profiles: dict[str, ModelFlavorProfile] = {}
    for model_id, model_raw in profiles_raw.items():
        profile = _parse_declared_profile(str(model_id), model_raw or {})
        if profile is not None:
            profiles[profile.model_id] = profile

    return profiles


def _parse_declared_profile(
    model_id: str,
    raw: dict[str, Any],
) -> ModelFlavorProfile | None:
    """Parse a single declared profile entry. Returns None on bad data."""
    assert isinstance(model_id, str), "model_id must be a string"

    if not isinstance(raw, dict):
        logger.warning("declared_profile_invalid_entry", model_id=model_id)
        return None

    task_scores = _parse_declared_dimension(
        raw.get("task_scores", {}),
        IBR_TASK_TYPES,
    )
    domain_scores = _parse_declared_dimension(
        raw.get("domain_scores", {}),
        IBR_DOMAINS,
    )
    qs_scores = _parse_declared_dimension(
        raw.get("qs_scores", {}),
        IBR_QUALITY_SPEED,
    )

    return ModelFlavorProfile(
        model_id=model_id,
        version=int(raw.get("version", 1)),
        updated_at=str(
            raw.get("updated_at", datetime.now(UTC).isoformat()),
        ),
        task_scores=task_scores,
        domain_scores=domain_scores,
        qs_scores=qs_scores,
    )


def _parse_declared_dimension(
    raw_scores: Any,
    allowed_keys: frozenset[str],
) -> dict[str, FlavorScore]:
    """Parse a declared dimension block into FlavorScore entries."""
    assert isinstance(allowed_keys, frozenset), "allowed_keys must be frozenset"

    scores: dict[str, FlavorScore] = {}
    parsed = raw_scores if isinstance(raw_scores, dict) else {}

    for key in allowed_keys:
        if key in parsed:
            value = max(0.0, min(1.0, float(parsed[key])))
            scores[key] = FlavorScore(
                score=value,
                confidence=1.0,
                sample_count=0,
            )
        else:
            scores[key] = IBR_NEUTRAL_FLAVOR

    return scores


def _compute_dimension_deltas(
    out: dict[str, CalibrationDelta],
    prefix: str,
    empirical_scores: dict[str, FlavorScore],
    declared_scores: dict[str, FlavorScore],
) -> None:
    """Compute calibration deltas for one dimension type, appending to out."""
    assert isinstance(out, dict), "out must be a dict"
    assert isinstance(prefix, str), "prefix must be a string"

    all_keys = set(empirical_scores) | set(declared_scores)
    for key in sorted(all_keys):
        emp_fs = empirical_scores.get(key, IBR_NEUTRAL_FLAVOR)
        decl_fs = declared_scores.get(key, IBR_NEUTRAL_FLAVOR)

        delta = abs(emp_fs.score - decl_fs.score)
        recommendation = _delta_recommendation(delta)

        dimension = f"{prefix}/{key}"
        out[dimension] = CalibrationDelta(
            dimension=dimension,
            declared=decl_fs.score,
            empirical=emp_fs.score,
            delta=delta,
            recommendation=recommendation,
        )


def _delta_recommendation(delta: float) -> str:
    """Map a delta value to a recommendation string."""
    assert delta >= 0.0, f"delta must be non-negative, got {delta}"

    if delta <= _CONFIRM_THRESHOLD:
        return "confirm"
    if delta <= _REVIEW_THRESHOLD:
        return "review"
    return "update"


# ---------------------------------------------------------------------------
# 4. build_fingerprints_yaml — serialize profiles to YAML
# ---------------------------------------------------------------------------


def build_fingerprints_yaml(
    profiles: dict[str, ModelFlavorProfile],
    run_id: str,
) -> str:
    """Produce a YAML string in the model_flavor_profiles.yaml schema.

    Output format:
        version: 1
        source: "spectrography-run-<run_id>"
        generated_at: "<iso timestamp>"
        profiles:
          "model/id":
            task_scores:
              generation: 0.82
              ...
    """
    assert isinstance(profiles, dict), "profiles must be a dict"
    assert isinstance(run_id, str) and run_id, "run_id must be a non-empty string"

    data: dict[str, Any] = {
        "version": _PROFILE_SCHEMA_VERSION,
        "source": f"spectrography-run-{run_id}",
        "generated_at": datetime.now(UTC).isoformat(),
        "profiles": {},
    }

    for model_id in sorted(profiles.keys()):
        profile = profiles[model_id]
        data["profiles"][model_id] = {
            "task_scores": _serialize_scores_for_yaml(profile.task_scores),
            "domain_scores": _serialize_scores_for_yaml(profile.domain_scores),
            "qs_scores": _serialize_scores_for_yaml(profile.qs_scores),
        }

    result = yaml.dump(data, default_flow_style=False, sort_keys=False)
    assert isinstance(result, str), "yaml.dump must return a string"
    return result


def _serialize_scores_for_yaml(
    scores: dict[str, FlavorScore],
) -> dict[str, float]:
    """Serialize FlavorScore dict to simple key->score mapping for YAML."""
    assert isinstance(scores, dict), "scores must be a dict"
    return {key: round(fs.score, 4) for key, fs in sorted(scores.items())}


# ---------------------------------------------------------------------------
# 5. build_model_rankings — per-dimension model rankings
# ---------------------------------------------------------------------------


# DEVIATION DCS-FUNC-LEN — build_model_rankings is 42 lines.
# Justification: iterates all dimension categories to build per-dimension rankings;
# tightly coupled aggregation logic. Approved by: architect. Scope: this function.
def build_model_rankings(
    profiles: dict[str, ModelFlavorProfile],
) -> dict[str, list[str]]:
    """For each dimension, return model IDs sorted by score descending.

    Dimension keys follow the format "task/<value>", "domain/<value>",
    "qs/<value>".
    """
    assert isinstance(profiles, dict), "profiles must be a dict"

    if not profiles:
        return {}

    rankings: dict[str, list[str]] = {}

    # Task dimensions
    for dim_key in IBR_TASK_TYPES:
        full_key = f"task/{dim_key}"
        rankings[full_key] = _rank_models_by_dimension(
            profiles,
            "task_scores",
            dim_key,
        )

    # Domain dimensions
    for dim_key in IBR_DOMAINS:
        full_key = f"domain/{dim_key}"
        rankings[full_key] = _rank_models_by_dimension(
            profiles,
            "domain_scores",
            dim_key,
        )

    # Quality/speed dimensions
    for dim_key in IBR_QUALITY_SPEED:
        full_key = f"qs/{dim_key}"
        rankings[full_key] = _rank_models_by_dimension(
            profiles,
            "qs_scores",
            dim_key,
        )

    logger.info(
        "model_rankings_built",
        dimensions=len(rankings),
        models=len(profiles),
    )
    return rankings


def _rank_models_by_dimension(
    profiles: dict[str, ModelFlavorProfile],
    score_attr: str,
    dim_key: str,
) -> list[str]:
    """Rank model IDs by score for a single dimension, descending."""
    assert isinstance(profiles, dict), "profiles must be a dict"

    model_scores: list[tuple[str, float]] = []
    for model_id, profile in profiles.items():
        scores = getattr(profile, score_attr)
        fs = scores.get(dim_key, IBR_NEUTRAL_FLAVOR)
        model_scores.append((model_id, fs.score))

    # Sort descending by score, then alphabetically by model_id for stability
    model_scores.sort(key=lambda x: (-x[1], x[0]))
    return [mid for mid, _score in model_scores]
