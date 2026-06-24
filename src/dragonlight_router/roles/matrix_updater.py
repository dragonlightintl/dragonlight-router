"""Spectrography-to-matrix bridge.

Converts empirical model rankings from spectrography probe evaluations into
role matrix format, and blends them with existing operator-curated ranks.

Spec reference: spectrography-bridge-v0.1.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
import yaml

from dragonlight_router.core.types import (
    IBR_TASK_TYPES,
    ModelSpectrographProfile,
    SpectrographScore,
)

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Rank scaling: spectrography scores are 0.0-1.0; matrix ranks are 0-100.
# 0 and 100 are reserved for explicit operator overrides.
_RANK_MIN: int = 10
_RANK_MAX: int = 99

# Matrix schema version written on update
_MATRIX_SCHEMA_VERSION: int = 1

# Default rank when a model appears in matrix but has no empirical data
_DEFAULT_RANK: int = 20

# IBR task_type -> role mapping
# Keys are task_type values from IBR_TASK_TYPES.
# A task type may contribute to multiple roles.
_TASK_TYPE_TO_ROLES: dict[str, list[str]] = {
    "generation": ["coding"],
    "refactoring": ["coding"],
    "analysis": ["review"],
    "summarization": ["review"],
    "reasoning": ["reasoning"],
}

# Domain-gated override: generation + code domain -> also contributes to testing
_GENERATION_CODE_ROLES: list[str] = ["testing"]

# All task types contribute to spec (general capability)
_SPEC_ROLE: str = "spec"

# Canonical set of roles managed by this module
_MANAGED_ROLES: frozenset[str] = frozenset({"coding", "review", "reasoning", "testing", "spec"})


# ---------------------------------------------------------------------------
# 1. score_to_rank — scale 0.0-1.0 to [10, 99]
# ---------------------------------------------------------------------------


def score_to_rank(score: float) -> int:
    """Scale a normalized spectrography score (0.0-1.0) to a matrix rank integer.

    Returns an integer in [_RANK_MIN, _RANK_MAX].
    0 and 100 are reserved for explicit operator overrides.
    """
    assert 0.0 <= score <= 1.0, f"score must be in [0.0, 1.0], got {score}"

    raw = int(score * 100)
    return max(_RANK_MIN, min(_RANK_MAX, raw))


# ---------------------------------------------------------------------------
# 2. rankings_to_matrix — convert spectrography rankings to role matrix format
# ---------------------------------------------------------------------------


def rankings_to_matrix(
    profiles: dict[str, ModelSpectrographProfile],
    *,
    roles: list[str] | None = None,
) -> dict[str, dict[str, int]]:
    """Convert spectrography profiles into role matrix format.

    Mapping logic:
      - generation, refactoring -> coding
      - analysis, summarization -> review
      - reasoning -> reasoning
      - generation (any occurrence) -> testing  [domain-agnostic per spec]
      - all task types -> spec

    For each role, aggregates task dimension scores per model by averaging,
    then scales to matrix rank.

    Args:
        profiles: model_id -> ModelSpectrographProfile from rank_normalize()
        roles: optional subset of roles to populate (default: all _MANAGED_ROLES)

    Returns:
        dict[role, dict[model_id, rank]]
    """
    assert isinstance(profiles, dict), "profiles must be a dict"

    target_roles: frozenset[str] = frozenset(roles) if roles is not None else _MANAGED_ROLES

    # Accumulate scores per (role, model): list of contributing task scores
    role_model_scores: dict[str, dict[str, list[float]]] = {role: {} for role in target_roles}

    for model_id, profile in profiles.items():
        assert isinstance(model_id, str) and model_id, "model_id must be a non-empty string"

        # Spec role aggregates all task dimensions
        if _SPEC_ROLE in target_roles:
            all_task_scores = [
                fs.score for fs in profile.task_scores.values() if isinstance(fs, SpectrographScore)
            ]
            if all_task_scores:
                role_model_scores[_SPEC_ROLE].setdefault(model_id, []).extend(all_task_scores)

        # Task-type mapped roles
        for task_type, role_list in _TASK_TYPE_TO_ROLES.items():
            if task_type not in IBR_TASK_TYPES:
                continue
            task_score = profile.task_scores.get(task_type)
            if task_score is None:
                continue
            for role in role_list:
                if role not in target_roles:
                    continue
                role_model_scores[role].setdefault(model_id, []).append(task_score.score)

        # Testing: generation task type (domain-agnostic per spec)
        if "testing" in target_roles:
            gen_score = profile.task_scores.get("generation")
            if gen_score is not None:
                role_model_scores["testing"].setdefault(model_id, []).append(gen_score.score)

    # Convert accumulated scores to rank integers
    result: dict[str, dict[str, int]] = {}
    for role in target_roles:
        model_scores = role_model_scores[role]
        if not model_scores:
            result[role] = {}
            continue

        role_ranks: dict[str, int] = {}
        for model_id, scores in model_scores.items():
            assert len(scores) > 0, f"scores list empty for {role}/{model_id}"
            avg_score = sum(scores) / len(scores)
            role_ranks[model_id] = score_to_rank(avg_score)

        result[role] = role_ranks

    logger.info(
        "rankings_to_matrix_complete",
        roles=sorted(target_roles),
        models=len(profiles),
    )
    return result


# ---------------------------------------------------------------------------
# 3. blend_ranks — blend empirical and existing ranks
# ---------------------------------------------------------------------------


def blend_ranks(
    empirical: dict[str, int],
    existing: dict[str, int],
    blend_weight: float,
) -> dict[str, int]:
    """Blend empirical and existing ranks for a single role.

    blend_weight controls empirical weight:
      1.0 — pure empirical (existing ignored)
      0.7 — 70% empirical, 30% existing (default)
      0.0 — keep existing (no update)

    New models (no existing rank): pure empirical.
    Models with existing rank but no empirical data: unchanged.

    Args:
        empirical: model_id -> empirical rank (from spectrography)
        existing: model_id -> existing rank (from current matrix)
        blend_weight: float in [0.0, 1.0]

    Returns:
        Blended dict[model_id, rank]
    """
    assert 0.0 <= blend_weight <= 1.0, f"blend_weight must be in [0.0, 1.0], got {blend_weight}"
    assert isinstance(empirical, dict), "empirical must be a dict"
    assert isinstance(existing, dict), "existing must be a dict"

    result: dict[str, int] = {}

    # Start from existing — keep models with no empirical data unchanged
    for model_id, ex_rank in existing.items():
        if model_id in empirical:
            emp_rank = empirical[model_id]
            blended = blend_weight * emp_rank + (1.0 - blend_weight) * ex_rank
            result[model_id] = max(_RANK_MIN, min(_RANK_MAX, int(round(blended))))
        else:
            # No empirical data for this model — keep existing
            result[model_id] = ex_rank

    # Add new models that only appear in empirical (pure empirical)
    for model_id, emp_rank in empirical.items():
        if model_id not in existing:
            result[model_id] = emp_rank

    return result


# ---------------------------------------------------------------------------
# 4. find_latest_spectrography_run — locate most recent run directory
# ---------------------------------------------------------------------------


def find_latest_spectrography_run(output_dir: Path) -> Path | None:
    """Find the most recent spectrography run directory by timestamp.

    Run directories are named with a timestamp prefix: YYYYMMDD-HHMMSS-<hex>.
    Returns the Path to the latest one, or None if output_dir is missing/empty.

    Args:
        output_dir: directory containing spectrography run subdirectories
    """
    assert isinstance(output_dir, Path), "output_dir must be a Path"

    if not output_dir.exists() or not output_dir.is_dir():
        logger.info("spectrography_output_dir_missing", path=str(output_dir))
        return None

    # Collect subdirectories that look like run dirs (contain fingerprints.yaml
    # or report.json)
    candidates: list[Path] = []
    for entry in output_dir.iterdir():
        if not entry.is_dir():
            continue
        has_fingerprints = (entry / "fingerprints.yaml").exists()
        has_report = (entry / "report.json").exists()
        if has_fingerprints or has_report:
            candidates.append(entry)

    if not candidates:
        logger.info(
            "no_spectrography_runs_found",
            output_dir=str(output_dir),
        )
        return None

    # Sort by directory name — timestamp prefix ensures chronological order
    latest = sorted(candidates, key=lambda p: p.name)[-1]
    logger.info(
        "latest_spectrography_run_found",
        path=str(latest),
        total_runs=len(candidates),
    )
    return latest


# ---------------------------------------------------------------------------
# 5. load_profiles_from_run — load fingerprints.yaml from a run directory
# ---------------------------------------------------------------------------


def load_profiles_from_run(run_dir: Path) -> dict[str, ModelSpectrographProfile]:
    """Load ModelSpectrographProfile instances from a run directory.

    Tries fingerprints.yaml first, then report.json profiles section.
    Returns empty dict on failure.

    Args:
        run_dir: spectrography run directory containing output files
    """
    assert isinstance(run_dir, Path), "run_dir must be a Path"

    fingerprints_path = run_dir / "fingerprints.yaml"
    report_path = run_dir / "report.json"

    if fingerprints_path.exists():
        return _load_profiles_from_yaml(fingerprints_path)

    if report_path.exists():
        return _load_profiles_from_report_json(report_path)

    logger.warning(
        "no_profiles_found_in_run_dir",
        run_dir=str(run_dir),
    )
    return {}


def _load_profiles_from_yaml(
    path: Path,
) -> dict[str, ModelSpectrographProfile]:
    """Parse fingerprints.yaml into ModelSpectrographProfile instances."""
    assert isinstance(path, Path), "path must be a Path"

    try:
        raw: dict[str, Any] = yaml.safe_load(path.read_text()) or {}
    except (yaml.YAMLError, OSError) as exc:
        logger.warning(
            "fingerprints_yaml_load_failed",
            path=str(path),
            error=str(exc),
        )
        return {}

    profiles_raw = raw.get("profiles", {})
    if not isinstance(profiles_raw, dict):
        logger.warning("fingerprints_yaml_invalid_format", path=str(path))
        return {}

    profiles: dict[str, ModelSpectrographProfile] = {}
    for model_id, model_data in profiles_raw.items():
        profile = _parse_yaml_profile(str(model_id), model_data or {})
        if profile is not None:
            profiles[profile.model_id] = profile

    logger.info(
        "profiles_loaded_from_yaml",
        path=str(path),
        count=len(profiles),
    )
    return profiles


def _parse_yaml_profile(
    model_id: str,
    raw: dict[str, Any],
) -> ModelSpectrographProfile | None:
    """Parse a single profile entry from fingerprints.yaml."""
    assert isinstance(model_id, str), "model_id must be a string"

    if not isinstance(raw, dict):
        logger.warning("invalid_yaml_profile_entry", model_id=model_id)
        return None

    task_scores = _parse_score_block(raw.get("task_scores", {}))
    domain_scores = _parse_score_block(raw.get("domain_scores", {}))
    qs_scores = _parse_score_block(raw.get("qs_scores", {}))

    return ModelSpectrographProfile(
        model_id=model_id,
        version=int(raw.get("version", 1)),
        updated_at=str(raw.get("updated_at", "")),
        task_scores=task_scores,
        domain_scores=domain_scores,
        qs_scores=qs_scores,
    )


def _parse_score_block(
    raw: Any,
) -> dict[str, SpectrographScore]:
    """Parse a scores block (key -> float or key -> {score, confidence, ...})."""
    if not isinstance(raw, dict):
        return {}

    scores: dict[str, SpectrographScore] = {}
    for key, value in raw.items():
        if isinstance(value, (int, float)):
            scores[key] = SpectrographScore(
                score=float(value),
                confidence=1.0,
                sample_count=0,
            )
        elif isinstance(value, dict):
            scores[key] = SpectrographScore(
                score=float(value.get("score", 0.5)),
                confidence=float(value.get("confidence", 1.0)),
                sample_count=int(value.get("sample_count", 0)),
            )

    return scores


def _load_profiles_from_report_json(
    path: Path,
) -> dict[str, ModelSpectrographProfile]:
    """Parse profiles from a spectrography report.json file."""
    assert isinstance(path, Path), "path must be a Path"

    try:
        raw: dict[str, Any] = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "report_json_load_failed",
            path=str(path),
            error=str(exc),
        )
        return {}

    profiles_raw = raw.get("profiles", {})
    if not isinstance(profiles_raw, dict):
        logger.warning("report_json_no_profiles", path=str(path))
        return {}

    profiles: dict[str, ModelSpectrographProfile] = {}
    for model_id, model_data in profiles_raw.items():
        if not isinstance(model_data, dict):
            continue
        profile = _parse_yaml_profile(str(model_id), model_data)
        if profile is not None:
            profiles[profile.model_id] = profile

    logger.info(
        "profiles_loaded_from_report_json",
        path=str(path),
        count=len(profiles),
    )
    return profiles


# ---------------------------------------------------------------------------
# 6. load_existing_matrix — read current role matrix from state_dir
# ---------------------------------------------------------------------------


def load_existing_matrix(state_dir: Path) -> dict[str, dict[str, int]]:
    """Load the existing role matrix from state_dir/model_role_matrix.json.

    Supports both full schema (with "roles" key) and flat dict format.
    Returns empty dict if file is missing or unparseable.

    Args:
        state_dir: router state directory containing model_role_matrix.json
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"

    matrix_path = state_dir / "model_role_matrix.json"
    if not matrix_path.exists():
        logger.info("existing_matrix_missing", path=str(matrix_path))
        return {}

    try:
        raw: dict[str, Any] = json.loads(matrix_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning(
            "existing_matrix_load_failed",
            path=str(matrix_path),
            error=str(exc),
        )
        return {}

    if "roles" in raw:
        return _parse_matrix_full_schema(raw["roles"])

    # Flat format: {"role": {"model_id": rank, ...}}
    if isinstance(raw, dict):
        result: dict[str, dict[str, int]] = {}
        for role, entries in raw.items():
            if isinstance(entries, dict):
                result[role] = {k: int(v) for k, v in entries.items()}
        return result

    return {}


def _parse_matrix_full_schema(
    roles_raw: dict[str, Any],
) -> dict[str, dict[str, int]]:
    """Parse full schema format: roles -> list of {model_id, rank} dicts."""
    assert isinstance(roles_raw, dict), "roles_raw must be a dict"

    matrix: dict[str, dict[str, int]] = {}
    for role, entries in roles_raw.items():
        if isinstance(entries, list):
            matrix[role] = {e["model_id"]: int(e["rank"]) for e in entries}
        elif isinstance(entries, dict):
            matrix[role] = {k: int(v) for k, v in entries.items()}
    return matrix


# ---------------------------------------------------------------------------
# 7. write_matrix — serialize and write updated matrix
# ---------------------------------------------------------------------------


def write_matrix(
    matrix: dict[str, dict[str, int]],
    state_dir: Path,
) -> None:
    """Serialize and write the role matrix to state_dir/model_role_matrix.json.

    Writes the full schema format with version and roles structure.
    Sorts roles and model entries by rank descending for readability.

    Args:
        matrix: role -> {model_id: rank} mapping
        state_dir: router state directory
    """
    assert isinstance(matrix, dict), "matrix must be a dict"
    assert isinstance(state_dir, Path), "state_dir must be a Path"

    # Build full schema format
    roles_out: dict[str, list[dict[str, Any]]] = {}
    for role in sorted(matrix.keys()):
        model_ranks = matrix[role]
        # Sort by rank descending, then model_id for stability
        entries = sorted(
            model_ranks.items(),
            key=lambda x: (-x[1], x[0]),
        )
        roles_out[role] = [{"model_id": mid, "rank": rank} for mid, rank in entries]

    payload: dict[str, Any] = {
        "version": _MATRIX_SCHEMA_VERSION,
        "default_rank": _DEFAULT_RANK,
        "roles": roles_out,
    }

    matrix_path = state_dir / "model_role_matrix.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    matrix_path.write_text(json.dumps(payload, indent=2))

    logger.info(
        "matrix_written",
        path=str(matrix_path),
        roles=sorted(matrix.keys()),
        total_models=sum(len(v) for v in matrix.values()),
    )


# ---------------------------------------------------------------------------
# 8. update_matrix_from_spectrography — main entry point
# ---------------------------------------------------------------------------


def update_matrix_from_spectrography(
    state_dir: Path,
    spectrography_dir: Path | None = None,
    *,
    blend_weight: float = 0.7,
) -> dict[str, dict[str, int]]:
    """Load latest spectrography results and blend into the role matrix.

    Steps:
      1. Find the latest spectrography run in spectrography_dir
         (defaults to state_dir / "spectrography")
      2. Load ModelSpectrographProfile instances from the run
      3. Convert profiles to role matrix rankings
      4. Load existing matrix from state_dir
      5. Blend: new_rank = blend_weight * empirical + (1 - blend_weight) * existing
      6. Write updated matrix to state_dir
      7. Return the updated matrix dict

    Graceful no-op if no spectrography data is found — existing matrix is
    returned unchanged.

    Args:
        state_dir: router state directory (contains model_role_matrix.json)
        spectrography_dir: directory containing spectrography run outputs
                           (default: state_dir / "spectrography")
        blend_weight: empirical weight in [0.0, 1.0] (default 0.7)

    Returns:
        Updated matrix as dict[role, dict[model_id, rank]]
    """
    assert isinstance(state_dir, Path), "state_dir must be a Path"
    assert 0.0 <= blend_weight <= 1.0, f"blend_weight must be in [0.0, 1.0], got {blend_weight}"

    output_dir = (
        spectrography_dir if spectrography_dir is not None else (state_dir / "spectrography")
    )

    logger.info(
        "matrix_update_starting",
        state_dir=str(state_dir),
        spectrography_dir=str(output_dir),
        blend_weight=blend_weight,
    )

    # Step 1: Find latest run
    run_dir = find_latest_spectrography_run(output_dir)
    if run_dir is None:
        logger.warning(
            "no_spectrography_data_found",
            output_dir=str(output_dir),
        )
        existing = load_existing_matrix(state_dir)
        return existing

    # Step 2: Load profiles
    profiles = load_profiles_from_run(run_dir)
    if not profiles:
        logger.warning(
            "no_profiles_in_run_dir",
            run_dir=str(run_dir),
        )
        existing = load_existing_matrix(state_dir)
        return existing

    # Step 3: Convert to role matrix rankings
    empirical_matrix = rankings_to_matrix(profiles)

    # Step 4: Load existing matrix
    existing_matrix = load_existing_matrix(state_dir)

    # Step 5: Blend per role
    blended: dict[str, dict[str, int]] = {}

    # Collect all roles from both empirical and existing
    all_roles = set(empirical_matrix.keys()) | set(existing_matrix.keys())

    for role in all_roles:
        emp_ranks = empirical_matrix.get(role, {})
        ex_ranks = existing_matrix.get(role, {})
        blended[role] = blend_ranks(emp_ranks, ex_ranks, blend_weight)

    # Step 6: Write
    write_matrix(blended, state_dir)

    logger.info(
        "matrix_update_complete",
        run_dir=str(run_dir),
        roles_updated=sorted(all_roles),
        total_models=sum(len(v) for v in blended.values()),
        blend_weight=blend_weight,
    )

    return blended
