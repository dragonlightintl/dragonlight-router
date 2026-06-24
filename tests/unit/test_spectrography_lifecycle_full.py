"""Comprehensive tests for spectrography/lifecycle.py — covering all missed lines.

Spec: model-spectrography-v0.1.0-spec
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    ModelSpectrographProfile,
    SpectrographScore,
)
from dragonlight_router.spectrography.lifecycle import (
    _collect_model_ids_from_matrix,
    _parse_single_profile,
    apply_spectrography_decay,
    check_staleness,
    get_models_needing_spectrography,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectrograph_profile(
    model_id: str = "test/model-a",
    updated_at: str | None = None,
    score: float = 0.7,
) -> ModelSpectrographProfile:
    """Build a ModelSpectrographProfile with uniform scores."""
    if updated_at is None:
        updated_at = datetime.now(UTC).isoformat()
    fs = SpectrographScore(score=score, confidence=0.8, sample_count=5)
    return ModelSpectrographProfile(
        model_id=model_id,
        version=1,
        updated_at=updated_at,
        task_scores=dict.fromkeys(IBR_TASK_TYPES, fs),
        domain_scores=dict.fromkeys(IBR_DOMAINS, fs),
        qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, fs),
    )


# ===========================================================================
# check_staleness — line 80 (naive datetime branch)
# ===========================================================================


class TestCheckStalenessNaive:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_naive_datetime_treated_as_utc(self) -> None:
        # Create a profile with a naive datetime (no timezone info)
        naive_old = (datetime.now(UTC) - timedelta(days=60)).strftime(
            "%Y-%m-%dT%H:%M:%S",
        )
        profile = _make_spectrograph_profile(
            model_id="m1",
            updated_at=naive_old,
        )
        results = check_staleness({"m1": profile})
        assert len(results) == 1
        assert results[0].needs_refresh is True
        assert results[0].age_days > 30


# ===========================================================================
# apply_spectrography_decay — line 127 (naive updated_at branch)
# ===========================================================================


class TestDecayNaiveDatetime:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_naive_updated_at_treated_as_utc(self) -> None:
        # Create a profile with naive datetime string (no +00:00 suffix)
        naive_old = (datetime.now(UTC) - timedelta(days=60)).strftime(
            "%Y-%m-%dT%H:%M:%S",
        )
        profile = _make_spectrograph_profile(
            model_id="m1",
            updated_at=naive_old,
            score=0.9,
        )
        result = apply_spectrography_decay(profile)
        # Should decay since profile is > 30 days old
        for t in IBR_TASK_TYPES:
            assert result.task_scores[t].score < 0.9


# ===========================================================================
# _parse_single_profile — lines 303-304 (non-dict raw)
# ===========================================================================


class TestParseSingleProfileInvalid:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_non_dict_raw_returns_none(self) -> None:
        result = _parse_single_profile("m1", "not-a-dict")  # type: ignore[arg-type]
        assert result is None

    def test_valid_dict_returns_profile(self) -> None:
        raw = {
            "version": 1,
            "task_scores": {"generation": 0.8},
            "domain_scores": {},
            "qs_scores": {},
        }
        result = _parse_single_profile("m1", raw)
        assert result is not None
        assert result.model_id == "m1"
        assert result.task_scores["generation"].score == pytest.approx(0.8)

    def test_list_raw_returns_none(self) -> None:
        result = _parse_single_profile("m1", [1, 2, 3])  # type: ignore[arg-type]
        assert result is None


# ===========================================================================
# get_models_needing_spectrography — line 400 (naive updated_at)
# ===========================================================================


class TestModelsNeedingSpectrographyNaive:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_naive_updated_at_treated_as_utc(self, tmp_path: Path) -> None:
        matrix = {
            "roles": {
                "coding": [{"model_id": "m1", "rank": 90}],
            },
        }
        matrix_path = tmp_path / "matrix.json"
        matrix_path.write_text(json.dumps(matrix))

        # Naive datetime 60 days ago => stale
        naive_old = (datetime.now(UTC) - timedelta(days=60)).strftime(
            "%Y-%m-%dT%H:%M:%S",
        )
        existing = {
            "m1": _make_spectrograph_profile(model_id="m1", updated_at=naive_old),
        }
        result = get_models_needing_spectrography(
            matrix_path,
            existing,
            staleness_days=30,
        )
        assert "m1" in result


# ===========================================================================
# _collect_model_ids_from_matrix — lines 430-436 (load failure),
#   446 (skip version/default_rank), 451-453 (dict entries format)
# ===========================================================================


class TestCollectModelIdsFromMatrix:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_invalid_json_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "matrix.json"
        path.write_text("not valid json{{{")
        result = _collect_model_ids_from_matrix(path)
        assert result == set()

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = _collect_model_ids_from_matrix(
            tmp_path / "nonexistent.json",
        )
        assert result == set()

    def test_skips_version_and_default_rank_keys(
        self,
        tmp_path: Path,
    ) -> None:
        data = {
            "version": 1,
            "default_rank": 50,
            "roles": {
                "coding": [{"model_id": "m1", "rank": 90}],
            },
        }
        path = tmp_path / "matrix.json"
        path.write_text(json.dumps(data))
        result = _collect_model_ids_from_matrix(path)
        assert "m1" in result
        assert "1" not in result
        assert "50" not in result

    def test_dict_format_entries(self, tmp_path: Path) -> None:
        # Flat dict format: {"coding": {"m1": 90, "m2": 80}}
        data = {
            "roles": {
                "coding": {"m1": 90, "m2": 80},
            },
        }
        path = tmp_path / "matrix.json"
        path.write_text(json.dumps(data))
        result = _collect_model_ids_from_matrix(path)
        assert "m1" in result
        assert "m2" in result

    def test_flat_dict_without_roles_key(self, tmp_path: Path) -> None:
        # No "roles" key — falls back to raw
        data = {
            "coding": [{"model_id": "m1", "rank": 90}],
        }
        path = tmp_path / "matrix.json"
        path.write_text(json.dumps(data))
        result = _collect_model_ids_from_matrix(path)
        assert "m1" in result

    def test_version_in_roles_data_skipped(self, tmp_path: Path) -> None:
        # "version" key in the roles_data dict should be skipped
        data = {
            "version": 2,
            "coding": [{"model_id": "m1", "rank": 90}],
        }
        path = tmp_path / "matrix.json"
        path.write_text(json.dumps(data))
        result = _collect_model_ids_from_matrix(path)
        assert "m1" in result
