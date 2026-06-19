"""Comprehensive tests for spectrography/analyzer.py — covering all missed lines.

Spec: model-spectrography-v0.1.0-spec
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    FlavorScore,
    ModelFlavorProfile,
)
from dragonlight_router.spectrography.analyzer import (
    DimensionStats,
    ProbeResult,
    RawFingerprint,
    _load_declared_profiles,
    _parse_declared_profile,
    _rank_normalize_dimension,
    compute_calibration_deltas,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probe_result(
    model_id: str = "test/model-a",
    probe_id: str = "disc-test-001",
    task_type: str = "generation",
    domain: str = "code",
    quality_speed: str = "quality",
    normalized_score: float = 0.8,
    error: str | None = None,
) -> ProbeResult:
    """Build a ProbeResult with sensible defaults."""
    return ProbeResult(
        model_id=model_id,
        probe_id=probe_id,
        task_type=task_type,
        domain=domain,
        quality_speed=quality_speed,
        normalized_score=normalized_score,
        judge_scores={"accuracy": 4, "completeness": 3, "clarity": 4, "relevance": 4},
        is_self_eval=False,
        error=error,
    )


def _make_flavor_profile(
    model_id: str = "test/model-a",
    score: float = 0.7,
) -> ModelFlavorProfile:
    """Build a ModelFlavorProfile with uniform scores."""
    fs = FlavorScore(score=score, confidence=0.8, sample_count=5)
    return ModelFlavorProfile(
        model_id=model_id,
        version=1,
        updated_at=datetime.now(UTC).isoformat(),
        task_scores=dict.fromkeys(IBR_TASK_TYPES, fs),
        domain_scores=dict.fromkeys(IBR_DOMAINS, fs),
        qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, fs),
    )


# ===========================================================================
# _rank_normalize_dimension — lines 266-267 (empty model_means branch)
# ===========================================================================


class TestRankNormalizeDimensionEmpty:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_empty_model_means_returns_empty(self) -> None:
        # No models have the dimension key "nonexistent"
        fp = RawFingerprint(
            model_id="m1",
            task_scores={"generation": DimensionStats(mean=0.8, stddev=0.1, count=3)},
            domain_scores={},
            qs_scores={},
        )
        raw = {"m1": fp}
        result = _rank_normalize_dimension(
            raw,
            "task_scores",
            {"nonexistent"},
        )
        assert result["nonexistent"] == {}


# ===========================================================================
# compute_calibration_deltas — lines 350-354 (model not in declared)
# ===========================================================================


class TestCalibrationDeltasModelNotDeclared:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_skips_undeclared_models(self, tmp_path: Path) -> None:
        # Write a declared profile for m2 only
        declared_yaml = {
            "profiles": {
                "m2": {
                    "task_scores": {"generation": 0.7},
                    "domain_scores": {},
                    "qs_scores": {},
                },
            },
        }
        declared_path = tmp_path / "declared.yaml"
        declared_path.write_text(yaml.dump(declared_yaml))

        # Empirical has m1 and m2
        empirical = {
            "m1": _make_flavor_profile("m1"),
            "m2": _make_flavor_profile("m2"),
        }
        result = compute_calibration_deltas(empirical, declared_path)
        # m1 should be skipped (not in declared)
        assert "m1" not in result
        assert "m2" in result


# ===========================================================================
# _load_declared_profiles — lines 399-405 (YAML load failure),
#   411-412 (invalid profiles format)
# ===========================================================================


class TestLoadDeclaredProfiles:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_invalid_yaml_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("%invalid yaml directive")
        result = _load_declared_profiles(path)
        assert result == {}

    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = _load_declared_profiles(tmp_path / "no-such-file.yaml")
        assert result == {}

    def test_profiles_not_dict_returns_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_profiles.yaml"
        yaml_data = {"profiles": "not-a-dict"}
        path.write_text(yaml.dump(yaml_data))
        result = _load_declared_profiles(path)
        assert result == {}

    def test_valid_profiles_parsed(self, tmp_path: Path) -> None:
        path = tmp_path / "valid.yaml"
        yaml_data = {
            "profiles": {
                "m1": {
                    "task_scores": {"generation": 0.8},
                    "domain_scores": {"code": 0.9},
                    "qs_scores": {"quality": 0.7},
                },
            },
        }
        path.write_text(yaml.dump(yaml_data))
        result = _load_declared_profiles(path)
        assert "m1" in result
        assert result["m1"].task_scores["generation"].score == pytest.approx(
            0.8,
        )


# ===========================================================================
# _parse_declared_profile — lines 431-432 (non-dict raw)
# ===========================================================================


class TestParseDeclaredProfileInvalid:
    """Spec: model-spectrography-v0.1.0-spec"""

    def test_non_dict_raw_returns_none(self) -> None:
        result = _parse_declared_profile("m1", "not-a-dict")  # type: ignore[arg-type]
        assert result is None

    def test_list_raw_returns_none(self) -> None:
        result = _parse_declared_profile("m1", [1, 2])  # type: ignore[arg-type]
        assert result is None

    def test_valid_dict_returns_profile(self) -> None:
        raw: dict[str, Any] = {
            "version": 1,
            "task_scores": {"generation": 0.8},
            "domain_scores": {},
            "qs_scores": {},
        }
        result = _parse_declared_profile("m1", raw)
        assert result is not None
        assert result.model_id == "m1"
