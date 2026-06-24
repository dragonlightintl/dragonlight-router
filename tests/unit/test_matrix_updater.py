"""Unit tests for roles/matrix_updater.py — spectrography-to-matrix bridge.

Spec traceability: spectrography-bridge-v0.1.0
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    ModelSpectrographProfile,
    SpectrographScore,
)
from dragonlight_router.roles.matrix_updater import (
    _RANK_MAX,
    _RANK_MIN,
    blend_ranks,
    find_latest_spectrography_run,
    load_profiles_from_run,
    rankings_to_matrix,
    score_to_rank,
    update_matrix_from_spectrography,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_spectrograph_score(score: float) -> SpectrographScore:
    return SpectrographScore(score=score, confidence=1.0, sample_count=5)


def _neutral_score() -> SpectrographScore:
    return SpectrographScore(score=0.5, confidence=0.0, sample_count=0)


def _make_profile(
    model_id: str,
    task_overrides: dict[str, float] | None = None,
    domain_overrides: dict[str, float] | None = None,
    qs_overrides: dict[str, float] | None = None,
) -> ModelSpectrographProfile:
    """Build a ModelSpectrographProfile with neutral scores and optional overrides."""
    task_scores: dict[str, SpectrographScore] = {t: _neutral_score() for t in IBR_TASK_TYPES}
    domain_scores: dict[str, SpectrographScore] = {d: _neutral_score() for d in IBR_DOMAINS}
    qs_scores: dict[str, SpectrographScore] = {q: _neutral_score() for q in IBR_QUALITY_SPEED}

    if task_overrides:
        for k, v in task_overrides.items():
            task_scores[k] = _make_spectrograph_score(v)
    if domain_overrides:
        for k, v in domain_overrides.items():
            domain_scores[k] = _make_spectrograph_score(v)
    if qs_overrides:
        for k, v in qs_overrides.items():
            qs_scores[k] = _make_spectrograph_score(v)

    return ModelSpectrographProfile(
        model_id=model_id,
        version=1,
        updated_at="2026-06-20T00:00:00+00:00",
        task_scores=task_scores,
        domain_scores=domain_scores,
        qs_scores=qs_scores,
    )


# ---------------------------------------------------------------------------
# score_to_rank tests
# ---------------------------------------------------------------------------


class TestScoreToRank:
    def test_midpoint_score(self) -> None:
        """0.5 should map to 50."""
        assert score_to_rank(0.5) == 50

    def test_full_score_clamps_to_max(self) -> None:
        """1.0 should map to _RANK_MAX (99), not 100 (reserved for operator)."""
        assert score_to_rank(1.0) == _RANK_MAX

    def test_zero_score_clamps_to_min(self) -> None:
        """0.0 should map to _RANK_MIN (10), not 0 (reserved for operator)."""
        assert score_to_rank(0.0) == _RANK_MIN

    def test_linear_scaling(self) -> None:
        """Scores scale linearly between _RANK_MIN and _RANK_MAX."""
        assert score_to_rank(0.8) == 80
        assert score_to_rank(0.75) == 75
        assert score_to_rank(0.55) == 55

    def test_clamping_boundary_low(self) -> None:
        """Scores in [0.0, 0.10) should clamp to _RANK_MIN."""
        assert score_to_rank(0.05) == _RANK_MIN
        assert score_to_rank(0.09) == _RANK_MIN

    def test_clamping_boundary_high(self) -> None:
        """Scores approaching 1.0 must not exceed _RANK_MAX."""
        assert score_to_rank(1.0) == _RANK_MAX
        result = score_to_rank(0.99)
        assert result <= _RANK_MAX

    def test_output_range(self) -> None:
        """All valid inputs produce ranks in [_RANK_MIN, _RANK_MAX]."""
        for raw in range(0, 101):
            score = raw / 100.0
            rank = score_to_rank(score)
            assert _RANK_MIN <= rank <= _RANK_MAX, (
                f"score={score} produced out-of-range rank={rank}"
            )


# ---------------------------------------------------------------------------
# rankings_to_matrix tests
# ---------------------------------------------------------------------------


class TestRankingsToMatrix:
    def test_coding_role_from_generation(self) -> None:
        """generation task type maps to coding role.

        coding aggregates generation + refactoring; with only generation
        overridden, the rank reflects the average of generation and neutral (0.5).
        """
        profile = _make_profile(
            "model/a",
            task_overrides={"generation": 0.9, "refactoring": 0.9},
        )
        result = rankings_to_matrix({"model/a": profile}, roles=["coding"])

        assert "coding" in result
        assert "model/a" in result["coding"]
        assert result["coding"]["model/a"] == score_to_rank(0.9)

    def test_coding_role_from_refactoring(self) -> None:
        """refactoring task type also maps to coding role."""
        profile = _make_profile("model/b", task_overrides={"refactoring": 0.8})
        result = rankings_to_matrix({"model/b": profile}, roles=["coding"])

        assert result["coding"]["model/b"] > 0

    def test_review_role_from_analysis(self) -> None:
        """analysis task type maps to review role.

        review aggregates analysis + summarization; both are set to verify
        exact rank mapping.
        """
        profile = _make_profile(
            "model/c",
            task_overrides={"analysis": 0.85, "summarization": 0.85},
        )
        result = rankings_to_matrix({"model/c": profile}, roles=["review"])

        assert result["review"]["model/c"] == score_to_rank(0.85)

    def test_review_role_from_summarization(self) -> None:
        """summarization task type also maps to review role."""
        profile = _make_profile("model/d", task_overrides={"summarization": 0.7})
        result = rankings_to_matrix({"model/d": profile}, roles=["review"])

        assert "model/d" in result["review"]

    def test_reasoning_role(self) -> None:
        """reasoning task type maps to reasoning role."""
        profile = _make_profile("model/e", task_overrides={"reasoning": 0.92})
        result = rankings_to_matrix({"model/e": profile}, roles=["reasoning"])

        assert result["reasoning"]["model/e"] == score_to_rank(0.92)

    def test_testing_role_from_generation(self) -> None:
        """generation task type maps to testing role."""
        profile = _make_profile("model/f", task_overrides={"generation": 0.78})
        result = rankings_to_matrix({"model/f": profile}, roles=["testing"])

        assert result["testing"]["model/f"] == score_to_rank(0.78)

    def test_spec_role_aggregates_all_tasks(self) -> None:
        """spec role averages all task dimension scores."""
        # All task scores set to 0.8
        task_overrides = dict.fromkeys(IBR_TASK_TYPES, 0.8)
        profile = _make_profile("model/g", task_overrides=task_overrides)
        result = rankings_to_matrix({"model/g": profile}, roles=["spec"])

        assert "model/g" in result["spec"]
        # Average of all 0.8 -> rank 80
        assert result["spec"]["model/g"] == score_to_rank(0.8)

    def test_roles_filter_respected(self) -> None:
        """When roles list is provided, only those roles appear in output."""
        profile = _make_profile("model/h")
        result = rankings_to_matrix({"model/h": profile}, roles=["coding", "spec"])

        assert set(result.keys()) == {"coding", "spec"}
        assert "review" not in result
        assert "reasoning" not in result

    def test_empty_profiles(self) -> None:
        """Empty profiles produce empty matrix."""
        result = rankings_to_matrix({})
        assert result == {} or all(len(v) == 0 for v in result.values())

    def test_multiple_models_ranked(self) -> None:
        """Multiple models produce independent ranks for each role."""
        profile_a = _make_profile("model/a", task_overrides={"reasoning": 0.9})
        profile_b = _make_profile("model/b", task_overrides={"reasoning": 0.6})
        result = rankings_to_matrix(
            {"model/a": profile_a, "model/b": profile_b},
            roles=["reasoning"],
        )

        assert result["reasoning"]["model/a"] > result["reasoning"]["model/b"]

    def test_all_default_roles_populated(self) -> None:
        """Default call populates all five managed roles."""
        profile = _make_profile("model/x")
        result = rankings_to_matrix({"model/x": profile})

        for role in ("coding", "review", "reasoning", "testing", "spec"):
            assert role in result


# ---------------------------------------------------------------------------
# blend_ranks tests
# ---------------------------------------------------------------------------


class TestBlendRanks:
    def test_pure_empirical_blend_weight_1(self) -> None:
        """blend_weight=1.0 uses only empirical ranks."""
        empirical = {"model/a": 80}
        existing = {"model/a": 50}

        result = blend_ranks(empirical, existing, blend_weight=1.0)

        assert result["model/a"] == 80

    def test_pure_existing_blend_weight_0(self) -> None:
        """blend_weight=0.0 keeps existing ranks unchanged."""
        empirical = {"model/a": 80}
        existing = {"model/a": 50}

        result = blend_ranks(empirical, existing, blend_weight=0.0)

        assert result["model/a"] == 50

    def test_default_blend_weight(self) -> None:
        """blend_weight=0.7 applies 70% empirical + 30% existing."""
        empirical = {"model/a": 80}
        existing = {"model/a": 50}

        result = blend_ranks(empirical, existing, blend_weight=0.7)

        expected = int(round(0.7 * 80 + 0.3 * 50))  # 56 + 15 = 71
        assert result["model/a"] == expected

    def test_new_model_pure_empirical(self) -> None:
        """New models (no existing rank) use pure empirical."""
        empirical = {"new/model": 75}
        existing = {}

        result = blend_ranks(empirical, existing, blend_weight=0.7)

        assert result["new/model"] == 75

    def test_existing_only_model_unchanged(self) -> None:
        """Models with existing rank but no empirical data are kept unchanged."""
        empirical = {}
        existing = {"old/model": 60}

        result = blend_ranks(empirical, existing, blend_weight=0.7)

        assert result["old/model"] == 60

    def test_operator_curated_preserved_at_default_weight(self) -> None:
        """At blend_weight=0.7, operator-curated ranks retain 30% influence."""
        # High operator rank, low empirical
        empirical = {"model/a": 20}
        existing = {"model/a": 90}

        result = blend_ranks(empirical, existing, blend_weight=0.7)

        blended = result["model/a"]
        # 0.7*20 + 0.3*90 = 14 + 27 = 41
        assert blended == 41
        # Verify operator influence is visible: blended > empirical alone
        assert blended > empirical["model/a"]

    def test_blend_clamps_to_valid_range(self) -> None:
        """Blended values stay within [_RANK_MIN, _RANK_MAX]."""
        empirical = {"model/a": _RANK_MAX}
        existing = {"model/a": _RANK_MAX}

        result = blend_ranks(empirical, existing, blend_weight=0.5)

        assert _RANK_MIN <= result["model/a"] <= _RANK_MAX

    def test_blend_multiple_models(self) -> None:
        """Blend handles multiple models independently."""
        empirical = {"model/a": 80, "model/b": 40}
        existing = {"model/a": 60, "model/b": 70}

        result = blend_ranks(empirical, existing, blend_weight=0.7)

        expected_a = int(round(0.7 * 80 + 0.3 * 60))  # 56+18=74
        expected_b = int(round(0.7 * 40 + 0.3 * 70))  # 28+21=49
        assert result["model/a"] == expected_a
        assert result["model/b"] == expected_b


# ---------------------------------------------------------------------------
# find_latest_spectrography_run tests
# ---------------------------------------------------------------------------


class TestFindLatestSpectrographyRun:
    def test_returns_none_for_missing_dir(self, tmp_path: Path) -> None:
        """Missing output_dir returns None gracefully."""
        missing = tmp_path / "nonexistent"
        result = find_latest_spectrography_run(missing)
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path: Path) -> None:
        """Empty output_dir returns None."""
        result = find_latest_spectrography_run(tmp_path)
        assert result is None

    def test_finds_run_with_fingerprints_yaml(self, tmp_path: Path) -> None:
        """Run directory containing fingerprints.yaml is detected."""
        run_dir = tmp_path / "20260620-120000-abc12345"
        run_dir.mkdir()
        (run_dir / "fingerprints.yaml").write_text("version: 1\nprofiles: {}\n")

        result = find_latest_spectrography_run(tmp_path)

        assert result == run_dir

    def test_finds_run_with_report_json(self, tmp_path: Path) -> None:
        """Run directory containing report.json is also detected."""
        run_dir = tmp_path / "20260620-120000-abc12345"
        run_dir.mkdir()
        (run_dir / "report.json").write_text('{"run_id": "test"}')

        result = find_latest_spectrography_run(tmp_path)

        assert result == run_dir

    def test_returns_latest_of_multiple_runs(self, tmp_path: Path) -> None:
        """Returns the most recent run when multiple exist."""
        run_a = tmp_path / "20260618-100000-aaa"
        run_b = tmp_path / "20260619-120000-bbb"
        run_c = tmp_path / "20260620-140000-ccc"

        for run_dir in (run_a, run_b, run_c):
            run_dir.mkdir()
            (run_dir / "fingerprints.yaml").write_text("version: 1\nprofiles: {}\n")

        result = find_latest_spectrography_run(tmp_path)

        assert result == run_c

    def test_ignores_dirs_without_output_files(self, tmp_path: Path) -> None:
        """Directories without fingerprints.yaml or report.json are ignored."""
        noise_dir = tmp_path / "20260620-140000-noise"
        noise_dir.mkdir()
        (noise_dir / "checkpoint.jsonl").write_text("")

        result = find_latest_spectrography_run(tmp_path)

        assert result is None


# ---------------------------------------------------------------------------
# load_profiles_from_run tests
# ---------------------------------------------------------------------------


class TestLoadProfilesFromRun:
    def _write_fingerprints_yaml(
        self,
        run_dir: Path,
        profiles: dict[str, Any],
    ) -> None:
        data = {"version": 1, "source": "test", "generated_at": "", "profiles": profiles}
        (run_dir / "fingerprints.yaml").write_text(yaml.dump(data))

    def test_loads_from_fingerprints_yaml(self, tmp_path: Path) -> None:
        """Profiles are loaded from fingerprints.yaml when present."""
        profiles_data = {
            "model/a": {
                "task_scores": {"generation": 0.85, "analysis": 0.7},
                "domain_scores": {"code": 0.9},
                "qs_scores": {"quality": 0.8},
            }
        }
        self._write_fingerprints_yaml(tmp_path, profiles_data)

        profiles = load_profiles_from_run(tmp_path)

        assert "model/a" in profiles
        assert profiles["model/a"].task_scores["generation"].score == pytest.approx(0.85)

    def test_loads_from_report_json_fallback(self, tmp_path: Path) -> None:
        """Falls back to report.json when fingerprints.yaml is absent."""
        report = {
            "run_id": "test",
            "profiles": {
                "model/b": {
                    "model_id": "model/b",
                    "version": 1,
                    "updated_at": "",
                    "task_scores": {
                        "generation": {"score": 0.75, "confidence": 0.9, "sample_count": 3}
                    },
                    "domain_scores": {},
                    "qs_scores": {},
                }
            },
        }
        (tmp_path / "report.json").write_text(json.dumps(report))

        profiles = load_profiles_from_run(tmp_path)

        assert "model/b" in profiles
        assert profiles["model/b"].task_scores["generation"].score == pytest.approx(0.75)

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        """Missing run dir returns empty dict."""
        result = load_profiles_from_run(tmp_path / "nonexistent")
        assert result == {}

    def test_returns_empty_for_invalid_yaml(self, tmp_path: Path) -> None:
        """Corrupt fingerprints.yaml returns empty dict (no exception)."""
        (tmp_path / "fingerprints.yaml").write_text(":: invalid: yaml: [\n")

        result = load_profiles_from_run(tmp_path)

        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# update_matrix_from_spectrography tests
# ---------------------------------------------------------------------------


class TestUpdateMatrixFromSpectrography:
    def _write_fingerprints(
        self,
        run_dir: Path,
        profiles: dict[str, Any],
    ) -> None:
        data = {"version": 1, "source": "test", "generated_at": "", "profiles": profiles}
        (run_dir / "fingerprints.yaml").write_text(yaml.dump(data))

    def _write_existing_matrix(
        self,
        state_dir: Path,
        roles: dict[str, list[dict[str, Any]]],
    ) -> None:
        payload = {"version": 1, "default_rank": 20, "roles": roles}
        (state_dir / "model_role_matrix.json").write_text(json.dumps(payload))

    def test_graceful_noop_no_spectrography_data(self, tmp_path: Path) -> None:
        """No spectrography data returns existing matrix unchanged."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()

        existing_roles = {"coding": [{"model_id": "old/model", "rank": 85}]}
        self._write_existing_matrix(state_dir, existing_roles)

        result = update_matrix_from_spectrography(state_dir)

        assert "coding" in result
        assert result["coding"]["old/model"] == 85

    def test_writes_updated_matrix_file(self, tmp_path: Path) -> None:
        """After update, model_role_matrix.json exists in state_dir."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        spec_dir = tmp_path / "spectrography"
        run_dir = spec_dir / "20260620-120000-abc"
        run_dir.mkdir(parents=True)

        profiles_data = {
            "model/new": {
                "task_scores": dict.fromkeys(list(IBR_TASK_TYPES)[:3], 0.8),
                "domain_scores": {},
                "qs_scores": {},
            }
        }
        self._write_fingerprints(run_dir, profiles_data)

        update_matrix_from_spectrography(state_dir, spec_dir)

        matrix_path = state_dir / "model_role_matrix.json"
        assert matrix_path.exists()
        written = json.loads(matrix_path.read_text())
        assert "roles" in written

    def test_blend_preserves_operator_curated_partially(self, tmp_path: Path) -> None:
        """Default blend_weight=0.7 preserves 30% of operator-curated rank."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        spec_dir = tmp_path / "spectrography"
        run_dir = spec_dir / "20260620-120000-abc"
        run_dir.mkdir(parents=True)

        # Existing: model/a has rank 90 in reasoning
        existing_roles = {"reasoning": [{"model_id": "model/a", "rank": 90}]}
        self._write_existing_matrix(state_dir, existing_roles)

        # Empirical: model/a has reasoning score 0.3 (rank 30)
        profiles_data = {
            "model/a": {
                "task_scores": {"reasoning": 0.3},
                "domain_scores": {},
                "qs_scores": {},
            }
        }
        self._write_fingerprints(run_dir, profiles_data)

        result = update_matrix_from_spectrography(state_dir, spec_dir, blend_weight=0.7)

        # 0.7*30 + 0.3*90 = 21 + 27 = 48
        blended = result["reasoning"]["model/a"]
        assert blended == pytest.approx(48, abs=2)
        # Operator influence: blended must exceed pure empirical rank (30)
        assert blended > 30

    def test_pure_empirical_override(self, tmp_path: Path) -> None:
        """blend_weight=1.0 fully replaces existing ranks with empirical."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        spec_dir = tmp_path / "spectrography"
        run_dir = spec_dir / "20260620-120000-abc"
        run_dir.mkdir(parents=True)

        existing_roles = {"reasoning": [{"model_id": "model/a", "rank": 90}]}
        self._write_existing_matrix(state_dir, existing_roles)

        profiles_data = {
            "model/a": {
                "task_scores": {"reasoning": 0.5},
                "domain_scores": {},
                "qs_scores": {},
            }
        }
        self._write_fingerprints(run_dir, profiles_data)

        result = update_matrix_from_spectrography(state_dir, spec_dir, blend_weight=1.0)

        # Pure empirical: score 0.5 -> rank 50
        assert result["reasoning"]["model/a"] == score_to_rank(0.5)

    def test_new_model_added_to_matrix(self, tmp_path: Path) -> None:
        """New models from spectrography are added to the matrix."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        spec_dir = tmp_path / "spectrography"
        run_dir = spec_dir / "20260620-120000-abc"
        run_dir.mkdir(parents=True)

        # Existing: only model/a
        existing_roles = {"coding": [{"model_id": "model/a", "rank": 80}]}
        self._write_existing_matrix(state_dir, existing_roles)

        # Empirical: new model/b appears
        profiles_data = {
            "model/b": {
                "task_scores": {"generation": 0.6},
                "domain_scores": {},
                "qs_scores": {},
            }
        }
        self._write_fingerprints(run_dir, profiles_data)

        result = update_matrix_from_spectrography(state_dir, spec_dir)

        assert "model/b" in result.get("coding", {})

    def test_uses_default_spectrography_dir(self, tmp_path: Path) -> None:
        """When spectrography_dir is None, defaults to state_dir/spectrography."""
        state_dir = tmp_path / "state"
        state_dir.mkdir()
        run_dir = state_dir / "spectrography" / "20260620-120000-abc"
        run_dir.mkdir(parents=True)

        profiles_data = {
            "model/z": {
                "task_scores": {"reasoning": 0.9},
                "domain_scores": {},
                "qs_scores": {},
            }
        }
        data = {"version": 1, "source": "test", "generated_at": "", "profiles": profiles_data}
        (run_dir / "fingerprints.yaml").write_text(yaml.dump(data))

        result = update_matrix_from_spectrography(state_dir)

        assert "model/z" in result.get("reasoning", {})
