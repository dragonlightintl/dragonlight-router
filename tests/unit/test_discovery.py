"""Tests for discovery/ — probes, analyzer, lifecycle, and runner.

Spec traceability: model-flavor-discovery-v0.1.0-spec.md
AC numbers: IBR-FLV-01 through IBR-FLV-06.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_FLAVOR,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    FlavorScore,
    ModelFlavorProfile,
)
from dragonlight_router.discovery.analyzer import (
    CalibrationDelta,
    DimensionStats,
    ProbeResult,
    RawFingerprint,
    _delta_recommendation,
    aggregate_scores,
    build_fingerprints_yaml,
    build_model_rankings,
    compute_calibration_deltas,
    rank_normalize,
)
from dragonlight_router.discovery.lifecycle import (
    StaleProfile,
    apply_discovery_decay,
    check_staleness,
    get_models_needing_discovery,
    load_existing_fingerprints,
    merge_incremental,
    write_fingerprints_yaml,
)
from dragonlight_router.discovery.probes import (
    DIFFICULTY_LEVELS,
    DISCRIMINATION_AXES,
    DiscoveryProbe,
    _validate_probe,
    get_all_probes,
    get_probes_by_axis,
    get_probes_by_domain,
    get_probes_by_task_type,
)
from dragonlight_router.discovery.runner import (
    ProviderPacer,
    _append_checkpoint,
    _create_adapter,
    _extract_provider,
    _interleaved_schedule,
    _load_checkpoint,
    _parse_delays,
)


# ---------------------------------------------------------------------------
# Helpers — build test data
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
        judge_scores={"correctness": 4, "style": 3},
        is_self_eval=False,
        error=error,
    )


def _make_flavor_profile(
    model_id: str = "test/model-a",
    updated_at: str | None = None,
    score: float = 0.7,
    confidence: float = 0.8,
    sample_count: int = 5,
) -> ModelFlavorProfile:
    """Build a ModelFlavorProfile with uniform scores across all dimensions."""
    if updated_at is None:
        updated_at = datetime.now(UTC).isoformat()
    fs = FlavorScore(score=score, confidence=confidence, sample_count=sample_count)
    return ModelFlavorProfile(
        model_id=model_id,
        version=1,
        updated_at=updated_at,
        task_scores={t: fs for t in IBR_TASK_TYPES},
        domain_scores={d: fs for d in IBR_DOMAINS},
        qs_scores={q: fs for q in IBR_QUALITY_SPEED},
    )


def _make_discovery_probe(**overrides) -> DiscoveryProbe:
    """Build a DiscoveryProbe with sensible defaults."""
    defaults = {
        "id": "disc-test-001",
        "task_type": "generation",
        "domain": "code",
        "quality_speed": "quality",
        "prompt": "Test prompt text here.",
        "judge_criteria": "Test judge criteria here.",
        "discrimination_axis": "style",
        "difficulty": "medium",
    }
    defaults.update(overrides)
    return DiscoveryProbe(**defaults)


# ===========================================================================
# probes.py tests
# ===========================================================================


class TestGetAllProbes:
    """Tests for the probe bank."""

    def test_returns_at_least_80_probes(self):
        probes = get_all_probes()
        assert len(probes) >= 80

    def test_all_ids_unique(self):
        probes = get_all_probes()
        ids = [p.id for p in probes]
        assert len(ids) == len(set(ids)), "Duplicate probe IDs found"

    def test_all_ids_start_with_disc(self):
        probes = get_all_probes()
        for p in probes:
            assert p.id.startswith("disc-"), f"Probe {p.id} does not start with 'disc-'"

    def test_all_task_types_covered_at_least_3_each(self):
        probes = get_all_probes()
        counts = Counter(p.task_type for p in probes)
        for task_type in IBR_TASK_TYPES:
            assert counts.get(task_type, 0) >= 3, (
                f"task_type '{task_type}' has only {counts.get(task_type, 0)} probes"
            )

    def test_all_domains_covered_at_least_2_each(self):
        probes = get_all_probes()
        counts = Counter(p.domain for p in probes)
        for domain in IBR_DOMAINS:
            assert counts.get(domain, 0) >= 2, (
                f"domain '{domain}' has only {counts.get(domain, 0)} probes"
            )

    def test_all_discrimination_axes_covered_at_least_2_each(self):
        probes = get_all_probes()
        counts = Counter(p.discrimination_axis for p in probes)
        for axis in DISCRIMINATION_AXES:
            assert counts.get(axis, 0) >= 2, (
                f"axis '{axis}' has only {counts.get(axis, 0)} probes"
            )


class TestValidateProbe:
    """Tests for _validate_probe rejection on invalid fields."""

    def test_rejects_invalid_task_type(self):
        probe = _make_discovery_probe(task_type="INVALID")
        with pytest.raises(AssertionError, match="Invalid task_type"):
            _validate_probe(probe)

    def test_rejects_invalid_domain(self):
        probe = _make_discovery_probe(domain="INVALID")
        with pytest.raises(AssertionError, match="Invalid domain"):
            _validate_probe(probe)

    def test_rejects_invalid_quality_speed(self):
        probe = _make_discovery_probe(quality_speed="INVALID")
        with pytest.raises(AssertionError, match="Invalid quality_speed"):
            _validate_probe(probe)

    def test_rejects_invalid_discrimination_axis(self):
        probe = _make_discovery_probe(discrimination_axis="INVALID")
        with pytest.raises(AssertionError, match="Invalid discrimination_axis"):
            _validate_probe(probe)

    def test_rejects_invalid_difficulty(self):
        probe = _make_discovery_probe(difficulty="INVALID")
        with pytest.raises(AssertionError, match="Invalid difficulty"):
            _validate_probe(probe)

    def test_rejects_id_not_starting_with_disc(self):
        probe = _make_discovery_probe(id="bad-id-001")
        with pytest.raises(AssertionError, match="Probe ID must start with 'disc-'"):
            _validate_probe(probe)


class TestProbeFilters:
    """Tests for get_probes_by_task_type, get_probes_by_domain, get_probes_by_axis."""

    def test_get_probes_by_task_type_filters_correctly(self):
        probes = get_probes_by_task_type("generation")
        assert len(probes) > 0
        for p in probes:
            assert p.task_type == "generation"

    def test_get_probes_by_domain_filters_correctly(self):
        probes = get_probes_by_domain("code")
        assert len(probes) > 0
        for p in probes:
            assert p.domain == "code"

    def test_get_probes_by_axis_filters_correctly(self):
        probes = get_probes_by_axis("style")
        assert len(probes) > 0
        for p in probes:
            assert p.discrimination_axis == "style"

    def test_get_probes_by_task_type_rejects_invalid(self):
        with pytest.raises(AssertionError, match="Invalid task_type"):
            get_probes_by_task_type("INVALID")

    def test_get_probes_by_domain_rejects_invalid(self):
        with pytest.raises(AssertionError, match="Invalid domain"):
            get_probes_by_domain("INVALID")

    def test_get_probes_by_axis_rejects_invalid(self):
        with pytest.raises(AssertionError, match="Invalid discrimination_axis"):
            get_probes_by_axis("INVALID")


# ===========================================================================
# analyzer.py tests
# ===========================================================================


class TestProbeResultDataclass:
    """Tests for ProbeResult construction."""

    def test_basic_construction(self):
        r = _make_probe_result()
        assert r.model_id == "test/model-a"
        assert r.probe_id == "disc-test-001"
        assert r.normalized_score == 0.8
        assert r.error is None

    def test_error_field(self):
        r = _make_probe_result(error="test_error")
        assert r.error == "test_error"

    def test_frozen(self):
        r = _make_probe_result()
        with pytest.raises(AttributeError):
            r.model_id = "other"  # type: ignore[misc]


class TestAggregateScores:
    """Tests for aggregate_scores."""

    def test_empty_list_returns_empty_dict(self):
        result = aggregate_scores([])
        assert result == {}

    def test_all_errored_returns_empty_dict(self):
        results = [
            _make_probe_result(error="fail_1"),
            _make_probe_result(error="fail_2"),
        ]
        result = aggregate_scores(results)
        assert result == {}

    def test_filters_out_errored_probes(self):
        results = [
            _make_probe_result(model_id="m1", normalized_score=0.9, error=None),
            _make_probe_result(model_id="m1", normalized_score=0.1, error="broken"),
        ]
        agg = aggregate_scores(results)
        assert "m1" in agg
        # Only one valid result (0.9), the errored one is excluded
        assert agg["m1"].task_scores["generation"].mean == pytest.approx(0.9)
        assert agg["m1"].task_scores["generation"].count == 1

    def test_computes_correct_mean_and_stddev(self):
        results = [
            _make_probe_result(model_id="m1", probe_id="p1", normalized_score=0.6),
            _make_probe_result(model_id="m1", probe_id="p2", normalized_score=0.8),
        ]
        agg = aggregate_scores(results)
        stats = agg["m1"].task_scores["generation"]
        assert stats.mean == pytest.approx(0.7)
        assert stats.count == 2
        # stddev of [0.6, 0.8] = sqrt(((0.6-0.7)^2 + (0.8-0.7)^2) / 2) = 0.1
        assert stats.stddev == pytest.approx(0.1)

    def test_multiple_models(self):
        results = [
            _make_probe_result(model_id="m1", probe_id="p1", normalized_score=0.5),
            _make_probe_result(model_id="m2", probe_id="p1", normalized_score=0.9),
        ]
        agg = aggregate_scores(results)
        assert "m1" in agg
        assert "m2" in agg
        assert agg["m1"].task_scores["generation"].mean == pytest.approx(0.5)
        assert agg["m2"].task_scores["generation"].mean == pytest.approx(0.9)

    def test_groups_by_task_type_and_domain(self):
        results = [
            _make_probe_result(
                model_id="m1", probe_id="p1",
                task_type="generation", domain="code", normalized_score=0.8,
            ),
            _make_probe_result(
                model_id="m1", probe_id="p2",
                task_type="analysis", domain="technical", normalized_score=0.6,
            ),
        ]
        agg = aggregate_scores(results)
        fp = agg["m1"]
        assert "generation" in fp.task_scores
        assert "analysis" in fp.task_scores
        assert "code" in fp.domain_scores
        assert "technical" in fp.domain_scores


class TestRankNormalize:
    """Tests for rank_normalize."""

    def test_empty_input_returns_empty(self):
        result = rank_normalize({})
        assert result == {}

    def test_single_model_gets_0_5_on_all_dimensions(self):
        results = [
            _make_probe_result(model_id="m1", probe_id="p1", normalized_score=0.9),
        ]
        raw = aggregate_scores(results)
        profiles = rank_normalize(raw)
        assert "m1" in profiles
        profile = profiles["m1"]
        # Single model should get 0.5 on all dimensions
        assert profile.task_scores["generation"].score == pytest.approx(0.5)
        assert profile.domain_scores["code"].score == pytest.approx(0.5)
        assert profile.qs_scores["quality"].score == pytest.approx(0.5)

    def test_two_models_get_0_and_1(self):
        results = [
            _make_probe_result(model_id="m1", probe_id="p1", normalized_score=0.3),
            _make_probe_result(model_id="m2", probe_id="p1", normalized_score=0.9),
        ]
        raw = aggregate_scores(results)
        profiles = rank_normalize(raw)
        # m1 has lower score -> 0.0, m2 has higher -> 1.0
        assert profiles["m1"].task_scores["generation"].score == pytest.approx(0.0)
        assert profiles["m2"].task_scores["generation"].score == pytest.approx(1.0)

    def test_missing_dimensions_get_neutral_flavor(self):
        results = [
            _make_probe_result(
                model_id="m1", probe_id="p1",
                task_type="generation", domain="code", quality_speed="quality",
                normalized_score=0.8,
            ),
        ]
        raw = aggregate_scores(results)
        profiles = rank_normalize(raw)
        profile = profiles["m1"]
        # "analysis" was never tested so should be neutral
        assert profile.task_scores["analysis"] == IBR_NEUTRAL_FLAVOR
        # "technical" domain was never tested so should be neutral
        assert profile.domain_scores["technical"] == IBR_NEUTRAL_FLAVOR

    def test_all_ibr_dimensions_present_in_output(self):
        results = [
            _make_probe_result(model_id="m1", probe_id="p1", normalized_score=0.5),
        ]
        raw = aggregate_scores(results)
        profiles = rank_normalize(raw)
        profile = profiles["m1"]
        for t in IBR_TASK_TYPES:
            assert t in profile.task_scores
        for d in IBR_DOMAINS:
            assert d in profile.domain_scores
        for q in IBR_QUALITY_SPEED:
            assert q in profile.qs_scores


class TestComputeCalibrationDeltas:
    """Tests for compute_calibration_deltas."""

    def test_nonexistent_path_returns_empty(self, tmp_path):
        profiles = {"m1": _make_flavor_profile(model_id="m1")}
        result = compute_calibration_deltas(profiles, tmp_path / "no-such-file.yaml")
        assert result == {}

    def test_correctly_classifies_confirm_review_update(self, tmp_path):
        # Write a declared profile with score=0.7 for generation
        declared_yaml = {
            "profiles": {
                "m1": {
                    "task_scores": {"generation": 0.7},
                    "domain_scores": {},
                    "qs_scores": {},
                },
            },
        }
        declared_path = tmp_path / "declared.yaml"
        declared_path.write_text(yaml.dump(declared_yaml))

        # Create empirical profile with 0.72 for generation (delta=0.02 -> confirm)
        empirical_profile = _make_flavor_profile(model_id="m1", score=0.72)
        result = compute_calibration_deltas({"m1": empirical_profile}, declared_path)
        assert "m1" in result
        # generation dimension should be "confirm" (delta ~0.02)
        gen_delta = result["m1"]["task/generation"]
        assert gen_delta.recommendation == "confirm"


class TestDeltaRecommendation:
    """Tests for _delta_recommendation thresholds."""

    def test_confirm_threshold(self):
        assert _delta_recommendation(0.0) == "confirm"
        assert _delta_recommendation(0.03) == "confirm"
        assert _delta_recommendation(0.05) == "confirm"

    def test_review_threshold(self):
        assert _delta_recommendation(0.06) == "review"
        assert _delta_recommendation(0.10) == "review"
        assert _delta_recommendation(0.15) == "review"

    def test_update_threshold(self):
        assert _delta_recommendation(0.16) == "update"
        assert _delta_recommendation(0.20) == "update"
        assert _delta_recommendation(0.50) == "update"


class TestBuildFingerprintsYaml:
    """Tests for build_fingerprints_yaml."""

    def test_produces_valid_yaml(self):
        profiles = {"m1": _make_flavor_profile(model_id="m1")}
        yaml_str = build_fingerprints_yaml(profiles, "test-run-001")
        parsed = yaml.safe_load(yaml_str)
        assert isinstance(parsed, dict)

    def test_includes_source_and_generated_at(self):
        profiles = {"m1": _make_flavor_profile(model_id="m1")}
        yaml_str = build_fingerprints_yaml(profiles, "test-run-001")
        parsed = yaml.safe_load(yaml_str)
        assert parsed["source"] == "discovery-run-test-run-001"
        assert "generated_at" in parsed
        assert parsed["version"] == 1

    def test_includes_all_models(self):
        profiles = {
            "m1": _make_flavor_profile(model_id="m1"),
            "m2": _make_flavor_profile(model_id="m2"),
        }
        yaml_str = build_fingerprints_yaml(profiles, "run-002")
        parsed = yaml.safe_load(yaml_str)
        assert "m1" in parsed["profiles"]
        assert "m2" in parsed["profiles"]

    def test_scores_are_rounded(self):
        profiles = {"m1": _make_flavor_profile(model_id="m1", score=0.123456789)}
        yaml_str = build_fingerprints_yaml(profiles, "run-003")
        parsed = yaml.safe_load(yaml_str)
        for _key, score_val in parsed["profiles"]["m1"]["task_scores"].items():
            # Should be rounded to 4 decimal places
            assert score_val == pytest.approx(0.1235, abs=1e-4)


class TestBuildModelRankings:
    """Tests for build_model_rankings."""

    def test_empty_profiles_returns_empty(self):
        result = build_model_rankings({})
        assert result == {}

    def test_returns_correct_ordering(self):
        # m2 has higher scores than m1
        p1 = _make_flavor_profile(model_id="m1", score=0.3)
        p2 = _make_flavor_profile(model_id="m2", score=0.9)
        rankings = build_model_rankings({"m1": p1, "m2": p2})
        # For each dimension, m2 should rank first
        assert rankings["task/generation"][0] == "m2"
        assert rankings["task/generation"][1] == "m1"

    def test_covers_all_dimension_types(self):
        p1 = _make_flavor_profile(model_id="m1")
        rankings = build_model_rankings({"m1": p1})
        # Check that all task, domain, and qs dimensions are present
        for t in IBR_TASK_TYPES:
            assert f"task/{t}" in rankings
        for d in IBR_DOMAINS:
            assert f"domain/{d}" in rankings
        for q in IBR_QUALITY_SPEED:
            assert f"qs/{q}" in rankings


# ===========================================================================
# lifecycle.py tests
# ===========================================================================


class TestCheckStaleness:
    """Tests for check_staleness."""

    def test_detects_stale_profiles(self):
        old_time = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        profiles = {"m1": _make_flavor_profile(model_id="m1", updated_at=old_time)}
        results = check_staleness(profiles)
        assert len(results) == 1
        assert results[0].needs_refresh is True
        assert results[0].age_days > 30

    def test_marks_fresh_profiles_as_not_needing_refresh(self):
        fresh_time = datetime.now(UTC).isoformat()
        profiles = {"m1": _make_flavor_profile(model_id="m1", updated_at=fresh_time)}
        results = check_staleness(profiles)
        assert len(results) == 1
        assert results[0].needs_refresh is False

    def test_custom_threshold(self):
        # 10 days old, with threshold=5 -> stale
        old_time = (datetime.now(UTC) - timedelta(days=10)).isoformat()
        profiles = {"m1": _make_flavor_profile(model_id="m1", updated_at=old_time)}
        results = check_staleness(profiles, threshold_days=5)
        assert results[0].needs_refresh is True


class TestApplyDiscoveryDecay:
    """Tests for apply_discovery_decay."""

    def test_no_op_for_profiles_30_days_or_younger(self):
        recent_time = datetime.now(UTC).isoformat()
        profile = _make_flavor_profile(model_id="m1", updated_at=recent_time, score=0.9)
        result = apply_discovery_decay(profile)
        # Scores should be unchanged
        for t in IBR_TASK_TYPES:
            assert result.task_scores[t].score == pytest.approx(0.9)

    def test_decays_toward_0_5_for_old_profiles(self):
        old_time = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        profile = _make_flavor_profile(model_id="m1", updated_at=old_time, score=0.9)
        result = apply_discovery_decay(profile)
        # Score should move toward 0.5 (be less than 0.9)
        for t in IBR_TASK_TYPES:
            assert result.task_scores[t].score < 0.9
            assert result.task_scores[t].score > 0.5  # Shouldn't overshoot

    def test_preserves_original_updated_at(self):
        old_time = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        profile = _make_flavor_profile(model_id="m1", updated_at=old_time, score=0.9)
        result = apply_discovery_decay(profile)
        assert result.updated_at == old_time

    def test_decay_moves_low_scores_up(self):
        old_time = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        profile = _make_flavor_profile(model_id="m1", updated_at=old_time, score=0.1)
        result = apply_discovery_decay(profile)
        # Score 0.1 should move toward 0.5 (increase)
        for t in IBR_TASK_TYPES:
            assert result.task_scores[t].score > 0.1
            assert result.task_scores[t].score < 0.5

    def test_explicit_now_parameter(self):
        updated = datetime(2026, 1, 1, tzinfo=UTC)
        profile = _make_flavor_profile(
            model_id="m1",
            updated_at=updated.isoformat(),
            score=0.9,
        )
        # now = 61 days later -> should decay
        now = updated + timedelta(days=61)
        result = apply_discovery_decay(profile, now=now)
        for t in IBR_TASK_TYPES:
            assert result.task_scores[t].score < 0.9


class TestMergeIncremental:
    """Tests for merge_incremental."""

    def test_adds_new_models(self):
        existing = {"m1": _make_flavor_profile(model_id="m1")}
        new_results = {"m2": _make_flavor_profile(model_id="m2")}
        merged = merge_incremental(existing, new_results)
        assert "m1" in merged
        assert "m2" in merged

    def test_replaces_existing_models_with_new_results(self):
        old = _make_flavor_profile(model_id="m1", score=0.3)
        new = _make_flavor_profile(model_id="m1", score=0.9)
        merged = merge_incremental({"m1": old}, {"m1": new})
        assert merged["m1"].task_scores["generation"].score == pytest.approx(0.9)

    def test_preserves_models_not_in_new_results(self):
        existing = {
            "m1": _make_flavor_profile(model_id="m1", score=0.5),
            "m2": _make_flavor_profile(model_id="m2", score=0.7),
        }
        new_results = {"m1": _make_flavor_profile(model_id="m1", score=0.9)}
        merged = merge_incremental(existing, new_results)
        # m2 should be unchanged
        assert merged["m2"].task_scores["generation"].score == pytest.approx(0.7)
        # m1 should be updated
        assert merged["m1"].task_scores["generation"].score == pytest.approx(0.9)

    def test_empty_existing_and_new(self):
        merged = merge_incremental({}, {})
        assert merged == {}


class TestWriteFingerprintsYaml:
    """Tests for write_fingerprints_yaml (lifecycle version)."""

    def test_creates_file(self, tmp_path):
        output_path = tmp_path / "sub" / "profiles.yaml"
        write_fingerprints_yaml("version: 1\nprofiles: {}", output_path)
        assert output_path.exists()
        assert output_path.read_text() == "version: 1\nprofiles: {}"

    def test_creates_parent_directories(self, tmp_path):
        output_path = tmp_path / "deep" / "nested" / "dir" / "profiles.yaml"
        write_fingerprints_yaml("test content", output_path)
        assert output_path.exists()


class TestLoadExistingFingerprints:
    """Tests for load_existing_fingerprints."""

    def test_returns_empty_for_missing_file(self, tmp_path):
        result = load_existing_fingerprints(tmp_path / "nonexistent.yaml")
        assert result == {}

    def test_parses_valid_yaml(self, tmp_path):
        yaml_data = {
            "version": 1,
            "profiles": {
                "m1": {
                    "task_scores": {"generation": 0.8},
                    "domain_scores": {"code": 0.9},
                    "qs_scores": {"quality": 0.7},
                },
            },
        }
        path = tmp_path / "profiles.yaml"
        path.write_text(yaml.dump(yaml_data))
        result = load_existing_fingerprints(path)
        assert "m1" in result
        assert result["m1"].task_scores["generation"].score == pytest.approx(0.8)

    def test_returns_empty_for_invalid_yaml(self, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text(":{[invalid yaml content")
        result = load_existing_fingerprints(path)
        assert result == {}


class TestGetModelsNeedingDiscovery:
    """Tests for get_models_needing_discovery."""

    def test_identifies_missing_profiles(self, tmp_path):
        matrix = {
            "roles": {
                "coding": [
                    {"model_id": "m1", "rank": 90},
                    {"model_id": "m2", "rank": 80},
                ],
            },
        }
        matrix_path = tmp_path / "matrix.json"
        matrix_path.write_text(json.dumps(matrix))
        # Only m1 has a profile
        existing = {"m1": _make_flavor_profile(model_id="m1")}
        result = get_models_needing_discovery(matrix_path, existing)
        assert "m2" in result
        assert "m1" not in result

    def test_identifies_stale_profiles(self, tmp_path):
        matrix = {
            "roles": {
                "coding": [{"model_id": "m1", "rank": 90}],
            },
        }
        matrix_path = tmp_path / "matrix.json"
        matrix_path.write_text(json.dumps(matrix))
        old_time = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        existing = {"m1": _make_flavor_profile(model_id="m1", updated_at=old_time)}
        result = get_models_needing_discovery(matrix_path, existing)
        assert "m1" in result

    def test_missing_matrix_returns_empty(self, tmp_path):
        result = get_models_needing_discovery(
            tmp_path / "no-such-matrix.json", {},
        )
        assert result == []


# ===========================================================================
# runner.py tests
# ===========================================================================


class TestProviderPacer:
    """Tests for ProviderPacer."""

    def test_uses_default_delays(self):
        pacer = ProviderPacer()
        assert pacer._delays["gemini"] == 1.0
        assert pacer._delays["groq"] == 1.5
        assert pacer._delays["openrouter"] == 2.0

    def test_accepts_overrides(self):
        pacer = ProviderPacer(overrides={"gemini": 5.0, "custom": 3.0})
        assert pacer._delays["gemini"] == 5.0
        assert pacer._delays["custom"] == 3.0
        # Others should still be at defaults
        assert pacer._delays["groq"] == 1.5


class TestExtractProvider:
    """Tests for _extract_provider."""

    def test_extracts_correctly(self):
        assert _extract_provider("gemini/gemini-2.5-pro") == "gemini"
        assert _extract_provider("openrouter/meta-llama/llama-3") == "openrouter"

    def test_no_slash_returns_unknown(self):
        assert _extract_provider("no-provider") == "unknown"


class TestInterleavedSchedule:
    """Tests for _interleaved_schedule."""

    def test_interleaves_models_across_probes(self):
        models = ["m1", "m2"]
        probes = [
            _make_discovery_probe(id="disc-a-001"),
            _make_discovery_probe(id="disc-a-002"),
        ]
        schedule = _interleaved_schedule(models, probes)
        # Should be: (m1, probe1), (m2, probe1), (m1, probe2), (m2, probe2)
        assert len(schedule) == 4
        assert schedule[0] == ("m1", probes[0])
        assert schedule[1] == ("m2", probes[0])
        assert schedule[2] == ("m1", probes[1])
        assert schedule[3] == ("m2", probes[1])

    def test_empty_models(self):
        probes = [_make_discovery_probe()]
        schedule = _interleaved_schedule([], probes)
        assert schedule == []

    def test_empty_probes(self):
        schedule = _interleaved_schedule(["m1"], [])
        assert schedule == []


class TestCheckpoint:
    """Tests for _load_checkpoint and _append_checkpoint."""

    def test_load_returns_empty_set_for_missing_file(self, tmp_path):
        result = _load_checkpoint(tmp_path / "no-such-checkpoint.jsonl")
        assert result == set()

    def test_round_trip(self, tmp_path):
        cp_path = tmp_path / "checkpoint.jsonl"
        r1 = _make_probe_result(model_id="m1", probe_id="disc-p1")
        r2 = _make_probe_result(model_id="m2", probe_id="disc-p2")
        _append_checkpoint(cp_path, r1)
        _append_checkpoint(cp_path, r2)
        loaded = _load_checkpoint(cp_path)
        assert ("m1", "disc-p1") in loaded
        assert ("m2", "disc-p2") in loaded
        assert len(loaded) == 2

    def test_load_skips_malformed_lines(self, tmp_path):
        cp_path = tmp_path / "checkpoint.jsonl"
        cp_path.write_text('{"model_id": "m1", "probe_id": "p1"}\nNOT JSON\n')
        loaded = _load_checkpoint(cp_path)
        assert ("m1", "p1") in loaded
        assert len(loaded) == 1


class TestCreateAdapter:
    """Tests for _create_adapter — real adapter factory."""

    def test_unknown_provider_returns_none(self, monkeypatch):
        """Unknown provider prefix → None."""
        monkeypatch.setattr(
            "dragonlight_router.discovery.runner._CACHED_PROVIDERS", {},
        )
        result = _create_adapter("totally_fake/some-model")
        assert result is None

    def test_missing_env_key_returns_none(self, monkeypatch):
        """Provider exists in config but required env var is unset → None."""
        monkeypatch.setattr(
            "dragonlight_router.discovery.runner._CACHED_PROVIDERS",
            {"gemini": {"name": "gemini", "env_key": "GEMINI_API_KEY_DEFINITELY_UNSET", "base_url": "https://example.com"}},
        )
        monkeypatch.delenv("GEMINI_API_KEY_DEFINITELY_UNSET", raising=False)
        result = _create_adapter("gemini/gemini-2.5-pro")
        assert result is None

    def test_valid_provider_returns_adapter(self, monkeypatch):
        """Known provider with env key set → non-None adapter."""
        monkeypatch.setattr(
            "dragonlight_router.discovery.runner._CACHED_PROVIDERS",
            {"groq": {"name": "groq", "env_key": "GROQ_API_KEY", "base_url": "https://api.groq.com/openai/v1", "model_prefix": "groq/", "rate_limits": {}}},
        )
        monkeypatch.setenv("GROQ_API_KEY", "test-key-for-unit-test")
        result = _create_adapter("groq/llama-3.3-70b-versatile")
        assert result is not None


class TestParseDelays:
    """Tests for _parse_delays."""

    def test_parses_key_value_pairs(self):
        result = _parse_delays(["gemini=2.0", "groq=3.5"])
        assert result == {"gemini": 2.0, "groq": 3.5}

    def test_none_input_returns_none(self):
        assert _parse_delays(None) is None

    def test_empty_list_returns_none(self):
        assert _parse_delays([]) is None

    def test_invalid_format_raises(self):
        with pytest.raises(SystemExit, match="Invalid --provider-delay"):
            _parse_delays(["bad-no-equals"])
