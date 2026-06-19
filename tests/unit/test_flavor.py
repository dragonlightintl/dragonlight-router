"""Tests for selection/flavor.py — model flavor profile system.

Spec traceability: IBR spec v0.1.0 sections 3, 4, 10.
AC numbers: IBR-FLV-01 through IBR-FLV-06, IBR-SCORE-01 through IBR-SCORE-04.
"""

from __future__ import annotations

import tempfile
import time
from pathlib import Path

import pytest
import yaml

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_FLAVOR,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    ClassifiedIntent,
    FlavorScore,
    ModelFlavorProfile,
)
from dragonlight_router.selection.flavor import (
    FlavorProfileLoader,
    _average_matched_confidence,
    _build_neutral_profile,
    _clamp_score,
    _merge_dimension_scores,
    _merge_single_profile,
    _parse_dimension_scores,
    compute_flavor_match,
    compute_flavor_scores,
    get_profile_for_model,
    should_apply_flavor_match,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(**overrides) -> ClassifiedIntent:
    """Build a ClassifiedIntent with sensible defaults."""
    defaults = {
        "task_type": "analysis",
        "domain": "code",
        "quality_speed": "balanced",
        "confidence": 0.9,
        "latency_ms": 15.0,
        "from_cache": False,
    }
    defaults.update(overrides)
    return ClassifiedIntent(**defaults)


def _make_profile(
    model_id: str = "test-model",
    task_scores: dict[str, float] | None = None,
    domain_scores: dict[str, float] | None = None,
    qs_scores: dict[str, float] | None = None,
) -> ModelFlavorProfile:
    """Build a ModelFlavorProfile with optional partial scores."""

    def _build_scores(
        raw: dict[str, float] | None,
        allowed: frozenset[str],
    ) -> dict[str, FlavorScore]:
        scores: dict[str, FlavorScore] = {}
        parsed = raw or {}
        for key in allowed:
            if key in parsed:
                scores[key] = FlavorScore(score=parsed[key], confidence=1.0, sample_count=0)
            else:
                scores[key] = IBR_NEUTRAL_FLAVOR
        return scores

    return ModelFlavorProfile(
        model_id=model_id,
        version=1,
        updated_at="2026-01-01T00:00:00+00:00",
        task_scores=_build_scores(task_scores, IBR_TASK_TYPES),
        domain_scores=_build_scores(domain_scores, IBR_DOMAINS),
        qs_scores=_build_scores(qs_scores, IBR_QUALITY_SPEED),
    )


def _write_yaml(path: Path, data: dict) -> None:
    """Write YAML to a file."""
    path.write_text(yaml.dump(data, default_flow_style=False))


# ---------------------------------------------------------------------------
# Profile loading
# ---------------------------------------------------------------------------


class TestProfileLoading:
    """[IBR-FLV-01] [IBR-FLV-02] [HAZ-019] YAML loading, defaults, error handling."""

    def test_valid_yaml_loads_profiles(self):
        """[IBR-FLV-01] Valid YAML with profiles loads correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            data = {
                "profiles": {
                    "model-a": {
                        "version": 1,
                        "task_scores": {"analysis": 0.9, "generation": 0.7},
                        "domain_scores": {"code": 0.85},
                        "qs_scores": {"quality": 0.8},
                    },
                },
            }
            _write_yaml(path, data)
            loader = FlavorProfileLoader(path)
            assert "model-a" in loader.profiles
            profile = loader.profiles["model-a"]
            assert profile.model_id == "model-a"
            assert profile.task_scores["analysis"].score == 0.9
            assert profile.task_scores["analysis"].confidence == 1.0

    def test_missing_file_returns_empty_dict(self):
        """[IBR-FLV-01] [IBR-CFG-03] Missing file results in empty profiles."""
        path = Path("/nonexistent/profiles.yaml")
        loader = FlavorProfileLoader(path)
        assert loader.profiles == {}

    def test_invalid_yaml_returns_empty_dict(self):
        """[HAZ-019] Malformed YAML results in empty profiles, not crash."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            path.write_text("{{{{not: valid: yaml: : :")
            loader = FlavorProfileLoader(path)
            assert loader.profiles == {}

    def test_unlisted_dimensions_get_neutral_defaults(self):
        """[IBR-FLV-02] Dimensions not declared in profile get neutral defaults."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            data = {
                "profiles": {
                    "model-b": {
                        "task_scores": {"analysis": 0.9},
                        "domain_scores": {},
                        "qs_scores": {},
                    },
                },
            }
            _write_yaml(path, data)
            loader = FlavorProfileLoader(path)
            profile = loader.profiles["model-b"]

            # Declared dimension
            assert profile.task_scores["analysis"].score == 0.9
            assert profile.task_scores["analysis"].confidence == 1.0

            # Undeclared dimension gets neutral
            assert profile.task_scores["generation"].score == 0.5
            assert profile.task_scores["generation"].confidence == 0.0
            assert profile.task_scores["generation"].sample_count == 0

    def test_operator_declared_scores_get_confidence_one(self):
        """[IBR-FLV-02] Operator-declared scores have confidence=1.0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            data = {
                "profiles": {
                    "model-c": {
                        "task_scores": {"creative": 0.6},
                        "domain_scores": {"code": 0.95},
                        "qs_scores": {"speed": 0.3},
                    },
                },
            }
            _write_yaml(path, data)
            loader = FlavorProfileLoader(path)
            profile = loader.profiles["model-c"]
            assert profile.task_scores["creative"].confidence == 1.0
            assert profile.domain_scores["code"].confidence == 1.0
            assert profile.qs_scores["speed"].confidence == 1.0

    def test_score_clamping_above_one(self):
        """[IBR-FLV-05] Scores > 1.0 are clamped to 1.0."""
        assert _clamp_score(1.5) == 1.0
        assert _clamp_score(100.0) == 1.0

    def test_score_clamping_below_zero(self):
        """[IBR-FLV-05] Scores < 0.0 are clamped to 0.0."""
        assert _clamp_score(-0.5) == 0.0
        assert _clamp_score(-100.0) == 0.0

    def test_score_clamping_within_range(self):
        """[IBR-FLV-05] Scores in [0.0, 1.0] pass through unchanged."""
        assert _clamp_score(0.0) == 0.0
        assert _clamp_score(0.5) == 0.5
        assert _clamp_score(1.0) == 1.0

    def test_clamping_applied_during_yaml_load(self):
        """[IBR-FLV-05] Out-of-range scores in YAML are clamped during loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            data = {
                "profiles": {
                    "model-d": {
                        "task_scores": {"analysis": 1.5, "generation": -0.3},
                    },
                },
            }
            _write_yaml(path, data)
            loader = FlavorProfileLoader(path)
            profile = loader.profiles["model-d"]
            assert profile.task_scores["analysis"].score == 1.0
            assert profile.task_scores["generation"].score == 0.0

    def test_multiple_profiles_loaded(self):
        """[IBR-FLV-01] Multiple model profiles load from one YAML."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            data = {
                "profiles": {
                    "model-x": {"task_scores": {"analysis": 0.8}},
                    "model-y": {"task_scores": {"creative": 0.9}},
                },
            }
            _write_yaml(path, data)
            loader = FlavorProfileLoader(path)
            assert len(loader.profiles) == 2
            assert "model-x" in loader.profiles
            assert "model-y" in loader.profiles

    def test_empty_profiles_section(self):
        """[IBR-FLV-01] Empty profiles section in YAML loads zero profiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            _write_yaml(path, {"profiles": {}})
            loader = FlavorProfileLoader(path)
            assert loader.profiles == {}

    def test_parse_dimension_scores_fills_all_keys(self):
        """[IBR-FLV-02] _parse_dimension_scores fills all allowed keys."""
        scores = _parse_dimension_scores({"analysis": 0.8}, IBR_TASK_TYPES)
        assert len(scores) == len(IBR_TASK_TYPES)
        assert scores["analysis"].score == 0.8
        assert scores["analysis"].confidence == 1.0
        for key in IBR_TASK_TYPES:
            assert key in scores

    def test_parse_dimension_scores_non_dict_input(self):
        """[IBR-FLV-02] Non-dict input to _parse_dimension_scores uses all defaults."""
        scores = _parse_dimension_scores(None, IBR_TASK_TYPES)
        assert len(scores) == len(IBR_TASK_TYPES)
        for key in IBR_TASK_TYPES:
            assert scores[key] == IBR_NEUTRAL_FLAVOR


# ---------------------------------------------------------------------------
# Flavor match scoring
# ---------------------------------------------------------------------------


class TestFlavorMatchScoring:
    """[IBR-SCORE-01] Weighted flavor match computation."""

    def test_full_profile_correct_weighted_sum(self):
        """[IBR-SCORE-01] compute_flavor_match returns correct 0.50/0.30/0.20 weighted sum."""
        profile = _make_profile(
            task_scores={"analysis": 0.8},
            domain_scores={"code": 0.6},
            qs_scores={"balanced": 0.9},
        )
        intent = _make_intent()
        result = compute_flavor_match(intent, profile)
        expected = 0.50 * 0.8 + 0.30 * 0.6 + 0.20 * 0.9
        assert result == pytest.approx(expected, abs=1e-9)

    def test_weights_are_0_50_0_30_0_20(self):
        """[IBR-SCORE-01] Dimension weights are exactly 0.50, 0.30, 0.20."""
        # Use 1.0, 0.0, 0.0 to isolate task weight
        profile_task = _make_profile(task_scores={"analysis": 1.0})
        intent = _make_intent()
        task_contrib = compute_flavor_match(intent, profile_task)
        # Domain and qs are 0.5 (neutral), so:
        # 0.50 * 1.0 + 0.30 * 0.5 + 0.20 * 0.5 = 0.50 + 0.15 + 0.10 = 0.75
        assert task_contrib == pytest.approx(0.75, abs=1e-9)

    def test_missing_dimensions_use_neutral_default(self):
        """[IBR-FLV-02] Missing dimensions in profile use 0.5 default score."""
        profile = _make_profile()  # All neutral
        intent = _make_intent()
        result = compute_flavor_match(intent, profile)
        # All 0.5: 0.50 * 0.5 + 0.30 * 0.5 + 0.20 * 0.5 = 0.5
        assert result == pytest.approx(0.5, abs=1e-9)

    def test_result_always_in_unit_interval(self):
        """[IBR-SCORE-01] Result is always in [0.0, 1.0]."""
        # Minimum: all zeros
        profile_min = _make_profile(
            task_scores={"analysis": 0.0},
            domain_scores={"code": 0.0},
            qs_scores={"balanced": 0.0},
        )
        intent = _make_intent()
        assert compute_flavor_match(intent, profile_min) >= 0.0

        # Maximum: all ones
        profile_max = _make_profile(
            task_scores={"analysis": 1.0},
            domain_scores={"code": 1.0},
            qs_scores={"balanced": 1.0},
        )
        assert compute_flavor_match(intent, profile_max) <= 1.0

    def test_all_zeros_profile(self):
        """[IBR-SCORE-01] Profile with all 0.0 scores yields 0.0 match."""
        profile = _make_profile(
            task_scores={"analysis": 0.0},
            domain_scores={"code": 0.0},
            qs_scores={"balanced": 0.0},
        )
        intent = _make_intent()
        result = compute_flavor_match(intent, profile)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_all_ones_profile(self):
        """[IBR-SCORE-01] Profile with all 1.0 scores yields 1.0 match."""
        profile = _make_profile(
            task_scores={"analysis": 1.0},
            domain_scores={"code": 1.0},
            qs_scores={"balanced": 1.0},
        )
        intent = _make_intent()
        result = compute_flavor_match(intent, profile)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_different_intents_different_scores(self):
        """[IBR-SCORE-01] Different intents produce different scores for same profile."""
        profile = _make_profile(
            task_scores={"analysis": 0.9, "creative": 0.3},
            domain_scores={"code": 0.8, "creative_writing": 0.4},
        )
        intent_code = _make_intent(task_type="analysis", domain="code")
        intent_creative = _make_intent(task_type="creative", domain="creative_writing")

        score_code = compute_flavor_match(intent_code, profile)
        score_creative = compute_flavor_match(intent_creative, profile)
        assert score_code != score_creative

    def test_symmetry_across_intent_types(self):
        """[IBR-SCORE-01] Each task_type dimension can be independently scored."""
        intent = _make_intent(task_type="refactoring")
        profile = _make_profile(task_scores={"refactoring": 0.95})
        result = compute_flavor_match(intent, profile)
        expected = 0.50 * 0.95 + 0.30 * 0.5 + 0.20 * 0.5
        assert result == pytest.approx(expected, abs=1e-9)


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------


class TestBatchScoring:
    """[IBR-SCORE-01] Batch flavor match scoring across candidates."""

    def test_compute_scores_multiple_candidates(self):
        """[IBR-SCORE-01] compute_flavor_scores returns scores for all candidates."""
        profiles = {
            "m1": _make_profile("m1", task_scores={"analysis": 0.9}),
            "m2": _make_profile("m2", task_scores={"analysis": 0.4}),
        }
        intent = _make_intent()
        scores = compute_flavor_scores(intent, profiles, ["m1", "m2"])
        assert len(scores) == 2
        assert "m1" in scores
        assert "m2" in scores
        assert scores["m1"] > scores["m2"]

    def test_returns_empty_dict_when_intent_is_none(self):
        """[IBR-SYS-03] Returns empty dict when intent is None (IBR inactive)."""
        profiles = {"m1": _make_profile("m1")}
        scores = compute_flavor_scores(None, profiles, ["m1"])
        assert scores == {}

    def test_missing_profiles_get_neutral_scores(self):
        """[IBR-FLV-02] Candidates without profiles get neutral (0.5) match score."""
        profiles: dict[str, ModelFlavorProfile] = {}
        intent = _make_intent()
        scores = compute_flavor_scores(intent, profiles, ["unknown-model"])
        assert len(scores) == 1
        assert scores["unknown-model"] == pytest.approx(0.5, abs=1e-9)

    def test_empty_candidate_list(self):
        """[IBR-SCORE-01] Empty candidate list returns empty scores."""
        intent = _make_intent()
        scores = compute_flavor_scores(intent, {}, [])
        assert scores == {}

    def test_get_profile_for_known_model(self):
        """[IBR-FLV-01] get_profile_for_model returns known profile."""
        profiles = {"m1": _make_profile("m1", task_scores={"analysis": 0.9})}
        result = get_profile_for_model("m1", profiles)
        assert result.model_id == "m1"
        assert result.task_scores["analysis"].score == 0.9

    def test_get_profile_for_unknown_model_returns_neutral(self):
        """[IBR-FLV-02] get_profile_for_model returns neutral for unknown model."""
        result = get_profile_for_model("unknown", {})
        assert result.model_id == "unknown"
        for key in IBR_TASK_TYPES:
            assert result.task_scores[key] == IBR_NEUTRAL_FLAVOR

    def test_build_neutral_profile_has_all_dimensions(self):
        """[IBR-FLV-02] Neutral profile has entries for all taxonomy values."""
        profile = _build_neutral_profile("test")
        assert len(profile.task_scores) == len(IBR_TASK_TYPES)
        assert len(profile.domain_scores) == len(IBR_DOMAINS)
        assert len(profile.qs_scores) == len(IBR_QUALITY_SPEED)


# ---------------------------------------------------------------------------
# Confidence gating
# ---------------------------------------------------------------------------


class TestConfidenceGating:
    """[IBR-SCORE-04] Confidence gating prevents low-quality signals."""

    def test_returns_false_when_intent_is_none(self):
        """[IBR-SCORE-04] should_apply returns False when intent is None."""
        profile = _make_profile()
        assert should_apply_flavor_match(None, profile) is False

    def test_returns_false_when_classifier_confidence_below_threshold(self):
        """[IBR-SCORE-04] Returns False when intent.confidence < threshold."""
        intent = _make_intent(confidence=0.3)
        profile = _make_profile(
            task_scores={"analysis": 0.9},
            domain_scores={"code": 0.9},
            qs_scores={"balanced": 0.9},
        )
        assert should_apply_flavor_match(intent, profile, confidence_threshold=0.6) is False

    def test_returns_false_when_profile_confidence_below_threshold(self):
        """[IBR-SCORE-04] Returns False when profile confidence < threshold."""
        intent = _make_intent(confidence=0.9)
        # Neutral profile has confidence=0.0 everywhere
        profile = _make_profile()
        assert (
            should_apply_flavor_match(
                intent,
                profile,
                profile_confidence_threshold=0.3,
            )
            is False
        )

    def test_returns_true_when_both_above_threshold(self):
        """[IBR-SCORE-04] Returns True when both confidences are above thresholds."""
        intent = _make_intent(confidence=0.9)
        profile = _make_profile(
            task_scores={"analysis": 0.8},
            domain_scores={"code": 0.7},
            qs_scores={"balanced": 0.6},
        )
        assert (
            should_apply_flavor_match(
                intent,
                profile,
                confidence_threshold=0.6,
                profile_confidence_threshold=0.3,
            )
            is True
        )

    def test_confidence_exactly_at_threshold_passes(self):
        """[IBR-SCORE-04] Confidence exactly at threshold still passes (>=)."""
        intent = _make_intent(confidence=0.6)
        profile = _make_profile(
            task_scores={"analysis": 0.8},
            domain_scores={"code": 0.7},
            qs_scores={"balanced": 0.6},
        )
        result = should_apply_flavor_match(
            intent,
            profile,
            confidence_threshold=0.6,
            profile_confidence_threshold=0.3,
        )
        assert result is True

    def test_confidence_just_below_threshold_fails(self):
        """[IBR-SCORE-04] Confidence just below threshold fails."""
        intent = _make_intent(confidence=0.59)
        profile = _make_profile(
            task_scores={"analysis": 0.8},
            domain_scores={"code": 0.7},
            qs_scores={"balanced": 0.6},
        )
        result = should_apply_flavor_match(
            intent,
            profile,
            confidence_threshold=0.6,
            profile_confidence_threshold=0.3,
        )
        assert result is False

    def test_average_matched_confidence_calculation(self):
        """[IBR-SCORE-04] _average_matched_confidence computes mean of 3 dimensions."""
        intent = _make_intent()
        profile = _make_profile(
            task_scores={"analysis": 0.8},
            domain_scores={"code": 0.7},
            qs_scores={"balanced": 0.6},
        )
        avg = _average_matched_confidence(intent, profile)
        # All declared → confidence=1.0 each → average=1.0
        assert avg == pytest.approx(1.0, abs=1e-9)

    def test_average_confidence_with_mixed_declared_neutral(self):
        """[IBR-SCORE-04] Average confidence mixes declared (1.0) and neutral (0.0)."""
        intent = _make_intent()
        # Only task_scores declared, domain and qs are neutral
        profile = _make_profile(task_scores={"analysis": 0.8})
        avg = _average_matched_confidence(intent, profile)
        # (1.0 + 0.0 + 0.0) / 3 = 0.333...
        assert avg == pytest.approx(1.0 / 3.0, abs=1e-9)

    def test_zero_confidence_threshold_always_passes_on_intent(self):
        """[IBR-SCORE-04] Zero confidence threshold allows any intent confidence."""
        intent = _make_intent(confidence=0.01)
        profile = _make_profile(
            task_scores={"analysis": 0.8},
            domain_scores={"code": 0.7},
            qs_scores={"balanced": 0.6},
        )
        result = should_apply_flavor_match(
            intent,
            profile,
            confidence_threshold=0.0,
            profile_confidence_threshold=0.0,
        )
        assert result is True


# ---------------------------------------------------------------------------
# Hot reload
# ---------------------------------------------------------------------------


class TestHotReload:
    """[IBR-FLV-04] mtime-based hot reload of flavor profiles."""

    def test_reload_if_changed_detects_mtime_change(self):
        """[IBR-FLV-04] reload_if_changed reloads when file mtime changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            data = {"profiles": {"model-a": {"task_scores": {"analysis": 0.7}}}}
            _write_yaml(path, data)

            loader = FlavorProfileLoader(path)
            assert loader.profiles["model-a"].task_scores["analysis"].score == 0.7

            # Update file with new content and ensure mtime advances
            time.sleep(0.05)
            data["profiles"]["model-a"]["task_scores"]["analysis"] = 0.95
            _write_yaml(path, data)

            loader.reload_if_changed()
            assert loader.profiles["model-a"].task_scores["analysis"].score == 0.95

    def test_no_reload_when_file_unchanged(self):
        """[IBR-FLV-04] reload_if_changed does not reload when mtime is unchanged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            data = {"profiles": {"model-a": {"task_scores": {"analysis": 0.7}}}}
            _write_yaml(path, data)

            loader = FlavorProfileLoader(path)
            assert len(loader.profiles) == 1  # initial load

            # Reload without changing file
            loader.reload_if_changed()
            assert loader.profiles is not None

    def test_reload_handles_missing_file_gracefully(self):
        """[IBR-FLV-04] reload_if_changed handles deleted file gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            _write_yaml(path, {"profiles": {"model-a": {"task_scores": {"analysis": 0.7}}}})

            loader = FlavorProfileLoader(path)
            assert len(loader.profiles) == 1

            # Delete file
            path.unlink()

            # Should not crash
            loader.reload_if_changed()

    def test_reload_after_bad_yaml_update(self):
        """[HAZ-019] After YAML becomes malformed, profiles become empty on reload."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            _write_yaml(path, {"profiles": {"model-a": {"task_scores": {"analysis": 0.7}}}})

            loader = FlavorProfileLoader(path)
            assert len(loader.profiles) == 1

            # Corrupt the file
            time.sleep(0.05)
            path.write_text("{{{{ bad yaml ::::")

            loader.reload_if_changed()
            assert loader.profiles == {}


# ---------------------------------------------------------------------------
# Profile merging (feedback overlay with floor enforcement)
# ---------------------------------------------------------------------------


class TestProfileMerging:
    """[IBR-FLV-03] Feedback overlay with floor enforcement."""

    def test_feedback_overlays_operator_score(self):
        """Feedback score replaces operator score when sample_count > 0."""
        operator = _make_profile(
            "m1",
            task_scores={"analysis": 0.7},
        )
        feedback = _make_profile("m1")
        # Manually set feedback task_scores with sample_count > 0
        feedback = ModelFlavorProfile(
            model_id="m1",
            version=1,
            updated_at="2026-06-01",
            task_scores={
                **feedback.task_scores,
                "analysis": FlavorScore(score=0.9, confidence=0.5, sample_count=25),
            },
            domain_scores=feedback.domain_scores,
            qs_scores=feedback.qs_scores,
        )
        merged = _merge_single_profile(operator, feedback)
        assert merged.task_scores["analysis"].score == pytest.approx(0.9, abs=1e-9)
        assert merged.task_scores["analysis"].sample_count == 25

    def test_floor_enforcement_prevents_lowering(self):
        """[IBR-FLV-03] Feedback cannot lower below 80% of operator value."""
        operator = _make_profile(
            "m1",
            task_scores={"analysis": 0.9},
        )
        feedback = ModelFlavorProfile(
            model_id="m1",
            version=1,
            updated_at="2026-06-01",
            task_scores=dict.fromkeys(IBR_TASK_TYPES, IBR_NEUTRAL_FLAVOR)
            | {
                "analysis": FlavorScore(score=0.5, confidence=0.3, sample_count=15),
            },
            domain_scores=dict.fromkeys(IBR_DOMAINS, IBR_NEUTRAL_FLAVOR),
            qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, IBR_NEUTRAL_FLAVOR),
        )
        merged = _merge_single_profile(operator, feedback)
        # Floor = 0.8 * 0.9 = 0.72, feedback = 0.5 -> floored to 0.72
        assert merged.task_scores["analysis"].score == pytest.approx(0.72, abs=1e-9)

    def test_no_feedback_preserves_operator(self):
        """Dimensions without feedback keep operator-declared values."""
        operator = _make_profile(
            "m1",
            task_scores={"analysis": 0.85},
        )
        feedback = _make_profile("m1")  # all neutral, sample_count=0
        merged = _merge_single_profile(operator, feedback)
        assert merged.task_scores["analysis"].score == 0.85
        assert merged.task_scores["analysis"].confidence == 1.0

    def test_merge_dimension_scores_mixed(self):
        """_merge_dimension_scores handles mix of feedback and operator."""
        operator_scores = {
            "analysis": FlavorScore(score=0.8, confidence=1.0, sample_count=0),
            "creative": FlavorScore(score=0.6, confidence=1.0, sample_count=0),
        }
        feedback_scores = {
            "analysis": FlavorScore(score=0.9, confidence=0.5, sample_count=25),
            "creative": IBR_NEUTRAL_FLAVOR,  # no feedback
        }
        merged = _merge_dimension_scores(operator_scores, feedback_scores)
        assert merged["analysis"].score == pytest.approx(0.9, abs=1e-9)
        assert merged["creative"].score == 0.6  # preserved from operator

    def test_get_merged_profiles_operator_plus_feedback(self):
        """get_merged_profiles merges feedback on top of operator profiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            data = {
                "profiles": {
                    "model-a": {
                        "task_scores": {"analysis": 0.8},
                        "domain_scores": {"code": 0.9},
                    },
                },
            }
            _write_yaml(path, data)
            loader = FlavorProfileLoader(path)

            feedback_profiles = {
                "model-a": ModelFlavorProfile(
                    model_id="model-a",
                    version=1,
                    updated_at="2026-06-01",
                    task_scores=dict.fromkeys(
                        IBR_TASK_TYPES,
                        IBR_NEUTRAL_FLAVOR,
                    )
                    | {
                        "analysis": FlavorScore(
                            score=0.95,
                            confidence=0.6,
                            sample_count=30,
                        ),
                    },
                    domain_scores=dict.fromkeys(
                        IBR_DOMAINS,
                        IBR_NEUTRAL_FLAVOR,
                    ),
                    qs_scores=dict.fromkeys(
                        IBR_QUALITY_SPEED,
                        IBR_NEUTRAL_FLAVOR,
                    ),
                ),
            }

            merged = loader.get_merged_profiles(feedback_profiles)
            assert "model-a" in merged
            assert merged["model-a"].task_scores["analysis"].score == (
                pytest.approx(0.95, abs=1e-9)
            )

    def test_get_merged_profiles_feedback_only_model(self):
        """Models with feedback but no operator profile are included."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            _write_yaml(path, {"profiles": {}})
            loader = FlavorProfileLoader(path)

            feedback_profiles = {
                "new-model": _make_profile("new-model"),
            }
            merged = loader.get_merged_profiles(feedback_profiles)
            assert "new-model" in merged

    def test_get_merged_profiles_empty_feedback(self):
        """Empty feedback dict preserves all operator profiles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            data = {
                "profiles": {
                    "model-a": {"task_scores": {"analysis": 0.8}},
                },
            }
            _write_yaml(path, data)
            loader = FlavorProfileLoader(path)

            merged = loader.get_merged_profiles({})
            assert "model-a" in merged
            assert merged["model-a"].task_scores["analysis"].score == 0.8


# ---------------------------------------------------------------------------
# Coverage for reload_if_changed OSError branch (lines 155-156)
# ---------------------------------------------------------------------------


class TestReloadIfChangedOSError:
    """[IBR-FLV-02] reload_if_changed handles OSError from os.path.getmtime."""

    def test_reload_if_changed_oserror_is_caught(self):
        """[IBR-FLV-02] OSError in getmtime is caught and logged (lines 155-156)."""
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "profiles.yaml"
            _write_yaml(path, {"profiles": {"model-a": {"task_scores": {"analysis": 0.8}}}})
            loader = FlavorProfileLoader(path)

            # Simulate OSError when checking mtime
            with patch(
                "dragonlight_router.selection.flavor.os.path.getmtime",
                side_effect=OSError("permission denied"),
            ):
                # Should not raise
                loader.reload_if_changed()

            # Profiles should remain unchanged
            assert "model-a" in loader.profiles


# ---------------------------------------------------------------------------
# Coverage for _parse_single_profile non-dict raw (lines 227-228)
# ---------------------------------------------------------------------------


class TestParseSingleProfileNonDict:
    """[IBR-FLV-03] _parse_single_profile returns None on non-dict input."""

    def test_non_dict_raw_returns_none(self):
        """_parse_single_profile returns None when raw is not a dict."""
        from dragonlight_router.selection.flavor import _parse_single_profile

        result = _parse_single_profile("test-model", "not a dict")  # type: ignore[arg-type]
        assert result is None

    def test_list_raw_returns_none(self):
        """[IBR-FLV-03] _parse_single_profile returns None when raw is a list."""
        from dragonlight_router.selection.flavor import _parse_single_profile

        result = _parse_single_profile("test-model", [1, 2, 3])  # type: ignore[arg-type]
        assert result is None

    def test_none_raw_returns_none(self):
        """[IBR-FLV-03] _parse_single_profile returns None when raw is None."""
        from dragonlight_router.selection.flavor import _parse_single_profile

        result = _parse_single_profile("test-model", None)  # type: ignore[arg-type]
        assert result is None
