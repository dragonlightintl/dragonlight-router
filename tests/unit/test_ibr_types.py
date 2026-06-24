"""Tests for IBR types in core/types.py — frozen dataclasses and constants.

Spec traceability: IBR spec v0.1.0 section 9.
AC numbers: IBR-DATA-01 through IBR-DATA-03.
"""

from __future__ import annotations

import dataclasses

import pytest

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_SPECTROGRAPH,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    BackendTier,
    ClassifiedIntent,
    EngineResponse,
    IBRScoringContext,
    ModelSpectrographProfile,
    SpectrographScore,
)
from dragonlight_router.selection.scoring import ScoringWeightsConfig

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ClassifiedIntent
# ---------------------------------------------------------------------------


class TestClassifiedIntent:
    """[IBR-DATA-01] ClassifiedIntent frozen dataclass."""

    def test_frozen(self):
        """[IBR-DATA-01] ClassifiedIntent is immutable (frozen)."""
        intent = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=0.9,
            latency_ms=15.0,
            from_cache=False,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            intent.task_type = "creative"  # type: ignore[misc]

    def test_all_fields_present(self):
        """[IBR-DATA-01] ClassifiedIntent has all 6 required fields."""
        intent = ClassifiedIntent(
            task_type="generation",
            domain="technical",
            quality_speed="quality",
            confidence=0.75,
            latency_ms=42.0,
            from_cache=True,
        )
        assert intent.task_type == "generation"
        assert intent.domain == "technical"
        assert intent.quality_speed == "quality"
        assert intent.confidence == 0.75
        assert intent.latency_ms == 42.0
        assert intent.from_cache is True

    def test_validation_constants_task_types(self):
        """[IBR-DATA-01] IBR_TASK_TYPES has exactly 8 values."""
        expected = {
            "generation",
            "analysis",
            "refactoring",
            "summarization",
            "creative",
            "reasoning",
            "lookup",
            "translation",
        }
        assert expected == IBR_TASK_TYPES
        assert len(IBR_TASK_TYPES) == 8

    def test_validation_constants_domains(self):
        """[IBR-DATA-01] IBR_DOMAINS has exactly 6 values."""
        expected = {"code", "technical", "legal", "business", "creative_writing", "general"}
        assert expected == IBR_DOMAINS
        assert len(IBR_DOMAINS) == 6

    def test_validation_constants_quality_speed(self):
        """[IBR-DATA-01] IBR_QUALITY_SPEED has exactly 3 values."""
        expected = {"quality", "balanced", "speed"}
        assert expected == IBR_QUALITY_SPEED
        assert len(IBR_QUALITY_SPEED) == 3

    def test_equality(self):
        """[IBR-DATA-01] Two ClassifiedIntents with same fields are equal."""
        a = ClassifiedIntent("analysis", "code", "balanced", 0.9, 15.0, False)
        b = ClassifiedIntent("analysis", "code", "balanced", 0.9, 15.0, False)
        assert a == b


# ---------------------------------------------------------------------------
# SpectrographScore
# ---------------------------------------------------------------------------


class TestSpectrographScore:
    """[IBR-DATA-01] SpectrographScore frozen dataclass."""

    def test_frozen(self):
        """[IBR-DATA-01] SpectrographScore is immutable."""
        fs = SpectrographScore(score=0.8, confidence=1.0, sample_count=10)
        with pytest.raises(dataclasses.FrozenInstanceError):
            fs.score = 0.5  # type: ignore[misc]

    def test_neutral_default_values(self):
        """[IBR-DATA-01] IBR_NEUTRAL_SPECTROGRAPH has correct default values."""
        assert IBR_NEUTRAL_SPECTROGRAPH.score == 0.5
        assert IBR_NEUTRAL_SPECTROGRAPH.confidence == 0.0
        assert IBR_NEUTRAL_SPECTROGRAPH.sample_count == 0

    def test_fields_accessible(self):
        """[IBR-DATA-01] SpectrographScore fields are accessible."""
        fs = SpectrographScore(score=0.3, confidence=0.7, sample_count=25)
        assert fs.score == 0.3
        assert fs.confidence == 0.7
        assert fs.sample_count == 25


# ---------------------------------------------------------------------------
# ModelSpectrographProfile
# ---------------------------------------------------------------------------


class TestModelSpectrographProfile:
    """[IBR-DATA-01] ModelSpectrographProfile frozen dataclass."""

    def test_frozen(self):
        """[IBR-DATA-01] ModelSpectrographProfile is immutable."""
        profile = ModelSpectrographProfile(
            model_id="test",
            version=1,
            updated_at="2026-01-01T00:00:00Z",
            task_scores={},
            domain_scores={},
            qs_scores={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            profile.model_id = "other"  # type: ignore[misc]

    def test_construct_with_partial_scores(self):
        """[IBR-DATA-01] ModelSpectrographProfile can be constructed with partial score dicts."""
        fs = SpectrographScore(score=0.9, confidence=1.0, sample_count=0)
        profile = ModelSpectrographProfile(
            model_id="partial-test",
            version=1,
            updated_at="2026-01-01T00:00:00Z",
            task_scores={"analysis": fs},
            domain_scores={},
            qs_scores={"quality": fs},
        )
        assert profile.model_id == "partial-test"
        assert "analysis" in profile.task_scores
        assert len(profile.domain_scores) == 0
        assert "quality" in profile.qs_scores

    def test_full_profile_construction(self):
        """[IBR-DATA-01] ModelSpectrographProfile can hold full taxonomy."""
        fs = SpectrographScore(score=0.5, confidence=0.0, sample_count=0)
        task_scores = dict.fromkeys(IBR_TASK_TYPES, fs)
        domain_scores = dict.fromkeys(IBR_DOMAINS, fs)
        qs_scores = dict.fromkeys(IBR_QUALITY_SPEED, fs)
        profile = ModelSpectrographProfile(
            model_id="full",
            version=2,
            updated_at="2026-06-18T00:00:00Z",
            task_scores=task_scores,
            domain_scores=domain_scores,
            qs_scores=qs_scores,
        )
        assert len(profile.task_scores) == 8
        assert len(profile.domain_scores) == 6
        assert len(profile.qs_scores) == 3


# ---------------------------------------------------------------------------
# IBRScoringContext
# ---------------------------------------------------------------------------


class TestIBRScoringContext:
    """[IBR-DATA-01] IBRScoringContext frozen dataclass."""

    def test_frozen(self):
        """[IBR-DATA-01] IBRScoringContext is immutable."""
        ctx = IBRScoringContext(
            classified_intent=None,
            spectrograph_profiles={},
            spectrograph_match_weight=0.15,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            ctx.spectrograph_match_weight = 0.0  # type: ignore[misc]

    def test_none_classified_intent_allowed(self):
        """[IBR-DATA-01] IBRScoringContext allows None classified_intent."""
        ctx = IBRScoringContext(
            classified_intent=None,
            spectrograph_profiles={},
            spectrograph_match_weight=0.0,
        )
        assert ctx.classified_intent is None

    def test_with_classified_intent(self):
        """[IBR-DATA-01] IBRScoringContext stores a ClassifiedIntent."""
        intent = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=0.9,
            latency_ms=15.0,
            from_cache=False,
        )
        ctx = IBRScoringContext(
            classified_intent=intent,
            spectrograph_profiles={},
            spectrograph_match_weight=0.15,
        )
        assert ctx.classified_intent is intent


# ---------------------------------------------------------------------------
# ScoringWeightsConfig with spectrograph_match
# ---------------------------------------------------------------------------


class TestScoringWeightsConfigIBR:
    """[IBR-DATA-03] ScoringWeightsConfig with spectrograph_match dimension."""

    def test_spectrograph_match_zero_backward_compatible(self):
        """[IBR-DATA-02] spectrograph_match=0.0 sums to 1.0 (v0.3.0)."""
        config = ScoringWeightsConfig()
        assert config.spectrograph_match == 0.0
        total = (
            config.cost
            + config.latency
            + config.priority
            + config.queue
            + config.health
            + config.spectrograph_match
        )
        assert abs(total - 1.0) < 1e-9

    def test_spectrograph_match_0_15_with_adjusted_weights(self):
        """[IBR-DATA-03] spectrograph_match=0.15 sums to 1.0."""
        config = ScoringWeightsConfig(
            cost=0.30,
            latency=0.20,
            priority=0.15,
            queue=0.10,
            health=0.10,
            spectrograph_match=0.15,
        )
        total = (
            config.cost
            + config.latency
            + config.priority
            + config.queue
            + config.health
            + config.spectrograph_match
        )
        assert abs(total - 1.0) < 1e-9

    def test_invalid_sum_raises(self):
        """[IBR-DATA-03] Weights that do not sum to 1.0 raise AssertionError."""
        with pytest.raises(AssertionError, match="Weights must sum to 1.0"):
            ScoringWeightsConfig(
                cost=0.35,
                latency=0.25,
                priority=0.20,
                queue=0.10,
                health=0.10,
                spectrograph_match=0.15,
            )

    def test_spectrograph_match_0_05_cost_governor(self):
        """[IBR-SCORE-05] Cost governor weights sum to 1.0."""
        config = ScoringWeightsConfig(
            cost=0.65,
            latency=0.10,
            priority=0.10,
            queue=0.05,
            health=0.05,
            spectrograph_match=0.05,
        )
        total = (
            config.cost
            + config.latency
            + config.priority
            + config.queue
            + config.health
            + config.spectrograph_match
        )
        assert abs(total - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# EngineResponse IBR fields
# ---------------------------------------------------------------------------


class TestEngineResponseIBRFields:
    """[IBR-DATA-02] EngineResponse gains optional IBR fields."""

    def test_default_ibr_fields(self):
        """[IBR-DATA-02] EngineResponse IBR fields default to None/False."""
        resp = EngineResponse(
            content="hello",
            backend_used="test",
            backend_tier=BackendTier.COMPLEX,
            tokens_in=10,
            tokens_out=5,
            estimated_cost_usd=0.001,
            latency_ms=50.0,
            was_fallback=False,
            fallback_chain=[],
        )
        assert resp.classified_intent is None
        assert resp.spectrograph_match_score is None
        assert resp.ibr_active is False

    def test_can_set_ibr_fields(self):
        """[IBR-DATA-02] EngineResponse can be constructed with IBR fields populated."""
        intent = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=0.9,
            latency_ms=15.0,
            from_cache=False,
        )
        resp = EngineResponse(
            content="hello",
            backend_used="test",
            backend_tier=BackendTier.COMPLEX,
            tokens_in=10,
            tokens_out=5,
            estimated_cost_usd=0.001,
            latency_ms=50.0,
            was_fallback=False,
            fallback_chain=[],
            classified_intent=intent,
            spectrograph_match_score=0.82,
            ibr_active=True,
        )
        assert resp.classified_intent is intent
        assert resp.spectrograph_match_score == 0.82
        assert resp.ibr_active is True

    def test_ibr_fields_frozen(self):
        """[IBR-DATA-01] EngineResponse IBR fields are frozen."""
        resp = EngineResponse(
            content="hello",
            backend_used="test",
            backend_tier=BackendTier.COMPLEX,
            tokens_in=10,
            tokens_out=5,
            estimated_cost_usd=0.001,
            latency_ms=50.0,
            was_fallback=False,
            fallback_chain=[],
            ibr_active=True,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            resp.ibr_active = False  # type: ignore[misc]
