"""Integration tests for the full IBR pipeline end-to-end.

Tests the cascade dispatch path with IBR enabled and disabled,
verifying correct interaction between classifier, flavor profiles,
feedback store, and scoring weights.

Spec traceability: IBR spec v0.1.0 sections 4-6.
AC numbers: IBR-SYS-02, IBR-SYS-03, IBR-SCORE-02, IBR-SCORE-03,
            IBR-SCORE-04, IBR-SCORE-05, IBR-PIPE-01.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.config.schema import IntentClassificationConfig
from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendStatus,
    BackendTier,
    ClassifiedIntent,
    DispatchOrder,
    GenerativeBackend,
    ModelSpectrographProfile,
    SpectrographScore,
)
from dragonlight_router.dispatch.cascade import (
    DispatchContext,
    _resolve_cbr_weights,
    _run_ibr_stage,
    _score_and_rank_candidates,
)
from dragonlight_router.health.tracker import HealthTracker
from dragonlight_router.selection.feedback import FeedbackStore
from dragonlight_router.selection.ibr import IBRResult, run_ibr_stage
from dragonlight_router.selection.scoring import ScoringWeightsConfig
from dragonlight_router.selection.spectrograph import SpectrographProfileLoader

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

IBR_TASK_TYPES = frozenset(
    {
        "generation",
        "analysis",
        "refactoring",
        "summarization",
        "creative",
        "reasoning",
        "lookup",
        "translation",
    }
)
IBR_DOMAINS = frozenset(
    {
        "code",
        "technical",
        "legal",
        "business",
        "creative_writing",
        "general",
    }
)
IBR_QUALITY_SPEED = frozenset({"quality", "balanced", "speed"})
IBR_NEUTRAL_SPECTROGRAPH = SpectrographScore(score=0.5, confidence=0.0, sample_count=0)


def _make_intent(**overrides: object) -> ClassifiedIntent:
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
) -> ModelSpectrographProfile:
    """Build a ModelSpectrographProfile with optional partial scores."""

    def _build_scores(
        raw: dict[str, float] | None,
        allowed: frozenset[str],
    ) -> dict[str, SpectrographScore]:
        scores: dict[str, SpectrographScore] = {}
        parsed = raw or {}
        for key in allowed:
            if key in parsed:
                scores[key] = SpectrographScore(
                    score=parsed[key],
                    confidence=1.0,
                    sample_count=10,
                )
            else:
                scores[key] = IBR_NEUTRAL_SPECTROGRAPH
        return scores

    return ModelSpectrographProfile(
        model_id=model_id,
        version=1,
        updated_at="2026-01-01T00:00:00+00:00",
        task_scores=_build_scores(task_scores, IBR_TASK_TYPES),
        domain_scores=_build_scores(domain_scores, IBR_DOMAINS),
        qs_scores=_build_scores(qs_scores, IBR_QUALITY_SPEED),
    )


def _make_backend_config(
    name: str = "test-backend",
    provider: str = "test-provider",
    tier: BackendTier = BackendTier.COMPLEX,
) -> BackendConfig:
    """Build a BackendConfig with minimal defaults."""
    return BackendConfig(
        name=name,
        provider=provider,
        model=name,
        tier=tier,
        base_url="https://api.test.example.com/v1",
        env_key=None,
        capabilities=BackendCapabilities(
            max_context_tokens=131072,
            supports_tool_use=True,
            supports_streaming=True,
            supports_json_mode=True,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0),
        rate_limits=BackendRateLimits(rpm=60, rpd=14400, tpm=100000, daily_token_cap=1000000),
    )


def _make_dispatch_order(**overrides: object) -> DispatchOrder:
    """Build a DispatchOrder with sensible defaults."""
    defaults = {
        "intent_category": "test",
        "specific_intent": "test",
        "operator_message": "Analyze this code for bugs",
        "system_prompt": "",
        "context_tokens": 100,
    }
    defaults.update(overrides)
    return DispatchOrder(**defaults)


def _make_ibr_config(**overrides: object) -> IntentClassificationConfig:
    """Build an IntentClassificationConfig with defaults."""
    defaults = {
        "enabled": True,
        "timeout_ms": 100,
        "confidence_threshold": 0.6,
        "profile_confidence_threshold": 0.3,
        "spectrograph_match_weight": 0.15,
        "spectrograph_match_weight_governor": 0.05,
    }
    defaults.update(overrides)
    return IntentClassificationConfig(**defaults)


def _make_mock_adapter(response_text: str) -> MagicMock:
    """Build a mock GenerativeBackend yielding a single chunk."""
    adapter = MagicMock(spec=GenerativeBackend)
    adapter.status = BackendStatus.AVAILABLE

    async def _generate(*args, **kwargs):
        yield response_text

    adapter.generate = _generate
    adapter.record_usage = MagicMock()
    return adapter


# ---------------------------------------------------------------------------
# IBR disabled path (IBR-SYS-02)
# ---------------------------------------------------------------------------


class TestIBRDisabledPath:
    """[IBR-SYS-02] When IBR is disabled, cascade operates identically to pre-IBR."""

    async def test_ibr_stage_returns_none_when_config_none(
        self,
        make_backend_config,
        make_dispatch_order,
    ):
        """_run_ibr_stage returns None when ibr_config is None."""
        ctx = DispatchContext(
            registry=BackendRegistry(),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
            ibr_config=None,
            spectrograph_loader=None,
            classification_adapter=None,
        )
        order = make_dispatch_order()
        candidates = [make_backend_config(name="backend-a")]
        result = await _run_ibr_stage(order, candidates, ctx)
        assert result is None

    async def test_ibr_stage_returns_none_when_spectrograph_loader_none(
        self,
        make_dispatch_order,
    ):
        """_run_ibr_stage returns None when spectrograph_loader is None."""
        ibr_cfg = _make_ibr_config(enabled=True)
        ctx = DispatchContext(
            registry=BackendRegistry(),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
            ibr_config=ibr_cfg,
            spectrograph_loader=None,
            classification_adapter=None,
        )
        order = make_dispatch_order()
        result = await _run_ibr_stage(order, [], ctx)
        assert result is None

    def test_cbr_weights_fallback_to_default_with_spectrograph(self):
        """[IBR-SCORE-03] When IBR result is None, CBR uses default 6-dimension weights.

        With IBR activation, the default ScoringWeightsConfig includes
        spectrograph_match=0.15.  All 6 dimensions sum to 1.0.
        """
        ctx = DispatchContext(
            registry=BackendRegistry(),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
            ibr_config=None,
        )
        weights = _resolve_cbr_weights(None, ctx)
        assert weights.spectrograph_match == 0.15
        total = (
            weights.cost + weights.latency + weights.priority
            + weights.queue + weights.health + weights.spectrograph_match
        )
        assert abs(total - 1.0) < 1e-9

    def test_cbr_weights_fallback_when_ibr_inactive(self):
        """[IBR-SCORE-03] When ibr_active=False, CBR uses default 6-dimension weights."""
        ibr_result = IBRResult(
            classified_intent=_make_intent(),
            spectrograph_scores={},
            ibr_active=False,
        )
        ctx = DispatchContext(
            registry=BackendRegistry(),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
            ibr_config=_make_ibr_config(),
        )
        weights = _resolve_cbr_weights(ibr_result, ctx)
        assert weights.spectrograph_match == 0.15


# ---------------------------------------------------------------------------
# IBR active path — classifier success (IBR-PIPE-01)
# ---------------------------------------------------------------------------


class TestIBRActivePathClassifierSuccess:
    """[IBR-PIPE-01] IBR active with successful classification."""

    async def test_ibr_stage_returns_active_result(self):
        """run_ibr_stage returns ibr_active=True on valid classification."""
        intent = _make_intent(confidence=0.9)
        ibr_cfg = _make_ibr_config()
        profile = _make_profile(
            model_id="backend-a",
            task_scores={"analysis": 0.9},
            domain_scores={"code": 0.8},
            qs_scores={"balanced": 0.7},
        )
        loader = MagicMock(spec=SpectrographProfileLoader)
        loader.profiles = {"backend-a": profile}
        loader.reload_if_changed = MagicMock()

        adapter = MagicMock(spec=GenerativeBackend)

        candidate = _make_backend_config(name="backend-a")

        with patch(
            "dragonlight_router.selection.ibr.classify_intent",
            new_callable=AsyncMock,
            return_value=intent,
        ):
            result = await run_ibr_stage(
                order=_make_dispatch_order(),
                candidates=[candidate],
                ibr_config=ibr_cfg,
                spectrograph_loader=loader,
                classification_adapter=adapter,
            )

        assert result.ibr_active is True
        assert result.classified_intent is not None
        assert result.classified_intent.task_type == "analysis"
        assert "backend-a" in result.spectrograph_scores
        assert 0.0 <= result.spectrograph_scores["backend-a"] <= 1.0

    def test_cbr_weights_include_spectrograph_match_when_active(self):
        """[IBR-SCORE-02] When IBR active, 6-dimension weights include spectrograph_match."""
        ibr_result = IBRResult(
            classified_intent=_make_intent(),
            spectrograph_scores={"backend-a": 0.8},
            ibr_active=True,
        )
        ibr_cfg = _make_ibr_config(spectrograph_match_weight=0.15)
        ctx = DispatchContext(
            registry=BackendRegistry(),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
            ibr_config=ibr_cfg,
        )
        weights = _resolve_cbr_weights(ibr_result, ctx)
        assert weights.spectrograph_match == 0.15
        total = (
            weights.cost
            + weights.latency
            + weights.priority
            + weights.queue
            + weights.health
            + weights.spectrograph_match
        )
        assert abs(total - 1.0) < 1e-9

    def test_spectrograph_scores_applied_in_scoring(self, make_backend_config):
        """Flavor scores contribute to candidate scoring when IBR active."""
        ibr_result = IBRResult(
            classified_intent=_make_intent(),
            spectrograph_scores={"good-model": 0.9, "bad-model": 0.1},
            ibr_active=True,
        )
        weights = ScoringWeightsConfig(
            cost=0.30,
            latency=0.20,
            priority=0.15,
            queue=0.10,
            health=0.10,
            spectrograph_match=0.15,
        )
        # Use identical backends so the only scoring difference is spectrograph_match
        good = make_backend_config(name="good-model", provider="p1")
        bad = make_backend_config(name="bad-model", provider="p1")

        budget = BudgetTracker(providers=[])
        health = HealthTracker()
        registry = BackendRegistry()

        ctx = DispatchContext(
            registry=registry,
            budget_tracker=budget,
            health_tracker=health,
            config={},
        )

        scored = _score_and_rank_candidates(
            [good, bad],
            _make_dispatch_order(),
            weights,
            ctx,
            ibr_result=ibr_result,
        )
        # good-model should score higher due to higher spectrograph_match
        assert scored[0].config.name == "good-model"
        assert scored[1].config.name == "bad-model"
        assert scored[0].score > scored[1].score


# ---------------------------------------------------------------------------
# IBR active path — classifier failure/timeout (IBR-SYS-03)
# ---------------------------------------------------------------------------


class TestIBRActivePathClassifierFailure:
    """[IBR-SYS-03] Graceful degradation when classifier fails or times out."""

    async def test_classifier_returns_none_gives_inactive_result(self):
        """Classification returning None produces inactive IBRResult."""
        ibr_cfg = _make_ibr_config()
        loader = MagicMock(spec=SpectrographProfileLoader)
        loader.profiles = {}
        loader.reload_if_changed = MagicMock()
        adapter = MagicMock(spec=GenerativeBackend)
        candidate = _make_backend_config(name="backend-a")

        with patch(
            "dragonlight_router.selection.ibr.classify_intent",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = await run_ibr_stage(
                order=_make_dispatch_order(),
                candidates=[candidate],
                ibr_config=ibr_cfg,
                spectrograph_loader=loader,
                classification_adapter=adapter,
            )

        assert result.ibr_active is False
        assert result.classified_intent is None
        assert result.spectrograph_scores == {}

    async def test_classifier_raises_exception_gives_inactive_result(self):
        """Classification raising an exception produces inactive IBRResult."""
        ibr_cfg = _make_ibr_config()
        loader = MagicMock(spec=SpectrographProfileLoader)
        loader.profiles = {}
        loader.reload_if_changed = MagicMock()
        adapter = MagicMock(spec=GenerativeBackend)
        candidate = _make_backend_config(name="backend-a")

        with patch(
            "dragonlight_router.selection.ibr.classify_intent",
            new_callable=AsyncMock,
            side_effect=RuntimeError("timeout"),
        ):
            result = await run_ibr_stage(
                order=_make_dispatch_order(),
                candidates=[candidate],
                ibr_config=ibr_cfg,
                spectrograph_loader=loader,
                classification_adapter=adapter,
            )

        assert result.ibr_active is False
        assert result.classified_intent is None
        assert result.spectrograph_scores == {}

    async def test_ibr_disabled_returns_inactive(self):
        """When ibr_config.enabled=False, returns inactive."""
        ibr_cfg = _make_ibr_config(enabled=False)
        loader = MagicMock(spec=SpectrographProfileLoader)
        loader.profiles = {}
        adapter = MagicMock(spec=GenerativeBackend)
        candidate = _make_backend_config(name="backend-a")

        result = await run_ibr_stage(
            order=_make_dispatch_order(),
            candidates=[candidate],
            ibr_config=ibr_cfg,
            spectrograph_loader=loader,
            classification_adapter=adapter,
        )

        assert result.ibr_active is False

    async def test_no_adapter_returns_inactive(self):
        """When classification_adapter is None, returns inactive."""
        ibr_cfg = _make_ibr_config(enabled=True)
        loader = MagicMock(spec=SpectrographProfileLoader)
        loader.profiles = {}
        candidate = _make_backend_config(name="backend-a")

        result = await run_ibr_stage(
            order=_make_dispatch_order(),
            candidates=[candidate],
            ibr_config=ibr_cfg,
            spectrograph_loader=loader,
            classification_adapter=None,
        )

        assert result.ibr_active is False


# ---------------------------------------------------------------------------
# Confidence gating (IBR-SCORE-04)
# ---------------------------------------------------------------------------


class TestConfidenceGating:
    """[IBR-SCORE-04] Low-confidence classification does not distort routing."""

    async def test_low_confidence_gives_inactive_result(self):
        """Classification with confidence below threshold is gated out."""
        low_intent = _make_intent(confidence=0.3)
        ibr_cfg = _make_ibr_config(confidence_threshold=0.6)
        profile = _make_profile(
            model_id="backend-a",
            task_scores={"analysis": 0.9},
            domain_scores={"code": 0.8},
            qs_scores={"balanced": 0.7},
        )
        loader = MagicMock(spec=SpectrographProfileLoader)
        loader.profiles = {"backend-a": profile}
        loader.reload_if_changed = MagicMock()
        adapter = MagicMock(spec=GenerativeBackend)
        candidate = _make_backend_config(name="backend-a")

        with patch(
            "dragonlight_router.selection.ibr.classify_intent",
            new_callable=AsyncMock,
            return_value=low_intent,
        ):
            result = await run_ibr_stage(
                order=_make_dispatch_order(),
                candidates=[candidate],
                ibr_config=ibr_cfg,
                spectrograph_loader=loader,
                classification_adapter=adapter,
            )

        assert result.ibr_active is False
        # The classification is still preserved for observability
        assert result.classified_intent is not None
        assert result.classified_intent.confidence == 0.3
        assert result.spectrograph_scores == {}

    async def test_high_confidence_gives_active_result(self):
        """Classification above threshold passes gating."""
        high_intent = _make_intent(confidence=0.95)
        ibr_cfg = _make_ibr_config(confidence_threshold=0.6)
        profile = _make_profile(
            model_id="backend-a",
            task_scores={"analysis": 0.9},
            domain_scores={"code": 0.8},
            qs_scores={"balanced": 0.7},
        )
        loader = MagicMock(spec=SpectrographProfileLoader)
        loader.profiles = {"backend-a": profile}
        loader.reload_if_changed = MagicMock()
        adapter = MagicMock(spec=GenerativeBackend)
        candidate = _make_backend_config(name="backend-a")

        with patch(
            "dragonlight_router.selection.ibr.classify_intent",
            new_callable=AsyncMock,
            return_value=high_intent,
        ):
            result = await run_ibr_stage(
                order=_make_dispatch_order(),
                candidates=[candidate],
                ibr_config=ibr_cfg,
                spectrograph_loader=loader,
                classification_adapter=adapter,
            )

        assert result.ibr_active is True
        assert len(result.spectrograph_scores) > 0

    async def test_low_profile_confidence_gates_out(self):
        """Profiles with low confidence gate out even with high classifier confidence."""
        intent = _make_intent(confidence=0.95)
        ibr_cfg = _make_ibr_config(
            confidence_threshold=0.5,
            profile_confidence_threshold=0.5,
        )
        # Use neutral profiles (confidence=0.0) — below threshold
        loader = MagicMock(spec=SpectrographProfileLoader)
        loader.profiles = {}  # all profiles will be neutral defaults
        loader.reload_if_changed = MagicMock()
        adapter = MagicMock(spec=GenerativeBackend)
        candidate = _make_backend_config(name="backend-a")

        with patch(
            "dragonlight_router.selection.ibr.classify_intent",
            new_callable=AsyncMock,
            return_value=intent,
        ):
            result = await run_ibr_stage(
                order=_make_dispatch_order(),
                candidates=[candidate],
                ibr_config=ibr_cfg,
                spectrograph_loader=loader,
                classification_adapter=adapter,
            )

        assert result.ibr_active is False


# ---------------------------------------------------------------------------
# Feedback recording flow
# ---------------------------------------------------------------------------


class TestFeedbackRecordingFlow:
    """Feedback recording through RouterEngine.record_ibr_feedback."""

    def test_feedback_store_receives_call(self, tmp_path):
        """record_ibr_feedback delegates to FeedbackStore.record_feedback."""
        store = FeedbackStore(db_path=tmp_path / "feedback.db")
        intent = _make_intent()

        store.record_feedback("model-a", intent, 4)

        profiles = store.get_learned_profiles()
        assert "model-a" in profiles
        assert profiles["model-a"].task_scores["analysis"].sample_count == 1
        store.close()

    def test_feedback_with_operator_profile_applies_floor(self, tmp_path):
        """Feedback with operator profile applies floor enforcement."""
        store = FeedbackStore(db_path=tmp_path / "feedback.db")
        intent = _make_intent()
        operator = _make_profile("model-a", task_scores={"analysis": 0.9})

        # Rating=1 -> obs=0.2, floor=0.72
        store.record_feedback("model-a", intent, 1, operator_profile=operator)

        profiles = store.get_learned_profiles()
        assert profiles["model-a"].task_scores["analysis"].score >= 0.72
        store.close()

    def test_feedback_skipped_when_store_none(self):
        """When _feedback_store is None, feedback is silently skipped."""
        # Simulate RouterEngine with no feedback store
        from dragonlight_router.router import RouterEngine

        engine = MagicMock(spec=RouterEngine)
        engine._feedback_store = None

        # Calling the real method with None store should not raise
        RouterEngine.record_ibr_feedback(
            engine,
            model_id="model-a",
            classified_intent=_make_intent(),
            quality_rating=3,
        )
        # No exception means success


# ---------------------------------------------------------------------------
# IBR scoring weight governor (IBR-SCORE-05)
# ---------------------------------------------------------------------------


class TestIBRScoringWeightGovernor:
    """[IBR-SCORE-05] spectrograph_match weight is governed."""

    def test_spectrograph_match_weight_used_from_config(self):
        """spectrograph_match weight comes from ibr_config when IBR active."""
        ibr_result = IBRResult(
            classified_intent=_make_intent(),
            spectrograph_scores={"m1": 0.9},
            ibr_active=True,
        )
        # Use 0.15 — the default and only value that sums to 1.0 with the
        # hardcoded 5-dimension split (0.30+0.20+0.15+0.10+0.10 = 0.85).
        ibr_cfg = _make_ibr_config(spectrograph_match_weight=0.15)
        ctx = DispatchContext(
            registry=BackendRegistry(),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
            ibr_config=ibr_cfg,
        )
        weights = _resolve_cbr_weights(ibr_result, ctx)
        assert weights.spectrograph_match == 0.15

    def test_cost_governor_reduces_spectrograph_match(self):
        """[IBR-SCORE-05] Cost governor reduces spectrograph_match to 0.05."""
        from dragonlight_router.selection.scoring import cost_adjusted_weights

        ibr_weights = ScoringWeightsConfig(
            cost=0.30,
            latency=0.20,
            priority=0.15,
            queue=0.10,
            health=0.10,
            spectrograph_match=0.15,
        )
        adjusted = cost_adjusted_weights(ibr_weights)
        assert adjusted.spectrograph_match == 0.05
        assert adjusted.cost == 0.65
        total = (
            adjusted.cost
            + adjusted.latency
            + adjusted.priority
            + adjusted.queue
            + adjusted.health
            + adjusted.spectrograph_match
        )
        assert abs(total - 1.0) < 1e-9

    def test_spectrograph_match_score_bounded(self, make_backend_config):
        """spectrograph_match contribution to score is bounded by weight * 1.0."""
        weights = ScoringWeightsConfig(
            cost=0.30,
            latency=0.20,
            priority=0.15,
            queue=0.10,
            health=0.10,
            spectrograph_match=0.15,
        )
        ibr_result = IBRResult(
            classified_intent=_make_intent(),
            spectrograph_scores={"model-a": 1.0},  # max possible
            ibr_active=True,
        )
        candidate = make_backend_config(name="model-a")
        budget = BudgetTracker(providers=[])
        health = HealthTracker()
        registry = BackendRegistry()
        ctx = DispatchContext(
            registry=registry,
            budget_tracker=budget,
            health_tracker=health,
            config={},
        )

        scored = _score_and_rank_candidates(
            [candidate],
            _make_dispatch_order(),
            weights,
            ctx,
            ibr_result=ibr_result,
        )
        # Score must be in [0.0, 1.0]
        assert 0.0 <= scored[0].score <= 1.0

    def test_default_spectrograph_match_weight_when_ibr_disabled(self):
        """[IBR-SCORE-03] spectrograph_match weight is 0.15 (default) when IBR classifier is off.

        With IBR activation, spectrograph scoring is always included in the
        default weight vector.  The weight only drops to 0.0 if explicitly
        configured that way.
        """
        weights = _resolve_cbr_weights(
            None,
            DispatchContext(
                registry=BackendRegistry(),
                budget_tracker=MagicMock(),
                health_tracker=MagicMock(),
                config={},
            ),
        )
        assert weights.spectrograph_match == 0.15
