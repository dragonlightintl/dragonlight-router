"""Tests for IBR cascade activation: spectrograph feedback merge, finer intent
categories, and per-request CBR weight adjustment (three-tier stakes).

Covers the design doc: docs/design/ibr-cascade-activation.md
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dragonlight_router.core.types import (
    BackendTier,
    ClassifiedIntent,
    DispatchOrder,
    ModelSpectrographProfile,
    SpectrographScore,
)
from dragonlight_router.selection.mbr import _INTENT_TIER_FLOOR
from dragonlight_router.selection.scoring import (
    ScoringWeightsConfig,
    _HIGH_STAKES_INTENTS,
    _HIGH_STAKES_WEIGHTS,
    _LOW_STAKES_INTENTS,
    _LOW_STAKES_WEIGHTS,
    _MID_STAKES_INTENTS,
    _MID_STAKES_WEIGHTS,
    _STAKES_TO_WEIGHTS,
    classify_request_stakes,
    intent_weights_for_category,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(
    intent_category: str = "test",
    context_tokens: int = 0,
    requires_tool_use: bool = False,
    context_trust_tier: str | None = None,
) -> DispatchOrder:
    return DispatchOrder(
        intent_category=intent_category,
        specific_intent="test",
        operator_message="test message",
        system_prompt="",
        context_tokens=context_tokens,
        requires_tool_use=requires_tool_use,
        context_trust_tier=context_trust_tier,
    )


# ---------------------------------------------------------------------------
# Section 1: Per-Request CBR Weight Adjustment (Three-Tier Stakes)
# ---------------------------------------------------------------------------


class TestClassifyRequestStakesLow:
    """Primary signal: low-stakes intent categories -> cost_optimized."""

    def test_test_generation(self):
        order = _make_order(intent_category="test_generation")
        assert classify_request_stakes(order) == "cost_optimized"

    def test_audit(self):
        order = _make_order(intent_category="audit")
        assert classify_request_stakes(order) == "cost_optimized"

    def test_summarization(self):
        order = _make_order(intent_category="summarization")
        assert classify_request_stakes(order) == "cost_optimized"

    def test_documentation(self):
        order = _make_order(intent_category="documentation")
        assert classify_request_stakes(order) == "cost_optimized"

    def test_test_fix(self):
        order = _make_order(intent_category="test_fix")
        assert classify_request_stakes(order) == "cost_optimized"

    def test_code_generation(self):
        order = _make_order(intent_category="code_generation")
        assert classify_request_stakes(order) == "cost_optimized"

    def test_data_analysis(self):
        order = _make_order(intent_category="data_analysis")
        assert classify_request_stakes(order) == "cost_optimized"

    def test_test_property(self):
        order = _make_order(intent_category="test_property")
        assert classify_request_stakes(order) == "cost_optimized"


class TestClassifyRequestStakesMid:
    """Primary signal: mid-stakes intent categories -> balanced."""

    def test_code_review(self):
        order = _make_order(intent_category="code_review")
        assert classify_request_stakes(order) == "balanced"

    def test_engineering_build(self):
        order = _make_order(intent_category="engineering_build")
        assert classify_request_stakes(order) == "balanced"

    def test_spec_writing(self):
        order = _make_order(intent_category="spec_writing")
        assert classify_request_stakes(order) == "balanced"

    def test_refactoring(self):
        order = _make_order(intent_category="refactoring")
        assert classify_request_stakes(order) == "balanced"

    def test_api_design(self):
        order = _make_order(intent_category="api_design")
        assert classify_request_stakes(order) == "balanced"


class TestClassifyRequestStakesHigh:
    """Primary signal: high-stakes intent categories -> capability_optimized."""

    def test_architecture(self):
        order = _make_order(intent_category="architecture")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_implementation(self):
        order = _make_order(intent_category="implementation")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_implementation_complex(self):
        order = _make_order(intent_category="implementation_complex")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_coherence_merge(self):
        order = _make_order(intent_category="coherence_merge")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_complex_reasoning(self):
        order = _make_order(intent_category="complex_reasoning")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_strategic_planning(self):
        order = _make_order(intent_category="strategic_planning")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_security_review(self):
        order = _make_order(intent_category="security_review")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_performance_optimization(self):
        order = _make_order(intent_category="performance_optimization")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_debugging(self):
        order = _make_order(intent_category="debugging")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_migration(self):
        order = _make_order(intent_category="migration")
        assert classify_request_stakes(order) == "capability_optimized"


class TestClassifyRequestStakesUnknown:
    """Unknown intent categories default to balanced."""

    def test_unknown_category(self):
        order = _make_order(intent_category="unknown_thing")
        assert classify_request_stakes(order) == "balanced"

    def test_empty_category(self):
        order = _make_order(intent_category="")
        assert classify_request_stakes(order) == "balanced"


class TestClassifyRequestStakesContextEscalation:
    """Secondary signals can escalate stakes but never reduce them."""

    def test_high_context_tokens_escalate_low_to_balanced(self):
        """test_generation (cost_optimized) + 9000 tokens -> balanced."""
        order = _make_order(intent_category="test_generation", context_tokens=9000)
        assert classify_request_stakes(order) == "balanced"

    def test_very_high_context_tokens_escalate_low_to_capability(self):
        """test_generation (cost_optimized) + 40000 tokens -> capability_optimized."""
        order = _make_order(intent_category="test_generation", context_tokens=40000)
        assert classify_request_stakes(order) == "capability_optimized"

    def test_tool_use_escalates_low_to_balanced(self):
        """documentation (cost_optimized) + requires_tool_use -> balanced."""
        order = _make_order(intent_category="documentation", requires_tool_use=True)
        assert classify_request_stakes(order) == "balanced"

    def test_trust_tier_escalates_to_capability(self):
        """summarization (cost_optimized) + context_trust_tier=trusted -> capability_optimized."""
        order = _make_order(intent_category="summarization", context_trust_tier="trusted")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_trust_tier_local_escalates_to_capability(self):
        """summarization (cost_optimized) + context_trust_tier=local -> capability_optimized."""
        order = _make_order(intent_category="summarization", context_trust_tier="local")
        assert classify_request_stakes(order) == "capability_optimized"

    def test_high_stakes_not_reduced_by_context(self):
        """architecture (capability_optimized) stays capability_optimized regardless of context."""
        order = _make_order(intent_category="architecture", context_tokens=100)
        assert classify_request_stakes(order) == "capability_optimized"

    def test_mid_stakes_with_high_context_escalates(self):
        """code_review (balanced) + 40000 tokens -> capability_optimized."""
        order = _make_order(intent_category="code_review", context_tokens=40000)
        assert classify_request_stakes(order) == "capability_optimized"

    def test_boundary_8000_no_escalation(self):
        """Exactly 8000 tokens does NOT escalate (> 8000 required)."""
        order = _make_order(intent_category="test_generation", context_tokens=8000)
        assert classify_request_stakes(order) == "cost_optimized"

    def test_boundary_8001_escalates(self):
        """8001 tokens escalates to balanced."""
        order = _make_order(intent_category="test_generation", context_tokens=8001)
        assert classify_request_stakes(order) == "balanced"

    def test_boundary_32000_no_full_escalation(self):
        """Exactly 32000 tokens does NOT escalate to capability_optimized (> 32000 required)."""
        order = _make_order(intent_category="test_generation", context_tokens=32000)
        # 32000 is not > 32000, so the elif branch fires: > 8000 -> balanced
        assert classify_request_stakes(order) == "balanced"

    def test_boundary_32001_escalates_full(self):
        """32001 tokens escalates to capability_optimized."""
        order = _make_order(intent_category="test_generation", context_tokens=32001)
        assert classify_request_stakes(order) == "capability_optimized"


# ---------------------------------------------------------------------------
# Section 2: Intent Weight Profiles
# ---------------------------------------------------------------------------


class TestIntentWeightsForCategory:
    """intent_weights_for_category returns correct profiles for all three tiers."""

    def test_low_stakes_returns_low_weights(self):
        assert intent_weights_for_category("test_generation") == _LOW_STAKES_WEIGHTS

    def test_high_stakes_returns_high_weights(self):
        assert intent_weights_for_category("architecture") == _HIGH_STAKES_WEIGHTS

    def test_mid_stakes_returns_mid_weights(self):
        assert intent_weights_for_category("code_review") == _MID_STAKES_WEIGHTS

    def test_unknown_returns_default(self):
        result = intent_weights_for_category("some_unknown_intent")
        assert result == ScoringWeightsConfig()

    def test_new_low_stakes_documentation(self):
        assert intent_weights_for_category("documentation") == _LOW_STAKES_WEIGHTS

    def test_new_low_stakes_test_fix(self):
        assert intent_weights_for_category("test_fix") == _LOW_STAKES_WEIGHTS

    def test_new_low_stakes_code_generation(self):
        assert intent_weights_for_category("code_generation") == _LOW_STAKES_WEIGHTS

    def test_new_high_stakes_security_review(self):
        assert intent_weights_for_category("security_review") == _HIGH_STAKES_WEIGHTS

    def test_new_high_stakes_performance_optimization(self):
        assert intent_weights_for_category("performance_optimization") == _HIGH_STAKES_WEIGHTS

    def test_new_high_stakes_debugging(self):
        assert intent_weights_for_category("debugging") == _HIGH_STAKES_WEIGHTS

    def test_new_high_stakes_migration(self):
        assert intent_weights_for_category("migration") == _HIGH_STAKES_WEIGHTS

    def test_new_mid_stakes_refactoring(self):
        assert intent_weights_for_category("refactoring") == _MID_STAKES_WEIGHTS

    def test_new_mid_stakes_api_design(self):
        assert intent_weights_for_category("api_design") == _MID_STAKES_WEIGHTS

    def test_new_mid_stakes_spec_writing(self):
        assert intent_weights_for_category("spec_writing") == _MID_STAKES_WEIGHTS

    def test_new_mid_stakes_engineering_build(self):
        assert intent_weights_for_category("engineering_build") == _MID_STAKES_WEIGHTS


class TestWeightProfilesSumToOne:
    """All weight profiles must sum to 1.0."""

    def test_low_stakes_weights_sum(self):
        total = (
            _LOW_STAKES_WEIGHTS.cost
            + _LOW_STAKES_WEIGHTS.latency
            + _LOW_STAKES_WEIGHTS.priority
            + _LOW_STAKES_WEIGHTS.spectrograph_match
            + _LOW_STAKES_WEIGHTS.queue
            + _LOW_STAKES_WEIGHTS.health
        )
        assert abs(total - 1.0) < 1e-9

    def test_mid_stakes_weights_sum(self):
        total = (
            _MID_STAKES_WEIGHTS.cost
            + _MID_STAKES_WEIGHTS.latency
            + _MID_STAKES_WEIGHTS.priority
            + _MID_STAKES_WEIGHTS.spectrograph_match
            + _MID_STAKES_WEIGHTS.queue
            + _MID_STAKES_WEIGHTS.health
        )
        assert abs(total - 1.0) < 1e-9

    def test_high_stakes_weights_sum(self):
        total = (
            _HIGH_STAKES_WEIGHTS.cost
            + _HIGH_STAKES_WEIGHTS.latency
            + _HIGH_STAKES_WEIGHTS.priority
            + _HIGH_STAKES_WEIGHTS.spectrograph_match
            + _HIGH_STAKES_WEIGHTS.queue
            + _HIGH_STAKES_WEIGHTS.health
        )
        assert abs(total - 1.0) < 1e-9


class TestStakesToWeightsMapping:
    """_STAKES_TO_WEIGHTS maps all three stakes levels to valid weight profiles."""

    def test_cost_optimized_maps_to_low(self):
        assert _STAKES_TO_WEIGHTS["cost_optimized"] is _LOW_STAKES_WEIGHTS

    def test_balanced_maps_to_mid(self):
        assert _STAKES_TO_WEIGHTS["balanced"] is _MID_STAKES_WEIGHTS

    def test_capability_optimized_maps_to_high(self):
        assert _STAKES_TO_WEIGHTS["capability_optimized"] is _HIGH_STAKES_WEIGHTS

    def test_all_three_present(self):
        assert set(_STAKES_TO_WEIGHTS.keys()) == {
            "cost_optimized",
            "balanced",
            "capability_optimized",
        }


class TestWeightProfileDimensions:
    """Verify the relative dimension emphasis for each profile."""

    def test_low_stakes_favors_cost(self):
        """Low-stakes: cost should be the highest-weighted dimension."""
        assert _LOW_STAKES_WEIGHTS.cost > _LOW_STAKES_WEIGHTS.priority
        assert _LOW_STAKES_WEIGHTS.cost > _LOW_STAKES_WEIGHTS.spectrograph_match

    def test_high_stakes_favors_priority_and_spectrograph(self):
        """High-stakes: priority and spectrograph should dominate."""
        assert _HIGH_STAKES_WEIGHTS.priority > _HIGH_STAKES_WEIGHTS.cost
        assert _HIGH_STAKES_WEIGHTS.spectrograph_match > _HIGH_STAKES_WEIGHTS.cost

    def test_mid_stakes_balanced(self):
        """Mid-stakes: priority is highest but cost is moderate."""
        assert _MID_STAKES_WEIGHTS.priority > _MID_STAKES_WEIGHTS.cost
        assert _MID_STAKES_WEIGHTS.spectrograph_match > _MID_STAKES_WEIGHTS.cost


# ---------------------------------------------------------------------------
# Section 3: Finer Intent Categories (MBR Tier Floors)
# ---------------------------------------------------------------------------


class TestMBRTierFloorNewCategories:
    """New intent categories produce correct tier floors."""

    def test_refactoring_moderate(self):
        assert _INTENT_TIER_FLOOR["refactoring"] == BackendTier.MODERATE

    def test_documentation_simple(self):
        assert _INTENT_TIER_FLOOR["documentation"] == BackendTier.SIMPLE

    def test_test_fix_simple(self):
        assert _INTENT_TIER_FLOOR["test_fix"] == BackendTier.SIMPLE

    def test_security_review_complex(self):
        assert _INTENT_TIER_FLOOR["security_review"] == BackendTier.COMPLEX

    def test_performance_optimization_complex(self):
        assert _INTENT_TIER_FLOOR["performance_optimization"] == BackendTier.COMPLEX

    def test_migration_moderate(self):
        assert _INTENT_TIER_FLOOR["migration"] == BackendTier.MODERATE

    def test_api_design_moderate(self):
        assert _INTENT_TIER_FLOOR["api_design"] == BackendTier.MODERATE

    def test_existing_categories_unchanged(self):
        """Verify existing categories are not affected by the additions."""
        assert _INTENT_TIER_FLOOR["complex_reasoning"] == BackendTier.COMPLEX
        assert _INTENT_TIER_FLOOR["architecture"] == BackendTier.COMPLEX
        assert _INTENT_TIER_FLOOR["implementation"] == BackendTier.MODERATE
        assert _INTENT_TIER_FLOOR["test_generation"] == BackendTier.SIMPLE
        assert _INTENT_TIER_FLOOR["summarization"] == BackendTier.SIMPLE


# ---------------------------------------------------------------------------
# Section 4: Intent Category Set Completeness
# ---------------------------------------------------------------------------


class TestIntentCategoryCompleteness:
    """All intent categories recognized by MBR should also have CBR weight profiles."""

    def test_all_mbr_categories_have_cbr_weights(self):
        """Every intent in _INTENT_TIER_FLOOR should be in LOW, MID, or HIGH stakes."""
        all_cbr = _LOW_STAKES_INTENTS | _MID_STAKES_INTENTS | _HIGH_STAKES_INTENTS
        for intent in _INTENT_TIER_FLOOR:
            assert intent in all_cbr, (
                f"Intent '{intent}' is in MBR tier floor but has no CBR weight profile "
                f"(not in _LOW/_MID/_HIGH_STAKES_INTENTS)"
            )

    def test_no_category_in_multiple_tiers(self):
        """No intent category should appear in more than one stakes tier."""
        low_mid = _LOW_STAKES_INTENTS & _MID_STAKES_INTENTS
        low_high = _LOW_STAKES_INTENTS & _HIGH_STAKES_INTENTS
        mid_high = _MID_STAKES_INTENTS & _HIGH_STAKES_INTENTS
        assert len(low_mid) == 0, f"Categories in both LOW and MID: {low_mid}"
        assert len(low_high) == 0, f"Categories in both LOW and HIGH: {low_high}"
        assert len(mid_high) == 0, f"Categories in both MID and HIGH: {mid_high}"


# ---------------------------------------------------------------------------
# Section 5: IBR Feedback Merge (GAP-SPEC-01)
# ---------------------------------------------------------------------------


class TestIBRFeedbackMerge:
    """Verify that feedback-learned profiles are merged into scoring."""

    @pytest.mark.asyncio
    async def test_feedback_merge_in_ibr(self):
        """Learned profiles merge with operator profiles in scoring."""
        from dragonlight_router.config.schema import IntentClassificationConfig
        from dragonlight_router.selection.ibr import _build_ibr_result

        intent = ClassifiedIntent(
            task_type="generation",
            domain="code",
            quality_speed="quality",
            confidence=0.9,
            latency_ms=100.0,
            from_cache=False,
        )

        # Create mock BackendConfig
        from dragonlight_router.core.types import (
            BackendCapabilities,
            BackendConfig,
            BackendCostProfile,
            BackendRateLimits,
        )

        candidate = BackendConfig(
            name="test-model",
            provider="test",
            model="test-model",
            tier=BackendTier.COMPLEX,
            base_url="http://test",
            env_key=None,
            capabilities=BackendCapabilities(
                max_context_tokens=131072,
                supports_tool_use=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_system_prompts=True,
            ),
            cost=BackendCostProfile(input_per_mtok=1.0, output_per_mtok=2.0),
            rate_limits=BackendRateLimits(rpm=60, rpd=14400, tpm=100000, daily_token_cap=1000000),
        )

        ibr_config = IntentClassificationConfig(
            enabled=True,
            confidence_threshold=0.0,  # Always pass confidence gate
            profile_confidence_threshold=0.0,
        )

        # Create mock spectrograph loader with profiles
        mock_loader = MagicMock()
        operator_profiles = {
            "test-model": ModelSpectrographProfile(
                model_id="test-model",
                version=1,
                updated_at="2026-01-01T00:00:00",
                task_scores={"generation": SpectrographScore(score=0.7, confidence=0.8, sample_count=20)},
                domain_scores={"code": SpectrographScore(score=0.8, confidence=0.8, sample_count=20)},
                qs_scores={"quality": SpectrographScore(score=0.9, confidence=0.8, sample_count=20)},
            )
        }
        mock_loader.profiles = operator_profiles

        # Create feedback store that returns learned profiles
        mock_feedback = MagicMock()
        learned_profiles = {
            "test-model": ModelSpectrographProfile(
                model_id="test-model",
                version=1,
                updated_at="2026-06-01T00:00:00",
                task_scores={"generation": SpectrographScore(score=0.9, confidence=0.9, sample_count=50)},
                domain_scores={"code": SpectrographScore(score=0.95, confidence=0.9, sample_count=50)},
                qs_scores={"quality": SpectrographScore(score=0.95, confidence=0.9, sample_count=50)},
            )
        }
        mock_feedback.get_learned_profiles.return_value = learned_profiles

        # Merged profiles should combine operator + feedback
        merged = {
            "test-model": ModelSpectrographProfile(
                model_id="test-model",
                version=1,
                updated_at="2026-06-01T00:00:00",
                task_scores={"generation": SpectrographScore(score=0.9, confidence=0.9, sample_count=50)},
                domain_scores={"code": SpectrographScore(score=0.95, confidence=0.9, sample_count=50)},
                qs_scores={"quality": SpectrographScore(score=0.95, confidence=0.9, sample_count=50)},
            )
        }
        mock_loader.get_merged_profiles.return_value = merged

        result = _build_ibr_result(
            intent, [candidate], ibr_config, mock_loader, feedback_store=mock_feedback
        )

        # Verify feedback store was queried
        mock_feedback.get_learned_profiles.assert_called_once()
        # Verify merged profiles were requested
        mock_loader.get_merged_profiles.assert_called_once_with(learned_profiles)
        # Result should be active
        assert result.ibr_active is True

    @pytest.mark.asyncio
    async def test_ibr_feedback_store_none(self):
        """None feedback_store degrades gracefully — no merge attempted."""
        from dragonlight_router.config.schema import IntentClassificationConfig
        from dragonlight_router.selection.ibr import _build_ibr_result

        intent = ClassifiedIntent(
            task_type="generation",
            domain="code",
            quality_speed="quality",
            confidence=0.9,
            latency_ms=100.0,
            from_cache=False,
        )

        from dragonlight_router.core.types import (
            BackendCapabilities,
            BackendConfig,
            BackendCostProfile,
            BackendRateLimits,
        )

        candidate = BackendConfig(
            name="test-model",
            provider="test",
            model="test-model",
            tier=BackendTier.COMPLEX,
            base_url="http://test",
            env_key=None,
            capabilities=BackendCapabilities(
                max_context_tokens=131072,
                supports_tool_use=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_system_prompts=True,
            ),
            cost=BackendCostProfile(input_per_mtok=1.0, output_per_mtok=2.0),
            rate_limits=BackendRateLimits(rpm=60, rpd=14400, tpm=100000, daily_token_cap=1000000),
        )

        ibr_config = IntentClassificationConfig(
            enabled=True,
            confidence_threshold=0.0,
            profile_confidence_threshold=0.0,
        )

        mock_loader = MagicMock()
        mock_loader.profiles = {
            "test-model": ModelSpectrographProfile(
                model_id="test-model",
                version=1,
                updated_at="2026-01-01T00:00:00",
                task_scores={"generation": SpectrographScore(score=0.7, confidence=0.8, sample_count=20)},
                domain_scores={"code": SpectrographScore(score=0.8, confidence=0.8, sample_count=20)},
                qs_scores={"quality": SpectrographScore(score=0.9, confidence=0.8, sample_count=20)},
            )
        }

        result = _build_ibr_result(
            intent, [candidate], ibr_config, mock_loader, feedback_store=None
        )

        # Should NOT attempt to merge
        mock_loader.get_merged_profiles.assert_not_called()
        # Should still produce active result
        assert result.ibr_active is True

    @pytest.mark.asyncio
    async def test_ibr_empty_learned_profiles_no_merge(self):
        """Empty learned profiles dict means no merge is needed."""
        from dragonlight_router.config.schema import IntentClassificationConfig
        from dragonlight_router.selection.ibr import _build_ibr_result

        intent = ClassifiedIntent(
            task_type="generation",
            domain="code",
            quality_speed="quality",
            confidence=0.9,
            latency_ms=100.0,
            from_cache=False,
        )

        from dragonlight_router.core.types import (
            BackendCapabilities,
            BackendConfig,
            BackendCostProfile,
            BackendRateLimits,
        )

        candidate = BackendConfig(
            name="test-model",
            provider="test",
            model="test-model",
            tier=BackendTier.COMPLEX,
            base_url="http://test",
            env_key=None,
            capabilities=BackendCapabilities(
                max_context_tokens=131072,
                supports_tool_use=True,
                supports_streaming=True,
                supports_json_mode=True,
                supports_system_prompts=True,
            ),
            cost=BackendCostProfile(input_per_mtok=1.0, output_per_mtok=2.0),
            rate_limits=BackendRateLimits(rpm=60, rpd=14400, tpm=100000, daily_token_cap=1000000),
        )

        ibr_config = IntentClassificationConfig(
            enabled=True,
            confidence_threshold=0.0,
            profile_confidence_threshold=0.0,
        )

        mock_loader = MagicMock()
        mock_loader.profiles = {
            "test-model": ModelSpectrographProfile(
                model_id="test-model",
                version=1,
                updated_at="2026-01-01T00:00:00",
                task_scores={"generation": SpectrographScore(score=0.7, confidence=0.8, sample_count=20)},
                domain_scores={"code": SpectrographScore(score=0.8, confidence=0.8, sample_count=20)},
                qs_scores={"quality": SpectrographScore(score=0.9, confidence=0.8, sample_count=20)},
            )
        }

        mock_feedback = MagicMock()
        mock_feedback.get_learned_profiles.return_value = {}  # Empty

        result = _build_ibr_result(
            intent, [candidate], ibr_config, mock_loader, feedback_store=mock_feedback
        )

        # Empty learned profiles -> no merge
        mock_loader.get_merged_profiles.assert_not_called()
        assert result.ibr_active is True


# ---------------------------------------------------------------------------
# Section 6: Cascade CBR Weight Resolution
# ---------------------------------------------------------------------------


class TestResolveCBRWeightsThreeTier:
    """_resolve_cbr_weights applies three-tier weight selection with context escalation."""

    def test_low_stakes_intent_uses_low_weights(self):
        from dragonlight_router.dispatch.cascade import DispatchContext, _resolve_cbr_weights

        order = _make_order(intent_category="test_generation")
        ctx = MagicMock(spec=DispatchContext)
        ctx.ibr_config = None

        weights = _resolve_cbr_weights(None, ctx, order)
        assert weights == _LOW_STAKES_WEIGHTS

    def test_mid_stakes_intent_uses_mid_weights(self):
        from dragonlight_router.dispatch.cascade import DispatchContext, _resolve_cbr_weights

        order = _make_order(intent_category="code_review")
        ctx = MagicMock(spec=DispatchContext)
        ctx.ibr_config = None

        weights = _resolve_cbr_weights(None, ctx, order)
        assert weights == _MID_STAKES_WEIGHTS

    def test_high_stakes_intent_uses_high_weights(self):
        from dragonlight_router.dispatch.cascade import DispatchContext, _resolve_cbr_weights

        order = _make_order(intent_category="architecture")
        ctx = MagicMock(spec=DispatchContext)
        ctx.ibr_config = None

        weights = _resolve_cbr_weights(None, ctx, order)
        assert weights == _HIGH_STAKES_WEIGHTS

    def test_context_escalation_overrides_low_to_high(self):
        """Low-stakes intent + very high context tokens -> capability_optimized weights."""
        from dragonlight_router.dispatch.cascade import DispatchContext, _resolve_cbr_weights

        order = _make_order(intent_category="test_generation", context_tokens=50000)
        ctx = MagicMock(spec=DispatchContext)
        ctx.ibr_config = None

        weights = _resolve_cbr_weights(None, ctx, order)
        # Context escalation to capability_optimized -> _HIGH_STAKES_WEIGHTS
        assert weights == _HIGH_STAKES_WEIGHTS

    def test_unknown_intent_uses_context_stakes(self):
        """Unknown intent with context signals should use context-derived weights."""
        from dragonlight_router.dispatch.cascade import DispatchContext, _resolve_cbr_weights

        order = _make_order(intent_category="unknown_thing", context_tokens=50000)
        ctx = MagicMock(spec=DispatchContext)
        ctx.ibr_config = None

        weights = _resolve_cbr_weights(None, ctx, order)
        # Unknown intent -> default weights, but context escalation -> capability_optimized
        assert weights == _HIGH_STAKES_WEIGHTS

    def test_no_order_falls_through_to_default(self):
        """No order -> falls through to IBR or default."""
        from dragonlight_router.dispatch.cascade import DispatchContext, _resolve_cbr_weights

        ctx = MagicMock(spec=DispatchContext)
        ctx.ibr_config = None

        weights = _resolve_cbr_weights(None, ctx, order=None)
        assert weights == ScoringWeightsConfig()


# ---------------------------------------------------------------------------
# Section 7: DispatchContext feedback_store field
# ---------------------------------------------------------------------------


class TestDispatchContextFeedbackStore:
    """DispatchContext carries feedback_store for IBR pipeline."""

    def test_default_feedback_store_is_none(self):
        from dragonlight_router.dispatch.cascade import DispatchContext

        ctx = DispatchContext(
            registry=MagicMock(),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
        )
        assert ctx.feedback_store is None

    def test_feedback_store_can_be_set(self):
        from dragonlight_router.dispatch.cascade import DispatchContext

        mock_store = MagicMock()
        ctx = DispatchContext(
            registry=MagicMock(),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
            feedback_store=mock_store,
        )
        assert ctx.feedback_store is mock_store
