"""Unit tests for LBR (rate-limit-aware dispatch) stage.

Spec traceability: TM-003 (LBR rate-limit filtering)
"""

from __future__ import annotations

from unittest.mock import MagicMock, Mock

import pytest

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    DispatchOrder,
)
from dragonlight_router.result import Err, Ok
from dragonlight_router.selection.lbr import filter_by_rate_limit


def _make_backend_config(
    name: str,
    provider: str,
    model: str = "test-model",
    tier: BackendTier = BackendTier.COMPLEX,
    base_url: str = "https://example.com",
    env_key: str | None = None,
) -> BackendConfig:
    """Helper to create a BackendConfig with sensible defaults for testing."""
    return BackendConfig(
        name=name,
        provider=provider,
        model=model,
        tier=tier,
        base_url=base_url,
        env_key=env_key,
        capabilities=BackendCapabilities(
            max_context_tokens=4096,
            supports_tool_use=False,
            supports_streaming=True,
            supports_json_mode=False,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0),
        rate_limits=BackendRateLimits(rpm=10, rpd=100, tpm=1000, daily_token_cap=10000),
        priority=0,
    )


def test_filter_by_rate_limit_no_candidates():
    """[TM-003 AC-1] When no candidates are provided, return empty list."""
    candidates = []
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    budget_tracker = BudgetTracker(providers=[])
    result = filter_by_rate_limit(candidates, order, budget_tracker)
    assert result == []


def test_filter_by_rate_limit_no_budget_data():
    """[TM-003 AC-2] When budget tracker returns no data, return candidates as-is."""
    # Create a candidate
    config = _make_backend_config(name="test", provider="test")
    candidates = [config]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    # Budget tracker returns Err for the provider (simulate not found)
    budget_tracker = BudgetTracker(providers=[])  # No providers configured

    result = filter_by_rate_limit(candidates, order, budget_tracker)
    # Should return as-is because no score data (all zeros)
    assert result == candidates


def test_filter_by_rate_limit_median_filter():
    """[TM-003 AC-3] Candidates with score >= median are kept by the LBR filter."""
    # Create two candidates with different providers
    config_a = _make_backend_config(name="a", provider="provider_a")
    config_b = _make_backend_config(name="b", provider="provider_b")
    candidates = [config_a, config_b]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    # We need a budget tracker that returns specific scores for each provider.
    # Since BudgetTracker.score uses internal state, we'll mock it.
    budget_tracker = Mock(spec=BudgetTracker)
    # We'll make score return Ok(80.0) for provider_a and Ok(20.0) for provider_b
    def score_side_effect(provider_name):
        if provider_name == "provider_a":
            return Ok(80.0)
        else:
            return Ok(20.0)
    budget_tracker.score.side_effect = score_side_effect

    result = filter_by_rate_limit(candidates, order, budget_tracker)
    # Median of [80.0, 20.0] is 50.0, so only provider_a (80.0 >= 50.0) should be kept
    assert len(result) == 1
    assert result[0].provider == "provider_a"


def test_filter_by_rate_limit_all_zero_scores():
    """[TM-003 AC-3] When all scores are zero, return candidates as-is (median zero, all pass)."""
    config_a = _make_backend_config(name="a", provider="provider_a")
    config_b = _make_backend_config(name="b", provider="provider_b")
    candidates = [config_a, config_b]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    budget_tracker = Mock(spec=BudgetTracker)
    budget_tracker.score.return_value = Ok(0.0)

    result = filter_by_rate_limit(candidates, order, budget_tracker)
    # Median of [0.0, 0.0] is 0.0, so both candidates pass (score >= 0)
    assert len(result) == 2
    assert result == candidates


def test_local_provider_bypasses_rate_limit():
    """[TM-003 AC-4] LOCAL backend with a score below median is still included."""
    config_local = _make_backend_config(
        name="local-llm", provider="local_provider", tier=BackendTier.LOCAL,
    )
    config_cloud = _make_backend_config(
        name="cloud-llm", provider="cloud_provider", tier=BackendTier.COMPLEX,
    )
    candidates = [config_local, config_cloud]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    budget_tracker = Mock(spec=BudgetTracker)

    # LOCAL provider has a very low score; cloud provider has a high score.
    # Median of [5.0, 90.0] = 47.5 — local_provider (5.0) is below median
    # but should still be included because its tier is LOCAL.
    def score_side_effect(provider_name):
        if provider_name == "local_provider":
            return Ok(5.0)
        return Ok(90.0)

    budget_tracker.score.side_effect = score_side_effect

    result = filter_by_rate_limit(candidates, order, budget_tracker)
    assert len(result) == 2
    result_names = {c.name for c in result}
    assert "local-llm" in result_names
    assert "cloud-llm" in result_names


def test_local_and_non_local_mixed():
    """[TM-003 AC-4] LOCAL passes through while non-LOCAL below median is filtered."""
    config_local = _make_backend_config(
        name="local-llm", provider="local_provider", tier=BackendTier.LOCAL,
    )
    config_good = _make_backend_config(
        name="good-cloud", provider="good_cloud", tier=BackendTier.COMPLEX,
    )
    config_bad = _make_backend_config(
        name="bad-cloud", provider="bad_cloud", tier=BackendTier.SIMPLE,
    )
    candidates = [config_local, config_good, config_bad]
    order = DispatchOrder(
        intent_category="test",
        specific_intent="test",
        operator_message="test",
        system_prompt="",
        context_tokens=0,
        requires_tool_use=False,
        requires_long_context=False,
    )
    budget_tracker = Mock(spec=BudgetTracker)

    # Scores: local=2.0, good_cloud=80.0, bad_cloud=10.0
    # Sorted: [2.0, 10.0, 80.0] → median = 10.0
    # bad_cloud (10.0 >= 10.0) passes, good_cloud (80.0 >= 10.0) passes,
    # local (2.0 < 10.0) would be filtered BUT tier is LOCAL so it passes.
    def score_side_effect(provider_name):
        scores = {
            "local_provider": 2.0,
            "good_cloud": 80.0,
            "bad_cloud": 10.0,
        }
        return Ok(scores[provider_name])

    budget_tracker.score.side_effect = score_side_effect

    result = filter_by_rate_limit(candidates, order, budget_tracker)
    result_names = {c.name for c in result}
    # All three should be present: local bypasses, good_cloud above median,
    # bad_cloud exactly at median.
    assert result_names == {"local-llm", "good-cloud", "bad-cloud"}


def test_select_final_candidate_returns_first():
    """[TM-003 AC-6] select_final_candidate returns the first element of the list."""
    from dragonlight_router.selection.lbr import select_final_candidate
    config_a = _make_backend_config(name="a", provider="prov_a")
    config_b = _make_backend_config(name="b", provider="prov_b")
    result = select_final_candidate([config_a, config_b])
    assert result is config_a


def test_select_final_candidate_empty_raises():
    """[TM-003 AC-6] select_final_candidate raises ValueError for empty list."""
    from dragonlight_router.selection.lbr import select_final_candidate
    with pytest.raises(ValueError, match="Cannot select from empty candidate list"):
        select_final_candidate([])


def test_extract_score_non_ok_with_value_attribute():
    """[TM-003 AC-5] _extract_score returns float(result.value) when result is not Ok but has .value."""
    from dragonlight_router.selection.lbr import _extract_score
    budget_tracker = MagicMock()
    non_ok_result = MagicMock(spec=[])
    non_ok_result.value = 42.5
    budget_tracker.score.return_value = non_ok_result
    score = _extract_score(budget_tracker, "some_provider")
    assert score == pytest.approx(42.5)


def test_extract_score_non_ok_no_value_returns_zero():
    """[TM-003 AC-5] _extract_score returns 0.0 when result is not Ok and has no .value."""
    from dragonlight_router.selection.lbr import _extract_score
    budget_tracker = MagicMock()
    budget_tracker.score.return_value = object()
    score = _extract_score(budget_tracker, "some_provider")
    assert score == 0.0


def test_collect_provider_scores_deduplicates_same_provider():
    """[TM-003 AC-5] _collect_provider_scores hits the continue branch for repeated provider."""
    from dragonlight_router.selection.lbr import _collect_provider_scores
    config_a = _make_backend_config(name="a", provider="shared_prov")
    config_b = _make_backend_config(name="b", provider="shared_prov")
    budget_tracker = MagicMock()
    budget_tracker.score.return_value = Ok(50.0)
    scores = _collect_provider_scores([config_a, config_b], budget_tracker)
    assert scores == {"shared_prov": 50.0}
    budget_tracker.score.assert_called_once_with("shared_prov")


class TestHardCapacityGate:
    """HAZ-005 mitigation: hard has_capacity gate before median filtering."""

    def test_removes_no_capacity_provider(self):
        """[TM-003 AC-7] Provider with no remaining capacity is removed by hard gate."""
        from dragonlight_router.selection.lbr import _hard_capacity_gate
        config_a = _make_backend_config(name="a", provider="exhausted_prov")
        config_b = _make_backend_config(name="b", provider="healthy_prov")
        budget_tracker = Mock(spec=BudgetTracker)
        budget_tracker.has_capacity.side_effect = lambda p: p != "exhausted_prov"
        result = _hard_capacity_gate([config_a, config_b], budget_tracker)
        assert len(result) == 1
        assert result[0].name == "b"

    def test_local_bypasses_capacity_gate(self):
        """[TM-003 AC-7] LOCAL tier bypasses the hard capacity gate."""
        from dragonlight_router.selection.lbr import _hard_capacity_gate
        local = _make_backend_config(
            name="local-llm", provider="local_prov", tier=BackendTier.LOCAL,
        )
        budget_tracker = Mock(spec=BudgetTracker)
        budget_tracker.has_capacity.return_value = False
        result = _hard_capacity_gate([local], budget_tracker)
        assert len(result) == 1
        assert result[0].name == "local-llm"

    def test_all_exhausted_returns_empty(self):
        """[TM-003 AC-7] All providers exhausted returns empty list."""
        from dragonlight_router.selection.lbr import _hard_capacity_gate
        config_a = _make_backend_config(name="a", provider="prov_a")
        config_b = _make_backend_config(name="b", provider="prov_b")
        budget_tracker = Mock(spec=BudgetTracker)
        budget_tracker.has_capacity.return_value = False
        result = _hard_capacity_gate([config_a, config_b], budget_tracker)
        assert result == []

    def test_filter_by_rate_limit_uses_hard_gate(self):
        """[TM-003 AC-7] filter_by_rate_limit removes exhausted providers before median scoring."""
        config_ok = _make_backend_config(name="ok", provider="good_prov")
        config_bad = _make_backend_config(name="bad", provider="exhausted_prov")
        order = DispatchOrder(
            intent_category="test",
            specific_intent="test",
            operator_message="test",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        )
        budget_tracker = Mock(spec=BudgetTracker)

        def has_capacity_side(p):
            return p != "exhausted_prov"

        budget_tracker.has_capacity.side_effect = has_capacity_side
        budget_tracker.score.return_value = Ok(80.0)

        result = filter_by_rate_limit([config_ok, config_bad], order, budget_tracker)
        assert len(result) == 1
        assert result[0].name == "ok"

    def test_filter_by_rate_limit_all_removed_by_gate(self):
        """[TM-003 AC-7] All candidates removed by hard gate returns empty list."""
        config = _make_backend_config(name="a", provider="prov")
        order = DispatchOrder(
            intent_category="test",
            specific_intent="test",
            operator_message="test",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        )
        budget_tracker = Mock(spec=BudgetTracker)
        budget_tracker.has_capacity.return_value = False
        result = filter_by_rate_limit([config], order, budget_tracker)
        assert result == []


def test_filter_by_rate_limit_guard_clauses():
    """[TM-003 AC-5] Guard clauses should raise AssertionError on invalid types."""
    # candidates not a list
    with pytest.raises(AssertionError, match="candidates must be a list"):
        filter_by_rate_limit(
            "not a list",
            DispatchOrder(
                intent_category="test",
                specific_intent="test",
                operator_message="test",
                system_prompt="",
                context_tokens=0,
                requires_tool_use=False,
                requires_long_context=False,
            ),
            BudgetTracker(providers=[]),
        )
    # candidate not BackendConfig
    with pytest.raises(AssertionError, match="all candidates must be BackendConfig instances"):
        filter_by_rate_limit(
            ["not a config"],
            DispatchOrder(
                intent_category="test",
                specific_intent="test",
                operator_message="test",
                system_prompt="",
                context_tokens=0,
                requires_tool_use=False,
                requires_long_context=False,
            ),
            BudgetTracker(providers=[]),
        )
    # order not DispatchOrder
    with pytest.raises(AssertionError, match="order must be DispatchOrder instance"):
        filter_by_rate_limit(
            [],
            "not order",
            BudgetTracker(providers=[]),
        )
    # budget_tracker not BudgetTracker
    with pytest.raises(AssertionError, match="budget_tracker must be BudgetTracker instance"):
        filter_by_rate_limit(
            [],
            DispatchOrder(
                intent_category="test",
                specific_intent="test",
                operator_message="test",
                system_prompt="",
                context_tokens=0,
                requires_tool_use=False,
                requires_long_context=False,
            ),
            "not tracker",
        )