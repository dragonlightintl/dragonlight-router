"""Unit tests for dispatch/cascade.py — uncovered branches.

Spec traceability: TM-004 (Cascade dispatch)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import (
    BackendStatus,
    BackendTier,
    DispatchOrder,
    ScoredCandidate,
)
from dragonlight_router.dispatch.cascade import (
    DispatchContext,
    _apply_degraded_penalty,
    _build_messages,
    _filter_by_trust_floor,
    _run_cascade,
    _run_cbr_stage,
    _run_lbr_stage,
    _try_streaming_dispatch,
    dispatch_stream,
    route,
)
from dragonlight_router.result import Err, Ok

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_order(**kwargs) -> DispatchOrder:
    defaults = {
        "intent_category": "test",
        "specific_intent": "test",
        "operator_message": "hello",
        "system_prompt": "",
        "context_tokens": 0,
        "requires_tool_use": False,
        "requires_long_context": False,
    }
    defaults.update(kwargs)
    return DispatchOrder(**defaults)


def _make_ctx(registry=None, budget_tracker=None, health_tracker=None, config=None):
    registry = registry or MagicMock(spec=BackendRegistry)
    budget_tracker = budget_tracker or MagicMock()
    health_tracker = health_tracker or MagicMock()
    config = config if config is not None else {}
    return DispatchContext(
        registry=registry,
        budget_tracker=budget_tracker,
        health_tracker=health_tracker,
        config=config,
    )


# ---------------------------------------------------------------------------
# Line 133 — cost_governor_active → cost_adjusted_weights called
# ---------------------------------------------------------------------------

class TestRunCbrStageCostGovernor:
    def test_cost_governor_active_calls_cost_adjusted_weights(self, make_backend_config):
        """[TM-004 AC-1] cost_governor_active True → cost_adjusted_weights used in CBR scoring."""
        backend = make_backend_config(name="b1", provider="prov", input_cost=1.0, output_cost=2.0)
        order = _make_order()

        budget_tracker = MagicMock()
        budget_tracker.daily_spend_usd.return_value = 200.0
        budget_tracker.monthly_spend_usd.return_value = 2000.0
        budget_tracker.score.return_value = Ok(90.0)

        health_tracker = MagicMock()
        health_tracker.score.return_value = Ok(80.0)

        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (MagicMock(), None)

        config = {
            "cost_down_threshold_daily": 100.0,
            "cost_down_threshold_monthly": 1000.0,
        }
        ctx = _make_ctx(registry=registry, budget_tracker=budget_tracker,
                        health_tracker=health_tracker, config=config)

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with (
            patch(f"{cascade_mod}.filter_by_cost") as mock_filter,
            patch(f"{cascade_mod}.cost_governor_active", return_value=True) as mock_gov,
            patch(f"{cascade_mod}.cost_adjusted_weights") as mock_adj,
        ):
            mock_adj.return_value = MagicMock(
                cost=0.70, latency=0.10, priority=0.10, queue=0.05, health=0.05
            )
            mock_filter.return_value = [backend]
            with patch("dragonlight_router.dispatch.cascade._score_and_rank_candidates",
                       return_value=[backend]):
                result = _run_cbr_stage(order, [backend], ctx)

        mock_gov.assert_called_once()
        mock_adj.assert_called_once()
        assert result.is_ok()


# ---------------------------------------------------------------------------
# Lines 171-178 — _apply_degraded_penalty when backend IS DEGRADED
# ---------------------------------------------------------------------------

class TestApplyDegradedPenalty:
    def test_degraded_backend_score_halved(self, make_backend_config):
        """[TM-004 AC-2] DEGRADED backend score is multiplied by 0.5."""
        backend = make_backend_config(name="b1", provider="prov")

        state = MagicMock()
        state.status = BackendStatus.DEGRADED

        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (MagicMock(), state)

        original_score = 0.8
        result = _apply_degraded_penalty(original_score, backend, registry)

        assert result == pytest.approx(0.4)

    def test_available_backend_score_unchanged(self, make_backend_config):
        """[TM-004 AC-2] Non-DEGRADED backend score is not modified."""
        backend = make_backend_config(name="b1", provider="prov")

        state = MagicMock()
        state.status = BackendStatus.AVAILABLE

        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (MagicMock(), state)

        original_score = 0.75
        result = _apply_degraded_penalty(original_score, backend, registry)

        assert result == pytest.approx(0.75)

    def test_none_state_score_unchanged(self, make_backend_config):
        """[TM-004 AC-2] None state returns original score unchanged."""
        backend = make_backend_config(name="b1", provider="prov")

        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (None, None)

        original_score = 0.6
        result = _apply_degraded_penalty(original_score, backend, registry)

        assert result == pytest.approx(0.6)


# ---------------------------------------------------------------------------
# Line 195 — _run_lbr_stage empty result → Err(LBRNoCapacityError)
# ---------------------------------------------------------------------------

class TestRunLbrStageEmpty:
    def test_empty_lbr_result_returns_err(self, make_backend_config):
        """[TM-004 AC-3] LBR returning no candidates → Err(LBRNoCapacityError)."""
        backend = make_backend_config(name="b1", provider="prov")
        scored = ScoredCandidate(config=backend, score=0.5)
        order = _make_order()
        ctx = _make_ctx()

        with patch("dragonlight_router.dispatch.cascade.filter_by_rate_limit", return_value=[]):
            result = _run_lbr_stage(order, [scored], ctx)

        assert result.is_err()
        from dragonlight_router.core.errors import LBRNoCapacityError
        assert isinstance(result.error, LBRNoCapacityError)


# ---------------------------------------------------------------------------
# Lines 248-268 — route() full cascade path including select_final_candidate
# ---------------------------------------------------------------------------

class TestRoute:
    def test_route_returns_ok_backend_config(self, make_backend_config):
        """[TM-004 AC-4] route() returns Ok(BackendConfig) on successful cascade."""
        backend = make_backend_config(name="b1", provider="prov")
        order = _make_order()
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = MagicMock()
        config = {}

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with (
            patch(f"{cascade_mod}._run_cascade", return_value=Ok([backend])),
            patch(f"{cascade_mod}.select_final_candidate", return_value=backend),
        ):
            result = route(order, registry, budget_tracker, health_tracker, config)

        assert result.is_ok()
        assert result.value is backend

    def test_route_propagates_cascade_err(self, make_backend_config):
        """[TM-004 AC-4] route() propagates Err when cascade fails."""
        order = _make_order()
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = MagicMock()
        config = {}

        from dragonlight_router.core.errors import BudgetExceededError
        err = BudgetExceededError("no budget")

        with patch("dragonlight_router.dispatch.cascade._run_cascade", return_value=Err(err)):
            result = route(order, registry, budget_tracker, health_tracker, config)

        assert result.is_err()
        assert result.error is err

    def test_route_select_final_candidate_called_with_candidates(self, make_backend_config):
        """[TM-004 AC-4] route() calls select_final_candidate with cascade results."""
        b1 = make_backend_config(name="b1", provider="prov1")
        b2 = make_backend_config(name="b2", provider="prov2")
        order = _make_order()
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = MagicMock()
        config = {}

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with (
            patch(f"{cascade_mod}._run_cascade", return_value=Ok([b1, b2])),
            patch(f"{cascade_mod}.select_final_candidate", return_value=b1) as mock_sel,
        ):
            route(order, registry, budget_tracker, health_tracker, config)

        mock_sel.assert_called_once_with([b1, b2])


# ---------------------------------------------------------------------------
# Lines 295-296 — _build_messages when system_content is a plain string
# ---------------------------------------------------------------------------

class TestBuildMessages:
    def test_system_content_is_string_appended(self):
        """[TM-004 AC-5] String system_content is added as system message."""
        filtered_context = {"system": "You are helpful.", "task": "Do the thing."}
        messages = _build_messages(filtered_context, "fallback")

        assert messages[0] == {"role": "system", "content": "You are helpful."}
        assert messages[1] == {"role": "user", "content": "Do the thing."}

    def test_empty_string_system_content_not_appended(self):
        """[TM-004 AC-5] Empty string system_content produces no system message."""
        filtered_context = {"system": "", "task": "Do something."}
        messages = _build_messages(filtered_context, "fallback")

        roles = [m["role"] for m in messages]
        assert "system" not in roles

    def test_dict_system_content_with_prompt(self):
        """[TM-004 AC-5] Dict system_content with prompt key appended correctly."""
        filtered_context = {"system": {"prompt": "You are an assistant."}, "task": "Help me."}
        messages = _build_messages(filtered_context, "fallback")

        assert messages[0] == {"role": "system", "content": "You are an assistant."}
        assert messages[1] == {"role": "user", "content": "Help me."}

    def test_missing_task_uses_fallback(self):
        """[TM-004 AC-5] Missing task key falls back to fallback_message."""
        filtered_context = {}
        messages = _build_messages(filtered_context, "fallback text")

        user_msg = next(m for m in messages if m["role"] == "user")
        assert user_msg["content"] == "fallback text"


# ---------------------------------------------------------------------------
# HAZ-001 — _filter_by_trust_floor context trust tier enforcement
# ---------------------------------------------------------------------------

class TestFilterByTrustFloor:
    """HAZ-001 mitigation: context_trust_tier enforcement in cascade."""

    def test_none_trust_tier_passes_all(self, make_backend_config):
        """[TM-004 AC-6] None context_trust_tier passes all candidates through."""
        simple = make_backend_config(name="s1", tier=BackendTier.SIMPLE)
        complex_ = make_backend_config(name="c1", tier=BackendTier.COMPLEX)
        result = _filter_by_trust_floor([simple, complex_], None)
        assert len(result) == 2

    def test_trusted_floor_filters_simple(self, make_backend_config):
        """[TM-004 AC-6] 'trusted' floor removes SIMPLE/MODERATE backends (semi_trusted)."""
        simple = make_backend_config(name="s1", tier=BackendTier.SIMPLE)
        complex_ = make_backend_config(name="c1", tier=BackendTier.COMPLEX)
        result = _filter_by_trust_floor([simple, complex_], "trusted")
        assert len(result) == 1
        assert result[0].name == "c1"

    def test_trusted_floor_keeps_local(self, make_backend_config):
        """[TM-004 AC-6] 'trusted' floor keeps LOCAL backends (LOCAL rank >= trusted)."""
        local = make_backend_config(name="l1", tier=BackendTier.LOCAL)
        complex_ = make_backend_config(name="c1", tier=BackendTier.COMPLEX)
        result = _filter_by_trust_floor([local, complex_], "trusted")
        assert len(result) == 2

    def test_semi_trusted_floor_filters_nothing(self, make_backend_config):
        """[TM-004 AC-6] 'semi_trusted' floor keeps SIMPLE, MODERATE, COMPLEX, LOCAL."""
        simple = make_backend_config(name="s1", tier=BackendTier.SIMPLE)
        moderate = make_backend_config(name="m1", tier=BackendTier.MODERATE)
        complex_ = make_backend_config(name="c1", tier=BackendTier.COMPLEX)
        local = make_backend_config(name="l1", tier=BackendTier.LOCAL)
        result = _filter_by_trust_floor(
            [simple, moderate, complex_, local], "semi_trusted",
        )
        assert len(result) == 4

    def test_local_floor_keeps_only_local(self, make_backend_config):
        """[TM-004 AC-6] 'local' floor keeps only LOCAL backends."""
        simple = make_backend_config(name="s1", tier=BackendTier.SIMPLE)
        local = make_backend_config(name="l1", tier=BackendTier.LOCAL)
        result = _filter_by_trust_floor([simple, local], "local")
        assert len(result) == 1
        assert result[0].name == "l1"

    def test_unknown_trust_tier_passes_all(self, make_backend_config):
        """[TM-004 AC-6] Unknown context_trust_tier string passes all candidates."""
        simple = make_backend_config(name="s1", tier=BackendTier.SIMPLE)
        result = _filter_by_trust_floor([simple], "bogus_tier")
        assert len(result) == 1

    def test_untrusted_floor_passes_all(self, make_backend_config):
        """[TM-004 AC-6] 'untrusted' floor passes all backends (lowest rank)."""
        simple = make_backend_config(name="s1", tier=BackendTier.SIMPLE)
        result = _filter_by_trust_floor([simple], "untrusted")
        assert len(result) == 1

    def test_case_insensitive(self, make_backend_config):
        """[TM-004 AC-6] Trust tier string is case-insensitive."""
        complex_ = make_backend_config(name="c1", tier=BackendTier.COMPLEX)
        simple = make_backend_config(name="s1", tier=BackendTier.SIMPLE)
        result = _filter_by_trust_floor([complex_, simple], "TRUSTED")
        assert len(result) == 1
        assert result[0].name == "c1"

    def test_empty_candidates_returns_empty(self):
        """[TM-004 AC-6] Empty candidate list returns empty."""
        result = _filter_by_trust_floor([], "trusted")
        assert result == []

    def test_cascade_returns_err_when_trust_floor_removes_all(self, make_backend_config):
        """[TM-004 AC-6] _run_cascade returns Err when trust floor filters all candidates."""
        simple = make_backend_config(name="s1", tier=BackendTier.SIMPLE)
        order = _make_order(context_trust_tier="local")
        ctx = _make_ctx()

        with patch(
            "dragonlight_router.dispatch.cascade._run_mbr_stage",
            return_value=Ok([simple]),
        ):
            result = _run_cascade(order, ctx)

        assert result.is_err()
        from dragonlight_router.selection.mbr import MBRNoCandidatesError
        assert isinstance(result.error, MBRNoCandidatesError)
        assert "context_trust_tier" in str(result.error)


# ---------------------------------------------------------------------------
# Streaming dispatch — _try_streaming_dispatch
# ---------------------------------------------------------------------------

def _make_mock_adapter(chunks: list[str]):
    """Create a mock adapter whose generate() yields the given chunks."""
    adapter = MagicMock()
    adapter.status = BackendStatus.AVAILABLE  # HAZ-014: fresh adapter starts AVAILABLE

    async def _gen(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    adapter.generate = _gen
    adapter.record_usage = MagicMock()
    return adapter


def _make_failing_adapter(exc: Exception):
    """Create a mock adapter whose generate() raises an exception."""
    adapter = MagicMock()
    adapter.status = BackendStatus.AVAILABLE  # HAZ-014: fresh adapter starts AVAILABLE

    async def _gen(*args, **kwargs):
        raise exc
        yield  # noqa: RET503 — unreachable, makes this an async generator

    adapter.generate = _gen
    adapter.record_usage = MagicMock()
    return adapter


class TestTryStreamingDispatch:
    @pytest.mark.asyncio
    async def test_streams_token_chunks_and_metadata(self, make_backend_config):
        """[TM-004 AC-1] _try_streaming_dispatch yields token chunks then metadata."""
        backend = make_backend_config(name="b1", provider="prov", tier=BackendTier.COMPLEX)
        order = _make_order()
        ctx = _make_ctx()
        base_context = {"task": "hello"}
        adapter = _make_mock_adapter(["Hello", " world"])

        adapter_path = "dragonlight_router.dispatch.cascade._adapters_mod.create_adapter"
        with patch(adapter_path, return_value=adapter):
            chunks = []
            async for chunk in _try_streaming_dispatch(backend, base_context, order, ctx, []):
                chunks.append(chunk)

        # Should have 2 token chunks + 1 metadata chunk
        assert len(chunks) == 3
        assert chunks[0].event_type == "token"
        assert chunks[0].content == "Hello"
        assert chunks[1].event_type == "token"
        assert chunks[1].content == " world"
        assert chunks[2].event_type == "metadata"
        assert chunks[2].backend_used == "b1"
        assert chunks[2].tokens_in >= 0
        assert chunks[2].tokens_out >= 0
        assert chunks[2].latency_ms > 0
        assert chunks[2].was_fallback is False

    @pytest.mark.asyncio
    async def test_metadata_shows_fallback_when_chain_nonempty(self, make_backend_config):
        """[TM-004 AC-2] _try_streaming_dispatch sets was_fallback=True
        when fallback_chain is non-empty."""
        backend = make_backend_config(name="b2", provider="prov")
        order = _make_order()
        ctx = _make_ctx()
        adapter = _make_mock_adapter(["ok"])

        adapter_path = "dragonlight_router.dispatch.cascade._adapters_mod.create_adapter"
        with patch(adapter_path, return_value=adapter):
            chunks = []
            async for chunk in _try_streaming_dispatch(backend, {"task": "hi"}, order, ctx, ["b1"]):
                chunks.append(chunk)

        metadata = [c for c in chunks if c.event_type == "metadata"]
        assert len(metadata) == 1
        assert metadata[0].was_fallback is True
        assert metadata[0].fallback_chain == ["b1"]

    @pytest.mark.asyncio
    async def test_records_health_and_budget(self, make_backend_config):
        """[TM-004 AC-5] _try_streaming_dispatch records success in health and budget trackers."""
        backend = make_backend_config(name="b1", provider="prov", model="test-model")
        order = _make_order()
        health_tracker = MagicMock()
        budget_tracker = MagicMock()
        ctx = _make_ctx(health_tracker=health_tracker, budget_tracker=budget_tracker)
        adapter = _make_mock_adapter(["response"])

        adapter_path = "dragonlight_router.dispatch.cascade._adapters_mod.create_adapter"
        with patch(adapter_path, return_value=adapter):
            async for _ in _try_streaming_dispatch(backend, {"task": "hi"}, order, ctx, []):
                pass

        health_tracker.record_success.assert_called_once()
        budget_tracker.record_request.assert_called_once()


# ---------------------------------------------------------------------------
# Streaming dispatch — dispatch_stream
# ---------------------------------------------------------------------------

class TestDispatchStream:
    @pytest.mark.asyncio
    async def test_streams_tokens_on_success(self, make_backend_config):
        """[TM-004 AC-1] dispatch_stream yields token and metadata chunks on success."""
        backend = make_backend_config(name="b1", provider="prov")
        scored = ScoredCandidate(config=backend, score=0.8)
        order = _make_order()
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = MagicMock()
        adapter = _make_mock_adapter(["Hi", " there"])

        cascade_mod = "dragonlight_router.dispatch.cascade"
        adapter_path = f"{cascade_mod}._adapters_mod.create_adapter"
        with (
            patch(f"{cascade_mod}._run_cascade", return_value=Ok([scored])),
            patch(adapter_path, return_value=adapter),
        ):
            chunks = []
            async for chunk in dispatch_stream(
                order, registry, budget_tracker, health_tracker, {},
            ):
                chunks.append(chunk)

        token_chunks = [c for c in chunks if c.event_type == "token"]
        metadata_chunks = [c for c in chunks if c.event_type == "metadata"]
        assert len(token_chunks) == 2
        assert len(metadata_chunks) == 1
        assert token_chunks[0].content == "Hi"
        assert token_chunks[1].content == " there"

    @pytest.mark.asyncio
    async def test_yields_error_on_cascade_failure(self, make_backend_config):
        """[TM-004 AC-6] dispatch_stream yields error chunk when cascade fails."""
        order = _make_order()
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = MagicMock()

        from dragonlight_router.core.errors import BudgetExceededError
        err = BudgetExceededError("no budget")

        with patch("dragonlight_router.dispatch.cascade._run_cascade", return_value=Err(err)):
            chunks = []
            async for chunk in dispatch_stream(order, registry, budget_tracker, health_tracker, {}):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].event_type == "error"
        assert "no budget" in chunks[0].error_message

    @pytest.mark.asyncio
    async def test_fallback_on_adapter_failure(self, make_backend_config):
        """[TM-004 AC-2] dispatch_stream falls back to next candidate on adapter failure."""
        b1 = make_backend_config(name="b1", provider="prov1")
        b2 = make_backend_config(name="b2", provider="prov2")
        sc1 = ScoredCandidate(config=b1, score=0.9)
        sc2 = ScoredCandidate(config=b2, score=0.7)
        order = _make_order()
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = MagicMock()

        failing_adapter = _make_failing_adapter(RuntimeError("timeout"))
        success_adapter = _make_mock_adapter(["fallback response"])

        call_count = 0

        def _create_adapter(config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return failing_adapter
            return success_adapter

        cascade_mod = "dragonlight_router.dispatch.cascade"
        adapter_path = f"{cascade_mod}._adapters_mod.create_adapter"
        with (
            patch(f"{cascade_mod}._run_cascade", return_value=Ok([sc1, sc2])),
            patch(adapter_path, side_effect=_create_adapter),
        ):
            chunks = []
            async for chunk in dispatch_stream(order, registry, budget_tracker, health_tracker, {}):
                chunks.append(chunk)

        token_chunks = [c for c in chunks if c.event_type == "token"]
        metadata_chunks = [c for c in chunks if c.event_type == "metadata"]
        assert len(token_chunks) == 1
        assert token_chunks[0].content == "fallback response"
        assert len(metadata_chunks) == 1
        assert metadata_chunks[0].was_fallback is True
        assert metadata_chunks[0].fallback_chain == ["b1"]

    @pytest.mark.asyncio
    async def test_all_backends_exhausted_yields_error(self, make_backend_config):
        """[TM-004 AC-6] dispatch_stream yields error when all backends fail."""
        b1 = make_backend_config(name="b1", provider="prov1")
        b2 = make_backend_config(name="b2", provider="prov2")
        sc1 = ScoredCandidate(config=b1, score=0.9)
        sc2 = ScoredCandidate(config=b2, score=0.7)
        order = _make_order()
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = MagicMock()

        failing_adapter = _make_failing_adapter(RuntimeError("boom"))

        cascade_mod = "dragonlight_router.dispatch.cascade"
        adapter_path = f"{cascade_mod}._adapters_mod.create_adapter"
        with (
            patch(f"{cascade_mod}._run_cascade", return_value=Ok([sc1, sc2])),
            patch(adapter_path, return_value=failing_adapter),
        ):
            chunks = []
            async for chunk in dispatch_stream(
                order, registry, budget_tracker, health_tracker, {},
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].event_type == "error"
        assert "2 backends exhausted" in chunks[0].error_message
        assert "b1" in chunks[0].error_message
        assert "b2" in chunks[0].error_message
