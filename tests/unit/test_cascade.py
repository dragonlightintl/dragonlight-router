"""Unit tests for dispatch/cascade.py — uncovered branches.

Spec traceability: TM-004 (Cascade dispatch)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

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

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_health_tracker_mock(**kwargs) -> MagicMock:
    """Create a MagicMock health tracker with is_retired defaulting to False."""
    ht = MagicMock(**kwargs)
    ht.is_retired.return_value = False
    return ht


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
    if health_tracker is None:
        health_tracker = _make_health_tracker_mock()
        # Default: no models are retired so cascade retirement filter passes all.
        health_tracker.is_retired.return_value = False
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

        health_tracker = _make_health_tracker_mock()
        health_tracker.score.return_value = Ok(80.0)

        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (MagicMock(), None)

        config = {
            "cost_down_threshold_daily": 100.0,
            "cost_down_threshold_monthly": 1000.0,
        }
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
            config=config,
        )

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
            with patch(
                "dragonlight_router.dispatch.cascade._score_and_rank_candidates",
                return_value=[backend],
            ):
                result = _run_cbr_stage(order, [backend], ctx)

        # TS-003: Primary assertion is output behavior, not just mock wiring.
        assert result.is_ok()
        assert result.value == [backend], "CBR stage must return the scored candidate list"
        # Mock assertions verify wiring at the cost-governor integration boundary.
        mock_gov.assert_called_once()
        mock_adj.assert_called_once()


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
    @pytest.mark.asyncio
    async def test_route_returns_ok_backend_config(self, make_backend_config):
        """[TM-004 AC-4] route() returns Ok(BackendConfig) on successful cascade."""
        backend = make_backend_config(name="b1", provider="prov")
        order = _make_order()
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = _make_health_tracker_mock()
        config = {}

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with (
            patch(
                f"{cascade_mod}._run_cascade",
                new_callable=AsyncMock,
                return_value=Ok([backend]),
            ),
            patch(f"{cascade_mod}.select_final_candidate", return_value=backend),
        ):
            result = await route(order, registry, budget_tracker, health_tracker, config)

        assert result.is_ok()
        assert result.value is backend

    @pytest.mark.asyncio
    async def test_route_propagates_cascade_err(self, make_backend_config):
        """[TM-004 AC-4] route() propagates Err when cascade fails."""
        order = _make_order()
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = _make_health_tracker_mock()
        config = {}

        from dragonlight_router.core.errors import BudgetExceededError

        err = BudgetExceededError("no budget")

        with patch(
            "dragonlight_router.dispatch.cascade._run_cascade",
            new_callable=AsyncMock,
            return_value=Err(err),
        ):
            result = await route(order, registry, budget_tracker, health_tracker, config)

        assert result.is_err()
        assert result.error is err

    @pytest.mark.asyncio
    async def test_route_select_final_candidate_called_with_candidates(self, make_backend_config):
        """[TM-004 AC-4] route() calls select_final_candidate with cascade results."""
        b1 = make_backend_config(name="b1", provider="prov1")
        b2 = make_backend_config(name="b2", provider="prov2")
        order = _make_order()
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = _make_health_tracker_mock()
        config = {}

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with (
            patch(f"{cascade_mod}._run_cascade", new_callable=AsyncMock, return_value=Ok([b1, b2])),
            patch(f"{cascade_mod}.select_final_candidate", return_value=b1) as mock_sel,
        ):
            result = await route(order, registry, budget_tracker, health_tracker, config)

        # TS-003: Assert output behavior, not just mock wiring.
        assert result.is_ok()
        assert result.value is b1, "route must return the selected candidate"
        # Mock assertion verifies the correct candidates were passed to selection.
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
            [simple, moderate, complex_, local],
            "semi_trusted",
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

    @pytest.mark.asyncio
    async def test_cascade_returns_err_when_trust_floor_removes_all(self, make_backend_config):
        """[TM-004 AC-6] _run_cascade returns Err when trust floor filters all candidates."""
        simple = make_backend_config(name="s1", tier=BackendTier.SIMPLE)
        order = _make_order(context_trust_tier="local")
        ctx = _make_ctx()

        with patch(
            "dragonlight_router.dispatch.cascade._run_mbr_stage",
            return_value=Ok([simple]),
        ):
            result = await _run_cascade(order, ctx)

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
        health_tracker = _make_health_tracker_mock()
        budget_tracker = MagicMock()
        ctx = _make_ctx(health_tracker=health_tracker, budget_tracker=budget_tracker)
        adapter = _make_mock_adapter(["response"])

        adapter_path = "dragonlight_router.dispatch.cascade._adapters_mod.create_adapter"
        with patch(adapter_path, return_value=adapter):
            chunks = []
            async for chunk in _try_streaming_dispatch(backend, {"task": "hi"}, order, ctx, []):
                chunks.append(chunk)

        # TS-003: Assert output behavior — stream must produce token + metadata chunks.
        token_chunks = [c for c in chunks if c.event_type == "token"]
        metadata_chunks = [c for c in chunks if c.event_type == "metadata"]
        assert len(token_chunks) == 1
        assert token_chunks[0].content == "response"
        assert len(metadata_chunks) == 1
        assert metadata_chunks[0].backend_used == "b1"
        # Mock assertions verify wiring at the tracker integration boundary.
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
        health_tracker = _make_health_tracker_mock()
        adapter = _make_mock_adapter(["Hi", " there"])

        cascade_mod = "dragonlight_router.dispatch.cascade"
        adapter_path = f"{cascade_mod}._adapters_mod.create_adapter"
        with (
            patch(f"{cascade_mod}._run_cascade", new_callable=AsyncMock, return_value=Ok([scored])),
            patch(adapter_path, return_value=adapter),
        ):
            chunks = []
            async for chunk in dispatch_stream(
                order,
                registry,
                budget_tracker,
                health_tracker,
                {},
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
        health_tracker = _make_health_tracker_mock()

        from dragonlight_router.core.errors import BudgetExceededError

        err = BudgetExceededError("no budget")

        with patch(
            "dragonlight_router.dispatch.cascade._run_cascade",
            new_callable=AsyncMock,
            return_value=Err(err),
        ):
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
        health_tracker = _make_health_tracker_mock()

        failing_adapter = _make_failing_adapter(RuntimeError("timeout"))
        success_adapter = _make_mock_adapter(["fallback response"])

        call_count = 0

        # DEVIATION TEST-MOCK-001: branching mock required
        # — factory returns different adapters per backend name.
        def _create_adapter(config):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return failing_adapter
            return success_adapter

        cascade_mod = "dragonlight_router.dispatch.cascade"
        adapter_path = f"{cascade_mod}._adapters_mod.create_adapter"
        with (
            patch(
                f"{cascade_mod}._run_cascade",
                new_callable=AsyncMock,
                return_value=Ok([sc1, sc2]),
            ),
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
        health_tracker = _make_health_tracker_mock()

        failing_adapter = _make_failing_adapter(RuntimeError("boom"))

        cascade_mod = "dragonlight_router.dispatch.cascade"
        adapter_path = f"{cascade_mod}._adapters_mod.create_adapter"
        with (
            patch(
                f"{cascade_mod}._run_cascade",
                new_callable=AsyncMock,
                return_value=Ok([sc1, sc2]),
            ),
            patch(adapter_path, return_value=failing_adapter),
        ):
            chunks = []
            async for chunk in dispatch_stream(
                order,
                registry,
                budget_tracker,
                health_tracker,
                {},
            ):
                chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].event_type == "error"
        assert "2 backends exhausted" in chunks[0].error_message
        assert "b1" in chunks[0].error_message
        assert "b2" in chunks[0].error_message


# ---------------------------------------------------------------------------
# Line 92 — get_cache() returns the module-level cache singleton
# ---------------------------------------------------------------------------


class TestGetCache:
    def test_get_cache_returns_none_when_not_configured(self):
        """[TM-004] get_cache returns None when caching is not configured (line 92)."""
        from dragonlight_router.dispatch.cascade import _reset_cache, get_cache

        _reset_cache()
        assert get_cache() is None

    def test_get_cache_returns_cache_after_configure(self, tmp_path):
        """[TM-004] get_cache returns the SimpleCache after configure_cache (line 92)."""
        from dragonlight_router.dispatch.cascade import _reset_cache, configure_cache, get_cache

        _reset_cache()
        cache = configure_cache(db_path=tmp_path / "test_cache.db")
        assert get_cache() is cache
        _reset_cache()


# ---------------------------------------------------------------------------
# Lines 274-286 — _run_ibr_stage exception → returns None (IBR-SYS-03)
# ---------------------------------------------------------------------------


class TestRunIbrStageSafe:
    @pytest.mark.asyncio
    async def test_ibr_stage_exception_returns_none(self, make_backend_config):
        """[IBR-SYS-03] _run_ibr_stage catches exceptions and returns None."""
        from dragonlight_router.config.schema import IntentClassificationConfig
        from dragonlight_router.dispatch.cascade import DispatchContext, _run_ibr_stage

        backend = make_backend_config(name="b1", provider="prov")
        order = _make_order()

        ctx = DispatchContext(
            registry=MagicMock(spec=BackendRegistry),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
            ibr_config=IntentClassificationConfig(enabled=True),
            spectrograph_loader=MagicMock(),
            classification_adapter=MagicMock(),
        )

        with patch(
            "dragonlight_router.dispatch.cascade.run_ibr_stage",
            new_callable=AsyncMock,
            side_effect=RuntimeError("IBR blew up"),
        ):
            result = await _run_ibr_stage(order, [backend], ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_ibr_stage_success_returns_result(self, make_backend_config):
        """[IBR-SYS-03] _run_ibr_stage returns IBRResult on success."""
        from dragonlight_router.config.schema import IntentClassificationConfig
        from dragonlight_router.dispatch.cascade import DispatchContext, _run_ibr_stage

        backend = make_backend_config(name="b1", provider="prov")
        order = _make_order()

        ctx = DispatchContext(
            registry=MagicMock(spec=BackendRegistry),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
            ibr_config=IntentClassificationConfig(enabled=True),
            spectrograph_loader=MagicMock(),
            classification_adapter=MagicMock(),
        )

        mock_ibr_result = MagicMock()
        mock_ibr_result.ibr_active = True

        with patch(
            "dragonlight_router.dispatch.cascade.run_ibr_stage",
            new_callable=AsyncMock,
            return_value=mock_ibr_result,
        ):
            result = await _run_ibr_stage(order, [backend], ctx)

        assert result is mock_ibr_result


# ---------------------------------------------------------------------------
# Line 610 — _pinned_dispatch_full returns preflight Err
# ---------------------------------------------------------------------------


class TestPinnedDispatchPreflightErr:
    @pytest.mark.asyncio
    async def test_pinned_dispatch_returns_preflight_err(self, make_backend_config):
        """[TM-004] _pinned_dispatch_full returns Err when preflight fails (line 610)."""
        from dragonlight_router.dispatch.cascade import _pinned_dispatch_full, _reset_cache

        _reset_cache()
        order = _make_order(model="nonexistent/model")
        ctx = _make_ctx()

        # Mock _pinned_preflight to return Err
        from dragonlight_router.core.types import ModelNotFoundError

        preflight_err = Err(
            ModelNotFoundError(
                model="nonexistent/model",
                message="Model not found",
            )
        )

        pinned_mod = "dragonlight_router.dispatch.pinned"
        with patch(f"{pinned_mod}._pinned_preflight", return_value=preflight_err):
            result = await _pinned_dispatch_full(order, ctx)

        assert result.is_err()


# ---------------------------------------------------------------------------
# Lines 1203, 1231-1233, 1244, 1329-1330 — cache operations
# ---------------------------------------------------------------------------


class TestCacheLookupAndStore:
    def test_try_cache_lookup_with_system_prompt(self, tmp_path):
        """_try_cache_lookup inserts system message when order has system_prompt."""
        from dragonlight_router.core.types import BackendTier, EngineResponse
        from dragonlight_router.dispatch.cascade import (
            _reset_cache,
            _store_cache_response,
            _try_cache_lookup,
            configure_cache,
        )

        _reset_cache()
        configure_cache(db_path=tmp_path / "cache.db")

        order = _make_order(system_prompt="Be helpful", operator_message="hello")
        response = EngineResponse(
            content="Hi there",
            backend_used="test-backend",
            backend_tier=BackendTier.SIMPLE,
            tokens_in=5,
            tokens_out=3,
            estimated_cost_usd=0.0,
            latency_ms=50.0,
            was_fallback=False,
            fallback_chain=[],
        )

        # Store and then look up — exercises lines 1203 and 1244
        _store_cache_response(order, response)
        cached = _try_cache_lookup(order)

        assert cached is not None
        assert cached.content == "Hi there"
        assert cached.backend_used == "test-backend"
        _reset_cache()

    def test_try_cache_lookup_deserialize_error_returns_none(self, tmp_path):
        """[TM-004] _try_cache_lookup returns None on corrupted cache data (lines 1231-1233)."""
        from dragonlight_router.caching.simple import SimpleCache
        from dragonlight_router.dispatch.cascade import (
            _reset_cache,
            _try_cache_lookup,
            configure_cache,
        )

        _reset_cache()
        cache = configure_cache(db_path=tmp_path / "cache.db")

        order = _make_order(operator_message="hello")

        # Manually put corrupted data in the cache
        messages = [{"role": "user", "content": order.operator_message}]
        cache_key = SimpleCache.make_key(
            model_id=order.intent_category,
            system_prompt=order.system_prompt,
            messages=messages,
            temperature=0.7,
            max_tokens=4096,
        )
        cache.put(cache_key, "not valid json {{{")

        result = _try_cache_lookup(order)
        assert result is None
        _reset_cache()


class TestDispatchCacheHit:
    @pytest.mark.asyncio
    async def test_dispatch_returns_ok_on_cache_hit(self, tmp_path):
        """[TM-004] dispatch() returns Ok(cached) on cache hit (lines 1329-1330)."""
        from dragonlight_router.core.types import BackendTier, EngineResponse
        from dragonlight_router.dispatch.cascade import (
            _reset_cache,
            _store_cache_response,
            configure_cache,
            dispatch,
        )

        _reset_cache()
        configure_cache(db_path=tmp_path / "cache.db")

        order = _make_order(operator_message="cached question")
        response = EngineResponse(
            content="cached answer",
            backend_used="cached-backend",
            backend_tier=BackendTier.SIMPLE,
            tokens_in=10,
            tokens_out=5,
            estimated_cost_usd=0.0,
            latency_ms=10.0,
            was_fallback=False,
            fallback_chain=[],
        )

        _store_cache_response(order, response)

        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = _make_health_tracker_mock()

        result = await dispatch(
            order,
            registry,
            budget_tracker,
            health_tracker,
            {},
        )

        assert result.is_ok()
        assert result.value.content == "cached answer"
        assert result.value.backend_used == "cached-backend"
        _reset_cache()


# ---------------------------------------------------------------------------
# Minimum output token threshold — insufficient output triggers fallback
# ---------------------------------------------------------------------------


class TestMinOutputTokens:
    @pytest.mark.asyncio
    async def test_insufficient_output_continues_fallback(self, make_backend_config):
        """Low output tokens from first backend triggers fallback to next candidate."""
        from dragonlight_router.dispatch.cascade import _handle_fallback_chain

        b1 = make_backend_config(name="b1", provider="prov1")
        b2 = make_backend_config(name="b2", provider="prov2")
        order = _make_order(min_output_tokens=50)
        ctx = _make_ctx()

        # First adapter returns only 5 output tokens (below threshold)
        low_response = MagicMock()
        low_response.content = "Hi"
        low_response.backend_used = "b1"
        low_response.tokens_out = 5

        good_response = MagicMock()
        good_response.content = "A proper response with enough tokens"
        good_response.backend_used = "b2"
        good_response.tokens_out = 100

        call_count = 0

        async def _mock_dispatch(bc, base_ctx, o, c, chain):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return Ok(low_response)
            return Ok(good_response)

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with patch(f"{cascade_mod}._try_adapter_dispatch", side_effect=_mock_dispatch):
            result = await _handle_fallback_chain([b1, b2], {"task": "hi"}, order, ctx)

        assert result.is_ok()
        assert result.value.backend_used == "b2"

    @pytest.mark.asyncio
    async def test_sufficient_output_returns_immediately(self, make_backend_config):
        """Output tokens meeting threshold returns without fallback."""
        from dragonlight_router.dispatch.cascade import _handle_fallback_chain

        b1 = make_backend_config(name="b1", provider="prov1")
        b2 = make_backend_config(name="b2", provider="prov2")
        order = _make_order(min_output_tokens=50)
        ctx = _make_ctx()

        response = MagicMock()
        response.content = "A proper response"
        response.backend_used = "b1"
        response.tokens_out = 200

        async def _mock_dispatch(bc, base_ctx, o, c, chain):
            return Ok(response)

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with patch(f"{cascade_mod}._try_adapter_dispatch", side_effect=_mock_dispatch):
            result = await _handle_fallback_chain([b1, b2], {"task": "hi"}, order, ctx)

        assert result.is_ok()
        assert result.value.backend_used == "b1"

    @pytest.mark.asyncio
    async def test_zero_threshold_disables_check(self, make_backend_config):
        """min_output_tokens=0 disables the minimum output check."""
        from dragonlight_router.dispatch.cascade import _handle_fallback_chain

        b1 = make_backend_config(name="b1", provider="prov1")
        order = _make_order(min_output_tokens=0)
        ctx = _make_ctx()

        response = MagicMock()
        response.content = "x"
        response.backend_used = "b1"
        response.tokens_out = 1

        async def _mock_dispatch(bc, base_ctx, o, c, chain):
            return Ok(response)

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with patch(f"{cascade_mod}._try_adapter_dispatch", side_effect=_mock_dispatch):
            result = await _handle_fallback_chain([b1], {"task": "hi"}, order, ctx)

        assert result.is_ok()
        assert result.value.tokens_out == 1

    @pytest.mark.asyncio
    async def test_all_backends_insufficient_output_returns_exhaustion(self, make_backend_config):
        """All backends returning insufficient output exhausts fallback chain."""
        from dragonlight_router.dispatch.cascade import _handle_fallback_chain

        b1 = make_backend_config(name="b1", provider="prov1")
        b2 = make_backend_config(name="b2", provider="prov2")
        order = _make_order(min_output_tokens=50)
        ctx = _make_ctx()

        low_response = MagicMock()
        low_response.content = "tiny"
        low_response.backend_used = "b1"
        low_response.tokens_out = 3

        async def _mock_dispatch(bc, base_ctx, o, c, chain):
            return Ok(low_response)

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with patch(f"{cascade_mod}._try_adapter_dispatch", side_effect=_mock_dispatch):
            result = await _handle_fallback_chain([b1, b2], {"task": "hi"}, order, ctx)

        assert result.is_err()


# ---------------------------------------------------------------------------
# Retired model filtering in cascade and fallback chain
# ---------------------------------------------------------------------------


class TestRetiredModelFiltering:
    @pytest.mark.asyncio
    async def test_cascade_filters_retired_models(self, make_backend_config):
        """_run_cascade excludes models retired by the health tracker."""
        from dragonlight_router.health.tracker import HealthTracker

        b1 = make_backend_config(name="b1", provider="prov1", model="model-a")
        b2 = make_backend_config(name="b2", provider="prov2", model="model-b")
        order = _make_order()

        health_tracker = HealthTracker()
        # Retire model-a via 404
        health_tracker.record_error("model-a", http_status=404)

        ctx = _make_ctx(health_tracker=health_tracker)

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with (
            patch(f"{cascade_mod}._run_mbr_stage", return_value=Ok([b1, b2])),
            patch(f"{cascade_mod}._run_ibr_stage", new_callable=AsyncMock, return_value=None),
            patch(f"{cascade_mod}._run_cbr_stage") as mock_cbr,
            patch(f"{cascade_mod}._run_lbr_stage") as mock_lbr,
        ):
            mock_cbr.return_value = Ok([ScoredCandidate(config=b2, score=0.8)])
            mock_lbr.return_value = Ok([ScoredCandidate(config=b2, score=0.8)])
            result = await _run_cascade(order, ctx)

        assert result.is_ok()
        # CBR should only have received model-b (model-a was retired)
        cbr_candidates = mock_cbr.call_args[0][1]  # second positional arg
        assert len(cbr_candidates) == 1
        assert cbr_candidates[0].model == "model-b"

    @pytest.mark.asyncio
    async def test_cascade_returns_err_when_all_retired(self, make_backend_config):
        """_run_cascade returns Err when all MBR candidates are retired."""
        from dragonlight_router.health.tracker import HealthTracker

        b1 = make_backend_config(name="b1", provider="prov1", model="model-a")
        order = _make_order()

        health_tracker = HealthTracker()
        health_tracker.record_error("model-a", http_status=404)

        ctx = _make_ctx(health_tracker=health_tracker)

        with patch(
            "dragonlight_router.dispatch.cascade._run_mbr_stage",
            return_value=Ok([b1]),
        ):
            result = await _run_cascade(order, ctx)

        assert result.is_err()
        assert "retired" in str(result.error).lower()

    @pytest.mark.asyncio
    async def test_fallback_chain_skips_retired_model(self, make_backend_config):
        """_handle_fallback_chain skips candidates retired mid-cascade."""
        from dragonlight_router.health.tracker import HealthTracker
        from dragonlight_router.dispatch.cascade import _handle_fallback_chain

        b1 = make_backend_config(name="b1", provider="prov1", model="model-a")
        b2 = make_backend_config(name="b2", provider="prov2", model="model-b")
        order = _make_order(min_output_tokens=0)

        health_tracker = HealthTracker()
        # model-a is retired before fallback chain starts
        health_tracker.record_error("model-a", http_status=403)

        ctx = _make_ctx(health_tracker=health_tracker)

        response = MagicMock()
        response.content = "good output"
        response.backend_used = "b2"
        response.tokens_out = 100

        async def _mock_dispatch(bc, base_ctx, o, c, chain):
            return Ok(response)

        cascade_mod = "dragonlight_router.dispatch.cascade"
        with patch(f"{cascade_mod}._try_adapter_dispatch", side_effect=_mock_dispatch):
            result = await _handle_fallback_chain([b1, b2], {"task": "hi"}, order, ctx)

        assert result.is_ok()
        assert result.value.backend_used == "b2"
