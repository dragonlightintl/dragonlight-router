"""Unit tests for model-pinning dispatch path.

Spec traceability: model-pinning-v0.1.0-spec.md
AC numbers: AC-PIN-001 through AC-PIN-022
"""
from __future__ import annotations

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dragonlight_router.config.schema import PinnedDispatchConfig, RouterConfig
from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import (
    BackendStatus,
    BackendTier,
    BudgetExhaustedError,
    DispatchFailure,
    DispatchOrder,
    EngineResponse,
    ModelNotFoundError,
    ModelUnhealthyError,
    StreamChunk,
)
from dragonlight_router.dispatch.cascade import (
    DispatchContext,
    _pinned_dispatch_full,
    _pinned_dispatch_stream,
    _pinned_preflight,
    _pinned_route,
    _reset_cache,
    dispatch,
    dispatch_stream,
    route,
)
from dragonlight_router.result import Err, Ok
from dragonlight_router.server.routes import (
    _build_dispatch_order,
    _format_dispatch_failure,
    _format_dispatch_response,
    _validate_dispatch_request,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CASCADE_MOD = "dragonlight_router.dispatch.cascade"
_ADAPTER_PATH = f"{_CASCADE_MOD}._adapters_mod.create_adapter"


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


def _make_ctx(
    registry=None,
    budget_tracker=None,
    health_tracker=None,
    config=None,
    pinned_dispatch_config=None,
):
    registry = registry or MagicMock(spec=BackendRegistry)
    budget_tracker = budget_tracker or MagicMock()
    health_tracker = health_tracker or MagicMock()
    config = config if config is not None else {}
    pdc = pinned_dispatch_config or PinnedDispatchConfig()
    return DispatchContext(
        registry=registry,
        budget_tracker=budget_tracker,
        health_tracker=health_tracker,
        config=config,
        pinned_dispatch_config=pdc,
    )


def _make_backend_and_state(
    make_backend_config,
    name="pinned/model-v1",
    provider="pinned-provider",
    tier=BackendTier.COMPLEX,
    status=BackendStatus.AVAILABLE,
    circuit_open=False,
):
    """Build a (GenerativeBackend-like wrapper, BackendState) pair for registry.get()."""
    backend_config = make_backend_config(name=name, provider=provider, tier=tier)
    backend = MagicMock()
    backend.config = backend_config
    state = MagicMock()
    state.status = status
    state.is_circuit_open.return_value = circuit_open
    return backend, state, backend_config


def _make_mock_adapter(chunks=None):
    """Create a mock adapter whose generate() yields the given chunks."""
    if chunks is None:
        chunks = ["Hello", " world"]
    adapter = MagicMock()
    adapter.status = BackendStatus.AVAILABLE

    async def _gen(*args, **kwargs):
        for chunk in chunks:
            yield chunk

    adapter.generate = _gen
    adapter.record_usage = MagicMock()
    return adapter


def _make_failing_adapter(exc):
    """Create a mock adapter whose generate() raises an exception."""
    adapter = MagicMock()
    adapter.status = BackendStatus.AVAILABLE

    async def _gen(*args, **kwargs):
        raise exc
        yield  # noqa: RET503 — unreachable, makes this an async generator

    adapter.generate = _gen
    adapter.record_usage = MagicMock()
    return adapter


# ===================================================================
# SECTION 1: Cascade pinned path — _pinned_preflight
# ===================================================================


class TestPinnedPreflightHappyPath:
    """[AC-PIN-002] Pinned preflight resolves backend via registry."""

    def test_valid_model_returns_ok_backend_config(self, make_backend_config):
        """[AC-PIN-002] registry.get(model) called, returns Ok(BackendConfig)."""
        backend, state, backend_config = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(registry=registry, budget_tracker=budget_tracker)

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Ok)
        assert result.value is backend_config
        registry.get.assert_called_once_with("pinned/model-v1")

    def test_returns_exact_backend_config_from_registry(self, make_backend_config):
        """[AC-PIN-002] The returned BackendConfig is the one from the registry entry."""
        backend, state, backend_config = _make_backend_and_state(
            make_backend_config, name="anthropic/claude-sonnet-4",
        )
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True

        order = _make_order(model="anthropic/claude-sonnet-4")
        ctx = _make_ctx(registry=registry, budget_tracker=budget_tracker)

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Ok)
        assert result.value.name == "anthropic/claude-sonnet-4"


class TestPinnedPreflightModelNotFound:
    """[AC-PIN-003] Model not found returns Err(ModelNotFoundError)."""

    def test_registry_returns_none_none(self, make_backend_config):
        """[AC-PIN-003] registry.get() returns (None, None) -> ModelNotFoundError."""
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (None, None)

        order = _make_order(model="nonexistent/model")
        ctx = _make_ctx(registry=registry)

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Err)
        assert isinstance(result.error, ModelNotFoundError)
        assert result.error.model == "nonexistent/model"
        assert "not found" in result.error.message

    def test_backend_none_state_some(self, make_backend_config):
        """[AC-PIN-003] Backend None with non-None state still fails as not found."""
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (None, MagicMock())

        order = _make_order(model="partial/model")
        ctx = _make_ctx(registry=registry)

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Err)
        assert isinstance(result.error, ModelNotFoundError)


class TestPinnedPreflightModelRetired:
    """[AC-PIN-004] Retired model returns Err(ModelUnhealthyError)."""

    def test_retired_status_returns_unhealthy_error(self, make_backend_config):
        """[AC-PIN-004] BackendStatus.RETIRED -> ModelUnhealthyError with status=retired."""
        backend, state, _ = _make_backend_and_state(
            make_backend_config, status=BackendStatus.RETIRED,
        )
        state.is_circuit_open.return_value = False
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(registry=registry)

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Err)
        assert isinstance(result.error, ModelUnhealthyError)
        assert result.error.status == "retired"
        assert "retired" in result.error.message

    def test_key_invalid_treated_as_retired(self, make_backend_config):
        """[HAZ-PIN-003] KEY_INVALID treated same as retired."""
        backend, state, _ = _make_backend_and_state(
            make_backend_config, status=BackendStatus.KEY_INVALID,
        )
        state.is_circuit_open.return_value = False
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(registry=registry)

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Err)
        assert isinstance(result.error, ModelUnhealthyError)
        assert result.error.status == "retired"


class TestPinnedPreflightCircuitOpen:
    """[AC-PIN-005] [AC-PIN-006] Circuit-open with honor_health toggling."""

    def test_circuit_open_honor_health_true_returns_unhealthy(self, make_backend_config):
        """[AC-PIN-005] Circuit open + honor_health=True -> ModelUnhealthyError."""
        backend, state, _ = _make_backend_and_state(
            make_backend_config, circuit_open=True,
        )
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            pinned_dispatch_config=PinnedDispatchConfig(honor_health=True),
        )

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Err)
        assert isinstance(result.error, ModelUnhealthyError)
        assert result.error.status == "circuit_open"

    def test_circuit_open_honor_health_false_passes(self, make_backend_config):
        """[AC-PIN-006] Circuit open + honor_health=False -> dispatch attempted."""
        backend, state, backend_config = _make_backend_and_state(
            make_backend_config, circuit_open=True,
        )
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            pinned_dispatch_config=PinnedDispatchConfig(honor_health=False),
        )

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Ok)
        assert result.value is backend_config

    def test_circuit_closed_honor_health_true_passes(self, make_backend_config):
        """[AC-PIN-005] Healthy model + honor_health=True -> passes through."""
        backend, state, backend_config = _make_backend_and_state(
            make_backend_config, circuit_open=False,
        )
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            pinned_dispatch_config=PinnedDispatchConfig(honor_health=True),
        )

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Ok)


class TestPinnedPreflightBudget:
    """[AC-PIN-007] Budget enforcement for pinned dispatch."""

    def test_budget_exhausted_returns_error(self, make_backend_config):
        """[AC-PIN-007] has_capacity returns False -> BudgetExhaustedError."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = False

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(registry=registry, budget_tracker=budget_tracker)

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Err)
        assert isinstance(result.error, BudgetExhaustedError)
        assert result.error.model == "pinned/model-v1"
        assert result.error.provider == "pinned-provider"
        assert "budget exhausted" in result.error.message

    def test_budget_available_passes(self, make_backend_config):
        """[AC-PIN-007] has_capacity returns True -> dispatch proceeds."""
        backend, state, backend_config = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(registry=registry, budget_tracker=budget_tracker)

        result = _pinned_preflight(order, ctx)

        assert isinstance(result, Ok)


# ===================================================================
# SECTION 2: Pinned route — _pinned_route / route()
# ===================================================================


class TestPinnedRoute:
    """[AC-PIN-001] [AC-PIN-002] Pinned route bypasses cascade."""

    @pytest.mark.asyncio
    async def test_pinned_route_returns_backend_config(self, make_backend_config):
        """[AC-PIN-002] _pinned_route returns Ok(BackendConfig) on success."""
        backend, state, backend_config = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(registry=registry, budget_tracker=budget_tracker)

        result = await _pinned_route(order, ctx)

        assert isinstance(result, Ok)
        assert result.value is backend_config

    @pytest.mark.asyncio
    async def test_route_with_model_set_skips_cascade(self, make_backend_config):
        """[AC-PIN-001] route() with model set does NOT call _run_cascade."""
        backend, state, backend_config = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        health_tracker = MagicMock()

        order = _make_order(model="pinned/model-v1")

        with patch(f"{_CASCADE_MOD}._run_cascade", new_callable=AsyncMock) as mock_cascade:
            result = await route(order, registry, budget_tracker, health_tracker, {})

        assert isinstance(result, Ok)
        mock_cascade.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_with_model_set_does_not_call_run_mbr(self, make_backend_config):
        """[AC-PIN-001] MBR stage never invoked for pinned dispatch."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        health_tracker = MagicMock()

        order = _make_order(model="pinned/model-v1")

        with patch(f"{_CASCADE_MOD}._run_mbr_stage") as mock_mbr:
            await route(order, registry, budget_tracker, health_tracker, {})

        mock_mbr.assert_not_called()

    @pytest.mark.asyncio
    async def test_route_without_model_runs_cascade(self, make_backend_config):
        """[AC-PIN-001] route() without model runs the normal cascade."""
        backend_config = make_backend_config(name="b1", provider="prov")
        order = _make_order(model=None)
        registry = MagicMock(spec=BackendRegistry)
        budget_tracker = MagicMock()
        health_tracker = MagicMock()

        with (
            patch(
                f"{_CASCADE_MOD}._run_cascade",
                new_callable=AsyncMock,
                return_value=Ok([MagicMock(config=backend_config, score=0.9)]),
            ) as mock_cascade,
            patch(
                f"{_CASCADE_MOD}.select_final_candidate",
                return_value=backend_config,
            ),
        ):
            result = await route(order, registry, budget_tracker, health_tracker, {})

        assert isinstance(result, Ok)
        mock_cascade.assert_called_once()


# ===================================================================
# SECTION 3: dispatch_mode and was_fallback fields
# ===================================================================


class TestDispatchModeField:
    """[AC-PIN-014] dispatch_mode is 'pinned' for pinned, 'cascade' for cascade."""

    def test_engine_response_default_dispatch_mode(self):
        """[AC-PIN-014] EngineResponse defaults to dispatch_mode='cascade'."""
        resp = EngineResponse(
            content="test",
            backend_used="b1",
            backend_tier=BackendTier.COMPLEX,
            tokens_in=10,
            tokens_out=20,
            estimated_cost_usd=0.001,
            latency_ms=100.0,
            was_fallback=False,
            fallback_chain=[],
        )
        assert resp.dispatch_mode == "cascade"

    def test_engine_response_pinned_dispatch_mode(self):
        """[AC-PIN-014] EngineResponse with dispatch_mode='pinned'."""
        resp = EngineResponse(
            content="test",
            backend_used="b1",
            backend_tier=BackendTier.COMPLEX,
            tokens_in=10,
            tokens_out=20,
            estimated_cost_usd=0.001,
            latency_ms=100.0,
            was_fallback=False,
            fallback_chain=[],
            dispatch_mode="pinned",
        )
        assert resp.dispatch_mode == "pinned"

    def test_stream_chunk_default_dispatch_mode(self):
        """[AC-PIN-014] StreamChunk defaults to dispatch_mode='cascade'."""
        chunk = StreamChunk(event_type="metadata")
        assert chunk.dispatch_mode == "cascade"

    def test_stream_chunk_pinned_dispatch_mode(self):
        """[AC-PIN-014] StreamChunk with dispatch_mode='pinned'."""
        chunk = StreamChunk(event_type="metadata", dispatch_mode="pinned")
        assert chunk.dispatch_mode == "pinned"


class TestWasFallbackAndFallbackChain:
    """[AC-PIN-015] Pinned dispatch always has was_fallback=False, empty fallback_chain."""

    @pytest.mark.asyncio
    async def test_pinned_dispatch_full_was_fallback_false(self, make_backend_config):
        """[AC-PIN-015] _pinned_dispatch_full sets was_fallback=False."""
        _reset_cache()
        backend, state, backend_config = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["ok"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            result = await _pinned_dispatch_full(order, ctx)

        assert isinstance(result, Ok)
        assert result.value.was_fallback is False
        assert result.value.fallback_chain == []

    @pytest.mark.asyncio
    async def test_pinned_dispatch_full_dispatch_mode_pinned(self, make_backend_config):
        """[AC-PIN-014] _pinned_dispatch_full sets dispatch_mode='pinned'."""
        _reset_cache()
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["response content"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            result = await _pinned_dispatch_full(order, ctx)

        assert isinstance(result, Ok)
        assert result.value.dispatch_mode == "pinned"


# ===================================================================
# SECTION 4: Full dispatch path — _pinned_dispatch_full
# ===================================================================


class TestPinnedDispatchFull:
    """[AC-PIN-009] [AC-PIN-010] [AC-PIN-012] Full pinned dispatch path."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _reset_cache()
        yield
        _reset_cache()

    @pytest.mark.asyncio
    async def test_happy_path_returns_engine_response(self, make_backend_config):
        """Pinned dispatch returns Ok(EngineResponse) with correct fields."""
        backend, state, backend_config = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["Hello", " world"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            result = await _pinned_dispatch_full(order, ctx)

        assert isinstance(result, Ok)
        resp = result.value
        assert resp.content == "Hello world"
        assert resp.backend_used == "pinned/model-v1"
        assert resp.dispatch_mode == "pinned"
        assert resp.was_fallback is False
        assert resp.fallback_chain == []
        assert resp.tokens_in >= 0
        assert resp.tokens_out >= 0
        assert resp.latency_ms > 0

    @pytest.mark.asyncio
    async def test_record_request_called_after_success(self, make_backend_config):
        """[AC-PIN-009] record_request called with correct provider after pinned dispatch."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["ok"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            await _pinned_dispatch_full(order, ctx)

        budget_tracker.record_request.assert_called_once()
        call_args = budget_tracker.record_request.call_args
        assert call_args[0][0] == "pinned-provider"

    @pytest.mark.asyncio
    async def test_record_success_called_on_success(self, make_backend_config):
        """[AC-PIN-010] health_tracker.record_success called after success."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["ok"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            await _pinned_dispatch_full(order, ctx)

        health_tracker.record_success.assert_called_once()

    @pytest.mark.asyncio
    async def test_record_error_called_on_adapter_failure(self, make_backend_config):
        """[AC-PIN-010] health_tracker.record_error called on adapter failure."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_failing_adapter(RuntimeError("adapter exploded"))

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            result = await _pinned_dispatch_full(order, ctx)

        assert isinstance(result, Err)
        health_tracker.record_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_adapter_failure_returns_err_no_fallback(self, make_backend_config):
        """Pinned dispatch adapter failure returns Err immediately, no fallback."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_failing_adapter(ConnectionError("timeout"))

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            result = await _pinned_dispatch_full(order, ctx)

        assert isinstance(result, Err)
        error = result.error
        assert isinstance(error, DispatchFailure)
        assert "pinned model dispatch failed" in error.message

    @pytest.mark.asyncio
    async def test_rate_limit_exhausted_returns_budget_error(self, make_backend_config):
        """[AC-PIN-008] check_and_reserve returns False -> BudgetExhaustedError."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=False)
        health_tracker = MagicMock()

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        result = await _pinned_dispatch_full(order, ctx)

        assert isinstance(result, Err)
        assert isinstance(result.error, BudgetExhaustedError)
        assert "rate limit exhausted" in result.error.message

    @pytest.mark.asyncio
    async def test_fresh_adapter_created_per_dispatch(self, make_backend_config):
        """[AC-PIN-012] [HAZ-014] Fresh adapter instance per pinned dispatch."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["ok"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter) as mock_create:
            await _pinned_dispatch_full(order, ctx)

        mock_create.assert_called_once()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_cached_response(self, make_backend_config):
        """[AC-PIN-018] Cache hit returns cached response without calling adapter."""
        cached_response = EngineResponse(
            content="cached",
            backend_used="pinned/model-v1",
            backend_tier=BackendTier.COMPLEX,
            tokens_in=5,
            tokens_out=10,
            estimated_cost_usd=0.0,
            latency_ms=0.0,
            was_fallback=False,
            fallback_chain=[],
        )
        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx()

        with patch(f"{_CASCADE_MOD}._try_cache_lookup", return_value=cached_response):
            result = await _pinned_dispatch_full(order, ctx)

        assert isinstance(result, Ok)
        assert result.value.content == "cached"

    @pytest.mark.asyncio
    async def test_cache_miss_calls_adapter(self, make_backend_config):
        """[AC-PIN-018] Cache miss proceeds with adapter dispatch."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["fresh response"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with (
            patch(f"{_CASCADE_MOD}._try_cache_lookup", return_value=None),
            patch(_ADAPTER_PATH, return_value=adapter),
        ):
            result = await _pinned_dispatch_full(order, ctx)

        assert isinstance(result, Ok)
        assert result.value.content == "fresh response"

    @pytest.mark.asyncio
    async def test_cache_store_called_on_success(self, make_backend_config):
        """[AC-PIN-019] Cache store called after successful pinned dispatch."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["cached after"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with (
            patch(f"{_CASCADE_MOD}._try_cache_lookup", return_value=None),
            patch(_ADAPTER_PATH, return_value=adapter),
            patch(f"{_CASCADE_MOD}._store_cache_response") as mock_store,
        ):
            result = await _pinned_dispatch_full(order, ctx)

        assert isinstance(result, Ok)
        mock_store.assert_called_once()


# ===================================================================
# SECTION 5: Streaming pinned dispatch — _pinned_dispatch_stream
# ===================================================================


class TestPinnedDispatchStream:
    """[AC-PIN-021] Pinned dispatch streaming path."""

    @pytest.mark.asyncio
    async def test_streams_token_chunks_and_metadata(self, make_backend_config):
        """[AC-PIN-021] Streaming pinned dispatch yields tokens then metadata."""
        backend, state, backend_config = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["Hello", " stream"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            chunks = []
            async for chunk in _pinned_dispatch_stream(order, ctx):
                chunks.append(chunk)

        token_chunks = [c for c in chunks if c.event_type == "token"]
        metadata_chunks = [c for c in chunks if c.event_type == "metadata"]
        assert len(token_chunks) == 2
        assert token_chunks[0].content == "Hello"
        assert token_chunks[1].content == " stream"
        assert len(metadata_chunks) == 1

    @pytest.mark.asyncio
    async def test_stream_token_dispatch_mode_pinned(self, make_backend_config):
        """[AC-PIN-014] Streaming token chunks have dispatch_mode='pinned'."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["content"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            chunks = []
            async for chunk in _pinned_dispatch_stream(order, ctx):
                chunks.append(chunk)

        for chunk in chunks:
            assert chunk.dispatch_mode == "pinned"

    @pytest.mark.asyncio
    async def test_stream_metadata_was_fallback_false(self, make_backend_config):
        """[AC-PIN-015] Stream metadata has was_fallback=False."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["ok"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            chunks = []
            async for chunk in _pinned_dispatch_stream(order, ctx):
                chunks.append(chunk)

        metadata = [c for c in chunks if c.event_type == "metadata"]
        assert len(metadata) == 1
        assert metadata[0].was_fallback is False
        assert metadata[0].fallback_chain == []

    @pytest.mark.asyncio
    async def test_stream_preflight_failure_yields_error(self, make_backend_config):
        """[AC-PIN-003] Preflight failure in streaming yields error chunk."""
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (None, None)

        order = _make_order(model="nonexistent/model")
        ctx = _make_ctx(registry=registry)

        chunks = []
        async for chunk in _pinned_dispatch_stream(order, ctx):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].event_type == "error"
        assert "not found" in chunks[0].error_message
        assert chunks[0].dispatch_mode == "pinned"

    @pytest.mark.asyncio
    async def test_stream_adapter_failure_yields_error(self, make_backend_config):
        """Stream adapter failure yields error chunk, no fallback."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_failing_adapter(RuntimeError("boom"))

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            chunks = []
            async for chunk in _pinned_dispatch_stream(order, ctx):
                chunks.append(chunk)

        error_chunks = [c for c in chunks if c.event_type == "error"]
        assert len(error_chunks) == 1
        assert "pinned model dispatch failed" in error_chunks[0].error_message
        assert error_chunks[0].dispatch_mode == "pinned"

    @pytest.mark.asyncio
    async def test_stream_rate_limit_yields_error(self, make_backend_config):
        """[AC-PIN-008] Rate limit exhaustion in streaming yields error chunk."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=False)
        health_tracker = MagicMock()

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        chunks = []
        async for chunk in _pinned_dispatch_stream(order, ctx):
            chunks.append(chunk)

        assert len(chunks) == 1
        assert chunks[0].event_type == "error"
        assert "rate limit exhausted" in chunks[0].error_message

    @pytest.mark.asyncio
    async def test_stream_records_health_and_budget(self, make_backend_config):
        """[AC-PIN-009] [AC-PIN-010] Streaming records success in trackers."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["ok"])

        order = _make_order(model="pinned/model-v1")
        ctx = _make_ctx(
            registry=registry,
            budget_tracker=budget_tracker,
            health_tracker=health_tracker,
        )

        with patch(_ADAPTER_PATH, return_value=adapter):
            chunks = []
            async for chunk in _pinned_dispatch_stream(order, ctx):
                chunks.append(chunk)

        health_tracker.record_success.assert_called_once()
        budget_tracker.record_request.assert_called_once()


# ===================================================================
# SECTION 6: dispatch() and dispatch_stream() top-level integration
# ===================================================================


class TestDispatchTopLevel:
    """Top-level dispatch() and dispatch_stream() with pinned model."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        _reset_cache()
        yield
        _reset_cache()

    @pytest.mark.asyncio
    async def test_dispatch_with_model_calls_pinned_path(self, make_backend_config):
        """dispatch() with model set routes to pinned dispatch path."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["dispatched"])

        order = _make_order(model="pinned/model-v1")

        with (
            patch(_ADAPTER_PATH, return_value=adapter),
            patch(f"{_CASCADE_MOD}._run_cascade", new_callable=AsyncMock) as mock_cascade,
        ):
            result = await dispatch(
                order, registry, budget_tracker, health_tracker, {},
            )

        assert isinstance(result, Ok)
        assert result.value.dispatch_mode == "pinned"
        mock_cascade.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatch_stream_with_model_calls_pinned_path(self, make_backend_config):
        """dispatch_stream() with model set routes to pinned dispatch path."""
        backend, state, _ = _make_backend_and_state(make_backend_config)
        registry = MagicMock(spec=BackendRegistry)
        registry.get.return_value = (backend, state)
        budget_tracker = MagicMock()
        budget_tracker.has_capacity.return_value = True
        budget_tracker.check_and_reserve = AsyncMock(return_value=True)
        health_tracker = MagicMock()

        adapter = _make_mock_adapter(["streamed"])

        order = _make_order(model="pinned/model-v1")

        with (
            patch(_ADAPTER_PATH, return_value=adapter),
            patch(f"{_CASCADE_MOD}._run_cascade", new_callable=AsyncMock) as mock_cascade,
        ):
            chunks = []
            async for chunk in dispatch_stream(
                order, registry, budget_tracker, health_tracker, {},
            ):
                chunks.append(chunk)

        token_chunks = [c for c in chunks if c.event_type == "token"]
        assert len(token_chunks) == 1
        assert token_chunks[0].content == "streamed"
        mock_cascade.assert_not_called()

        # Verify all chunks have pinned dispatch mode
        for chunk in chunks:
            assert chunk.dispatch_mode == "pinned"


# ===================================================================
# SECTION 7: API layer — validation, build, format
# ===================================================================


class TestValidateDispatchRequest:
    """[AC-PIN-016] [AC-PIN-017] Model field validation in API layer."""

    def test_model_present_intent_optional(self):
        """[AC-PIN-016] model present -> intent_category/specific_intent optional."""
        body = {
            "model": "anthropic/claude-sonnet-4",
            "operator_message": "hello",
            "context_tokens": 0,
        }
        assert _validate_dispatch_request(body) is None

    def test_model_absent_requires_intent_fields(self):
        """[AC-PIN-017] model absent -> intent_category/specific_intent required."""
        body = {
            "operator_message": "hello",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "intent_category" in error

    def test_model_absent_all_fields_present_passes(self):
        """[AC-PIN-017] model absent with all fields -> passes."""
        body = {
            "intent_category": "test",
            "specific_intent": "test",
            "operator_message": "hello",
            "context_tokens": 0,
        }
        assert _validate_dispatch_request(body) is None

    def test_model_must_be_string(self):
        """model must be a string."""
        body = {
            "model": 42,
            "operator_message": "hello",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "model" in error.lower()

    def test_model_must_be_nonempty(self):
        """model must be non-empty."""
        body = {
            "model": "",
            "operator_message": "hello",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None

    def test_model_under_500_chars(self):
        """model must be under 500 chars."""
        body = {
            "model": "x" * 501,
            "operator_message": "hello",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "500" in error

    def test_model_exactly_500_chars_passes(self):
        """model at exactly 500 chars passes."""
        body = {
            "model": "x" * 500,
            "operator_message": "hello",
            "context_tokens": 0,
        }
        assert _validate_dispatch_request(body) is None

    def test_model_none_treated_as_cascade(self):
        """model=None acts as cascade mode (requires intent fields)."""
        body = {
            "model": None,
            "operator_message": "hello",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "intent_category" in error

    def test_model_with_intent_fields_passes(self):
        """model present with optional intent fields included still passes."""
        body = {
            "model": "anthropic/claude-sonnet-4",
            "intent_category": "test",
            "specific_intent": "test",
            "operator_message": "hello",
            "context_tokens": 0,
        }
        assert _validate_dispatch_request(body) is None

    def test_pinned_still_validates_context_tokens(self):
        """Pinned dispatch still requires context_tokens."""
        body = {
            "model": "anthropic/claude-sonnet-4",
            "operator_message": "hello",
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "context_tokens" in error

    def test_pinned_still_validates_operator_message(self):
        """Pinned dispatch still requires operator_message."""
        body = {
            "model": "anthropic/claude-sonnet-4",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "operator_message" in error


class TestBuildDispatchOrder:
    """_build_dispatch_order passes model field through."""

    def test_model_field_passed_through(self):
        """model field from body is set on DispatchOrder."""
        body = {
            "model": "anthropic/claude-sonnet-4",
            "operator_message": "hello",
            "context_tokens": 100,
        }
        order = _build_dispatch_order(body)
        assert order.model == "anthropic/claude-sonnet-4"

    def test_model_absent_defaults_to_none(self):
        """model absent from body -> DispatchOrder.model is None."""
        body = {
            "intent_category": "test",
            "specific_intent": "test",
            "operator_message": "hello",
            "context_tokens": 100,
        }
        order = _build_dispatch_order(body)
        assert order.model is None

    def test_pinned_defaults_intent_fields(self):
        """[AC-PIN-016] Pinned dispatch defaults intent_category/specific_intent to ''."""
        body = {
            "model": "anthropic/claude-sonnet-4",
            "operator_message": "hello",
            "context_tokens": 100,
        }
        order = _build_dispatch_order(body)
        assert order.intent_category == ""
        assert order.specific_intent == ""


class TestFormatDispatchResponse:
    """Response includes dispatch_mode field."""

    def test_dispatch_mode_in_json_response(self):
        """[AC-PIN-014] dispatch_mode included in JSON response."""
        resp = EngineResponse(
            content="test content",
            backend_used="test-backend",
            backend_tier=BackendTier.COMPLEX,
            tokens_in=10,
            tokens_out=20,
            estimated_cost_usd=0.001,
            latency_ms=50.0,
            was_fallback=False,
            fallback_chain=[],
            dispatch_mode="pinned",
        )
        json_resp = _format_dispatch_response(resp)
        import json
        body = json.loads(json_resp.body)
        assert body["dispatch_mode"] == "pinned"

    def test_cascade_dispatch_mode_in_response(self):
        """[AC-PIN-014] Cascade dispatch_mode in JSON response."""
        resp = EngineResponse(
            content="test",
            backend_used="test-backend",
            backend_tier=BackendTier.COMPLEX,
            tokens_in=10,
            tokens_out=20,
            estimated_cost_usd=0.001,
            latency_ms=50.0,
            was_fallback=False,
            fallback_chain=[],
            dispatch_mode="cascade",
        )
        json_resp = _format_dispatch_response(resp)
        import json
        body = json.loads(json_resp.body)
        assert body["dispatch_mode"] == "cascade"


class TestFormatDispatchFailure:
    """Error mapping for pinned dispatch error types."""

    def test_model_not_found_returns_400(self):
        """[AC-PIN-003] ModelNotFoundError -> 400."""
        error = ModelNotFoundError(
            model="nonexistent/model",
            message="pinned model not found in registry: nonexistent/model",
        )
        resp = _format_dispatch_failure(error)
        assert resp.status_code == 400

    def test_model_not_found_includes_model_in_body(self):
        """ModelNotFoundError body includes model field."""
        error = ModelNotFoundError(model="bad/model", message="not found")
        resp = _format_dispatch_failure(error)
        import json
        body = json.loads(resp.body)
        assert body["model"] == "bad/model"

    def test_model_unhealthy_returns_503(self):
        """[AC-PIN-005] ModelUnhealthyError -> 503."""
        error = ModelUnhealthyError(
            model="sick/model",
            status="circuit_open",
            message="pinned model is unhealthy",
        )
        resp = _format_dispatch_failure(error)
        assert resp.status_code == 503

    def test_model_unhealthy_includes_status(self):
        """ModelUnhealthyError body includes status field."""
        error = ModelUnhealthyError(
            model="sick/model", status="retired", message="retired",
        )
        resp = _format_dispatch_failure(error)
        import json
        body = json.loads(resp.body)
        assert body["status"] == "retired"

    def test_budget_exhausted_returns_429(self):
        """[AC-PIN-007] BudgetExhaustedError -> 429."""
        error = BudgetExhaustedError(
            model="expensive/model",
            provider="expensive-provider",
            message="budget exhausted",
        )
        resp = _format_dispatch_failure(error)
        assert resp.status_code == 429

    def test_budget_exhausted_includes_provider(self):
        """BudgetExhaustedError body includes provider field."""
        error = BudgetExhaustedError(
            model="m", provider="prov", message="exhausted",
        )
        resp = _format_dispatch_failure(error)
        import json
        body = json.loads(resp.body)
        assert body["provider"] == "prov"

    def test_dispatch_failure_returns_500(self):
        """DispatchFailure (cascade exhaustion) -> 500."""
        error = DispatchFailure(
            message="all backends exhausted",
            attempted_backends=["b1", "b2"],
            error_details={"error_type": "RuntimeError"},
        )
        resp = _format_dispatch_failure(error)
        assert resp.status_code == 500


# ===================================================================
# SECTION 8: Config — PinnedDispatchConfig
# ===================================================================


class TestPinnedDispatchConfig:
    """[AC-PIN-022] PinnedDispatchConfig defaults and construction."""

    def test_honor_health_defaults_to_true(self):
        """[AC-PIN-022] Default honor_health is True."""
        config = PinnedDispatchConfig()
        assert config.honor_health is True

    def test_honor_health_set_to_false(self):
        """honor_health can be set to False."""
        config = PinnedDispatchConfig(honor_health=False)
        assert config.honor_health is False

    def test_from_dict(self):
        """PinnedDispatchConfig constructed from dict."""
        config = PinnedDispatchConfig(**{"honor_health": False})
        assert config.honor_health is False

    def test_frozen(self):
        """PinnedDispatchConfig is frozen (immutable)."""
        from pydantic import ValidationError
        config = PinnedDispatchConfig()
        with pytest.raises(ValidationError):
            config.honor_health = False  # type: ignore[misc]

    def test_router_config_includes_pinned_dispatch(self):
        """RouterConfig includes pinned_dispatch field with defaults."""
        router_config = RouterConfig()
        assert isinstance(router_config.pinned_dispatch, PinnedDispatchConfig)
        assert router_config.pinned_dispatch.honor_health is True


# ===================================================================
# SECTION 9: DispatchOrder model field
# ===================================================================


class TestDispatchOrderModelField:
    """DispatchOrder model field behavior."""

    def test_model_default_is_none(self):
        """DispatchOrder.model defaults to None."""
        order = _make_order()
        assert order.model is None

    def test_model_set_to_string(self):
        """DispatchOrder.model can be set to a string."""
        order = _make_order(model="anthropic/claude-sonnet-4")
        assert order.model == "anthropic/claude-sonnet-4"

    def test_model_frozen(self):
        """DispatchOrder.model is frozen."""
        order = _make_order(model="test")
        with pytest.raises(FrozenInstanceError):
            order.model = "different"  # type: ignore[misc]


# ===================================================================
# SECTION 10: Pinned dispatch context (DispatchContext)
# ===================================================================


class TestDispatchContextPinnedConfig:
    """DispatchContext carries pinned_dispatch_config."""

    def test_default_pinned_config(self):
        """DispatchContext defaults to PinnedDispatchConfig()."""
        ctx = _make_ctx()
        assert isinstance(ctx.pinned_dispatch_config, PinnedDispatchConfig)
        assert ctx.pinned_dispatch_config.honor_health is True

    def test_custom_pinned_config(self):
        """DispatchContext accepts custom pinned_dispatch_config."""
        config = PinnedDispatchConfig(honor_health=False)
        ctx = _make_ctx(pinned_dispatch_config=config)
        assert ctx.pinned_dispatch_config.honor_health is False
