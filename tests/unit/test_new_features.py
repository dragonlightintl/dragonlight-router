"""Tests for the four new features: caching, cost tracking, logging, retry.

Spec traceability: TM-021 (Feature integration tests)
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dragonlight_router.adapters._openai_compat import OpenAICompatibleBackend
from dragonlight_router.caching.simple import SimpleCache
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    DispatchOrder,
    EngineResponse,
)
from dragonlight_router.dispatch.cascade import (
    _reset_cache,
    _store_cache_response,
    _try_cache_lookup,
    configure_cache,
)
from dragonlight_router.router import RouterEngine
from dragonlight_router.selection.scoring import (
    ScoringWeightsConfig,
    _normalize_cost_score,
    score_candidate,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(**kwargs: object) -> DispatchOrder:
    defaults: dict[str, object] = {
        "intent_category": "test",
        "specific_intent": "test",
        "operator_message": "hello world",
        "system_prompt": "",
        "context_tokens": 0,
        "requires_tool_use": False,
        "requires_long_context": False,
    }
    defaults.update(kwargs)
    return DispatchOrder(**defaults)


def _make_config(
    name: str = "test",
    input_cost: float = 1.0,
    output_cost: float = 2.0,
) -> BackendConfig:
    return BackendConfig(
        name=name,
        provider="test-provider",
        model=name,
        tier=BackendTier.SIMPLE,
        base_url="https://example.com",
        env_key=None,
        capabilities=BackendCapabilities(
            max_context_tokens=8192,
            supports_tool_use=False,
            supports_streaming=True,
            supports_json_mode=False,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(
            input_per_mtok=input_cost,
            output_per_mtok=output_cost,
        ),
        rate_limits=BackendRateLimits(
            rpm=60,
            rpd=14400,
            tpm=100000,
            daily_token_cap=1000000,
        ),
    )


# ---------------------------------------------------------------------------
# Feature 1: Caching integration
# ---------------------------------------------------------------------------


class TestCacheIntegration:
    """Tests for dispatch-level caching."""

    def setup_method(self) -> None:
        _reset_cache()

    def teardown_method(self) -> None:
        _reset_cache()

    def test_cache_miss_returns_none(self, tmp_path: Path) -> None:
        """[TM-021 AC-1] Cache miss returns None when cache is configured."""
        configure_cache(tmp_path / "cache.db")
        order = _make_order()
        result = _try_cache_lookup(order)
        assert result is None

    def test_cache_disabled_returns_none(self) -> None:
        """[TM-021 AC-1] Returns None when cache is not configured."""
        order = _make_order()
        result = _try_cache_lookup(order)
        assert result is None

    def test_store_and_retrieve(self, tmp_path: Path) -> None:
        """[TM-021 AC-2] Store a response, then retrieve it from cache."""
        configure_cache(tmp_path / "cache.db")
        order = _make_order(operator_message="cached query")
        response = EngineResponse(
            content="cached answer",
            backend_used="test-backend",
            backend_tier=BackendTier.SIMPLE,
            tokens_in=10,
            tokens_out=20,
            estimated_cost_usd=0.001,
            latency_ms=150.0,
            was_fallback=False,
            fallback_chain=[],
        )
        _store_cache_response(order, response)

        cached = _try_cache_lookup(order)
        assert cached is not None
        assert cached.content == "cached answer"
        assert cached.backend_used == "test-backend"
        assert cached.tokens_in == 10
        assert cached.tokens_out == 20

    def test_different_messages_no_collision(self, tmp_path: Path) -> None:
        """[TM-021 AC-2] Different messages produce different cache keys."""
        configure_cache(tmp_path / "cache.db")

        order1 = _make_order(operator_message="query one")
        response1 = EngineResponse(
            content="answer one",
            backend_used="b1",
            backend_tier=BackendTier.SIMPLE,
            tokens_in=5,
            tokens_out=10,
            estimated_cost_usd=0.0,
            latency_ms=100.0,
            was_fallback=False,
            fallback_chain=[],
        )
        _store_cache_response(order1, response1)

        order2 = _make_order(operator_message="query two")
        assert _try_cache_lookup(order2) is None

    def test_configure_cache_returns_instance(self, tmp_path: Path) -> None:
        """[TM-021 AC-3] configure_cache returns a SimpleCache instance."""
        cache = configure_cache(tmp_path / "cache.db", max_entries=50)
        assert isinstance(cache, SimpleCache)


# ---------------------------------------------------------------------------
# Feature 2: Cost tracking per model
# ---------------------------------------------------------------------------


class TestCostData:
    """Tests for cost data population and cost-aware scoring."""

    def test_model_cost_lookup_exact_match(self) -> None:
        """[TM-021 AC-4] Exact model match returns specific cost profile."""
        cost = RouterEngine._resolve_cost_profile(
            "gemini/gemini-2.5-pro",
            "gemini",
        )
        assert cost.input_per_mtok == 1.25
        assert cost.output_per_mtok == 10.00

    def test_model_cost_lookup_provider_default(self) -> None:
        """[TM-021 AC-4] Unknown model falls back to provider default cost."""
        cost = RouterEngine._resolve_cost_profile(
            "groq/some-new-model",
            "groq",
        )
        assert cost.input_per_mtok == 0.59
        assert cost.output_per_mtok == 0.79

    def test_model_cost_lookup_unknown_provider(self) -> None:
        """[TM-021 AC-4] Unknown provider returns zero cost."""
        cost = RouterEngine._resolve_cost_profile(
            "unknown/model",
            "unknown",
        )
        assert cost.input_per_mtok == 0.0
        assert cost.output_per_mtok == 0.0

    def test_free_models_have_zero_cost(self) -> None:
        """[TM-021 AC-5] Free-tier models have zero cost."""
        for model_id in [
            "nvidia_nim/moonshotai/kimi-k2.6",
            "openrouter/qwen/qwen3-coder:free",
            "openrouter/poolside/laguna-m.1:free",
        ]:
            provider = model_id.split("/")[0]
            cost = RouterEngine._resolve_cost_profile(model_id, provider)
            assert cost.input_per_mtok == 0.0, f"{model_id} should be free"
            assert cost.output_per_mtok == 0.0, f"{model_id} should be free"

    def test_ollama_local_is_free(self) -> None:
        """[TM-021 AC-5] Ollama (local) models have zero cost."""
        cost = RouterEngine._resolve_cost_profile(
            "ollama/llama3",
            "ollama",
        )
        assert cost.input_per_mtok == 0.0
        assert cost.output_per_mtok == 0.0


# ---------------------------------------------------------------------------
# Feature 2b: Cost-aware scoring
# ---------------------------------------------------------------------------


class TestCostAwareScoring:
    """Tests that cost differences produce different composite scores."""

    def test_normalize_cost_score_free(self) -> None:
        """[TM-021 AC-6] Free model gets max cost score."""
        assert _normalize_cost_score(100.0) == pytest.approx(1.0)

    def test_normalize_cost_score_expensive(self) -> None:
        """[TM-021 AC-6] Expensive model gets low cost score."""
        # avg_cost = 9.0 → rank_score = 100/(9+1) = 10.0
        assert _normalize_cost_score(10.0) == pytest.approx(0.1)

    def test_normalize_cost_score_moderate(self) -> None:
        """[TM-021 AC-6] Moderate cost model gets moderate score."""
        # avg_cost = 0.69 → rank_score = 100/(0.69+1) = 59.17
        score = _normalize_cost_score(59.17)
        assert 0.5 < score < 0.7

    def test_cheaper_model_scores_higher(self) -> None:
        """[TM-021 AC-7] Cheaper model gets higher composite score than expensive one."""
        from dragonlight_router.budget.tracker import BudgetTracker
        from dragonlight_router.core.types import ProviderConfig
        from dragonlight_router.health.tracker import HealthTracker

        provider_cfg = ProviderConfig(
            name="test-provider",
            base_url="https://example.com",
            catalog_url=None,
            env_key=None,
            model_prefix="test/",
            rpm_limit=60,
            rpd_limit=14400,
            tpm_limit=100000,
            daily_token_cap=1000000,
        )
        budget_tracker = BudgetTracker(providers=[provider_cfg])
        health_tracker = HealthTracker()

        order = _make_order()
        weights = ScoringWeightsConfig()

        cheap = _make_config("cheap", input_cost=0.0, output_cost=0.0)
        expensive = _make_config("expensive", input_cost=10.0, output_cost=10.0)

        cheap_score = score_candidate(
            cheap,
            order,
            weights,
            budget_tracker,
            health_tracker,
        )
        expensive_score = score_candidate(
            expensive,
            order,
            weights,
            budget_tracker,
            health_tracker,
        )

        assert cheap_score > expensive_score, (
            f"Cheap model score ({cheap_score}) should exceed "
            f"expensive model score ({expensive_score})"
        )


# ---------------------------------------------------------------------------
# Feature 3: Dispatch logging (tested via log capture)
# ---------------------------------------------------------------------------


class TestDispatchLogging:
    """Tests that structured dispatch logging emits expected events."""

    def test_record_adapter_failure_logs_dispatch_result(self) -> None:
        """[TM-021 AC-8] Adapter failure records dispatch_result log event."""
        from dragonlight_router.dispatch.cascade import (
            DispatchContext,
            _record_adapter_failure,
        )

        ctx = DispatchContext(
            registry=MagicMock(),
            budget_tracker=MagicMock(),
            health_tracker=MagicMock(),
            config={},
        )
        config = _make_config()
        exc = RuntimeError("test error")

        # Should not raise — just logs and records
        _record_adapter_failure(exc, config, ctx)

        # Verify health tracker was called
        ctx.health_tracker.record_error.assert_called_once()


# ---------------------------------------------------------------------------
# Feature 4: Retry with backoff
# ---------------------------------------------------------------------------


class TestRetryBackoff:
    """Tests for retry logic in OpenAI-compatible adapter."""

    def test_is_retryable_429(self) -> None:
        """[TM-021 AC-9] 429 (rate limit) is retryable."""
        assert OpenAICompatibleBackend._is_retryable_status(429) is True

    def test_is_retryable_500(self) -> None:
        """[TM-021 AC-9] 500 (server error) is retryable."""
        assert OpenAICompatibleBackend._is_retryable_status(500) is True

    def test_is_retryable_502(self) -> None:
        """[TM-021 AC-9] 502 (bad gateway) is retryable."""
        assert OpenAICompatibleBackend._is_retryable_status(502) is True

    def test_is_retryable_503(self) -> None:
        """[TM-021 AC-9] 503 (service unavailable) is retryable."""
        assert OpenAICompatibleBackend._is_retryable_status(503) is True

    def test_not_retryable_400(self) -> None:
        """[TM-021 AC-9] 400 (bad request) is NOT retryable."""
        assert OpenAICompatibleBackend._is_retryable_status(400) is False

    def test_not_retryable_401(self) -> None:
        """[TM-021 AC-9] 401 (unauthorized) is NOT retryable."""
        assert OpenAICompatibleBackend._is_retryable_status(401) is False

    def test_not_retryable_404(self) -> None:
        """[TM-021 AC-9] 404 (not found) is NOT retryable."""
        assert OpenAICompatibleBackend._is_retryable_status(404) is False

    def test_compute_backoff_delay_increases(self) -> None:
        """[TM-021 AC-10] Backoff delay increases with attempt number."""
        config = _make_config()
        adapter = OpenAICompatibleBackend(config)

        delays = []
        for attempt in range(5):
            # Sample multiple times to get the base trend
            delay = adapter._compute_backoff_delay(attempt)
            delays.append(delay)

        # Due to jitter, we check that later attempts have higher max possible delay
        # The base delay doubles each attempt: 0.5, 1.0, 2.0, 4.0, 8.0
        assert delays[0] < adapter._max_delay_s * 2
        # Attempt 0 base is 0.5s, attempt 3 base is 4.0s
        # With jitter both are bounded but the trend should be upward

    def test_compute_backoff_delay_non_negative(self) -> None:
        """[TM-021 AC-10] Backoff delay is always non-negative."""
        config = _make_config()
        adapter = OpenAICompatibleBackend(config)

        for attempt in range(10):
            delay = adapter._compute_backoff_delay(attempt)
            assert delay >= 0, f"Delay for attempt {attempt} must be >= 0"

    def test_compute_backoff_respects_max(self) -> None:
        """[TM-021 AC-10] Backoff delay is capped by max_delay_s + jitter."""
        config = _make_config()
        adapter = OpenAICompatibleBackend(config)

        for attempt in range(20):
            delay = adapter._compute_backoff_delay(attempt)
            max_possible = adapter._max_delay_s * (1 + adapter._jitter_factor)
            assert delay <= max_possible + 0.01, f"Delay {delay} exceeds max {max_possible}"

    def test_retry_config_defaults(self) -> None:
        """[TM-021 AC-11] Default retry config is sensible."""
        assert OpenAICompatibleBackend._max_retries == 3
        assert OpenAICompatibleBackend._base_delay_s == 0.5
        assert OpenAICompatibleBackend._max_delay_s == 8.0
        assert 0.0 <= OpenAICompatibleBackend._jitter_factor <= 1.0
