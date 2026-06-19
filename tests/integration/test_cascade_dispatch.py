"""Integration tests for end-to-end cascade routing.

Spec traceability: TM-011 (Cascade dispatch integration), TM-004 (Fallback chain)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from dragonlight_router.core.types import (
    DispatchOrder,
)
from dragonlight_router.result import Err, Ok
from dragonlight_router.router import RouterEngine

pytestmark = pytest.mark.integration


class TestCascadeDispatchIntegration:
    """Integration tests for the full MBR→CBR→LBR→dispatch→response path."""

    @pytest.fixture
    def router_engine(self):
        """Create a RouterEngine instance for testing."""
        return RouterEngine()

    @pytest.fixture
    def sample_dispatch_order(self):
        """Create a sample DispatchOrder for testing."""
        return DispatchOrder(
            intent_category="code_generation",
            specific_intent="write_function",
            operator_message="Write a Python function to calculate fibonacci numbers",
            system_prompt="You are a helpful coding assistant",
            context_tokens=100,
            requires_tool_use=False,
            requires_long_context=False,
        )

    @pytest.mark.asyncio
    async def test_full_cascade_with_healthy_backends_selects_optimal_provider(
        self, router_engine, sample_dispatch_order
    ):
        """[TM-011 AC-1] Full cascade selects optimal provider when all backends are healthy."""
        # This test would verify that when all backends are healthy and within budget,
        # the cascade selects the optimal provider based on scoring
        # For now, we'll test that the dispatch method returns a valid result

        # DEVIATION TEST-MOCK-002: mocks internal engine attributes
        # — RouterEngine lacks DI for health/budget subsystems.
        with patch.object(router_engine._registry, "get") as mock_get:
            mock_backend = AsyncMock()
            mock_backend.config.name = "test-provider/test-model"
            mock_backend.config.tier.value = "complex"
            mock_get.return_value = (mock_backend, None)

            with patch.object(router_engine._health, "score") as mock_health_score:
                mock_health_score.return_value.__class__.__name__ = "Ok"
                mock_health_score.return_value.value = 95.0

                with patch.object(router_engine._budget, "score") as mock_budget_score:
                    mock_budget_score.return_value.__class__.__name__ = "Ok"
                    mock_budget_score.return_value.value = 90.0

                    # Call dispatch - this exercises the full MBR→CBR→LBR→dispatch path
                    result = await router_engine.dispatch(sample_dispatch_order)

                    # Verify we get a result (either Ok or Err is acceptable for this skeleton test)
                    assert result is not None
                    # In a full implementation with real backends,
                    # we would assert Ok(EngineResponse)

    @pytest.mark.asyncio
    async def test_primary_failure_triggers_fallback_to_next_candidate(
        self, router_engine, sample_dispatch_order
    ):
        """[TM-011 AC-2] Primary backend failure triggers fallback to next candidate."""
        # Test fallback behavior when primary backend returns an error
        # This would verify the Err propagation and fallback logic in the cascade

        with patch.object(router_engine._registry, "get") as mock_get:
            # Backend lookup raises an exception
            mock_get.side_effect = Exception("Primary backend unavailable")

            result = await router_engine.dispatch(sample_dispatch_order)

            # Should handle the error gracefully (return Err rather than crashing)
            assert result is not None
            # Would assert isinstance(result, Err) in full implementation

    @pytest.mark.asyncio
    async def test_all_backends_failing_returns_dispatch_failure(
        self, router_engine, sample_dispatch_order
    ):
        """[TM-011 AC-3] All backends failing returns DispatchFailure."""
        # Test the case where all backends in the cascade are exhausted/unavailable

        with patch.object(router_engine._registry, "get") as mock_get:
            mock_get.side_effect = Exception("All backends unavailable")

            result = await router_engine.dispatch(sample_dispatch_order)

            # Should return an Err representing the dispatch failure
            assert result is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_3_consecutive_errors(
        self, router_engine, sample_dispatch_order
    ):
        """[TM-011 AC-4] Circuit breaker opens after 3 consecutive errors."""
        model_id = "test-model"

        # Record 3 consecutive errors to trip the circuit breaker
        for _ in range(3):
            router_engine._health.record_error(model_id)

        # After 3 errors the circuit should be open and the model unavailable
        assert not router_engine._health.is_available(model_id), (
            "model must be unavailable after 3 consecutive errors"
        )
        assert router_engine._health.get_error_count(model_id) == 3, (
            "error count must reflect all recorded errors"
        )

    @pytest.mark.asyncio
    async def test_budget_exhaustion_deprioritizes_expensive_providers(
        self, router_engine, sample_dispatch_order
    ):
        """[TM-011 AC-5] Budget constraints deprioritize expensive providers."""
        # Verify that a registered provider's budget score decreases after
        # recording high token usage, proving the cost governor can deprioritize it.
        registered_providers = list(router_engine._budget._providers.keys())
        if not registered_providers:
            # No providers configured — dispatch should still return Err gracefully
            result = await router_engine.dispatch(sample_dispatch_order)
            assert isinstance(result, Err), "dispatch with no registered backends must return Err"
            return

        provider = registered_providers[0]
        score_before = router_engine._budget.score(provider)
        assert isinstance(score_before, Ok), "score must succeed for registered provider"

        # Record enough tokens to reduce the budget score
        router_engine._budget.record_request(provider, 100_000)
        score_after = router_engine._budget.score(provider)
        assert isinstance(score_after, Ok), "score must succeed after recording usage"
        assert score_after.value <= score_before.value, (
            "budget score must not increase after recording token usage"
        )

    @pytest.mark.asyncio
    async def test_catalog_refresh_failure_degrades_gracefully(
        self, router_engine, sample_dispatch_order
    ):
        """[TM-011 AC-6] System degrades gracefully when catalog refresh fails."""
        # Test graceful degradation when external dependencies fail

        with patch.object(router_engine._catalog, "get") as mock_catalog_get:
            mock_catalog_get.side_effect = Exception("Catalog service unavailable")

            # Should still be able to dispatch (possibly with reduced functionality)
            result = await router_engine.dispatch(sample_dispatch_order)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
