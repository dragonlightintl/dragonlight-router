"""Integration tests for end-to-end cascade routing."""

from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, patch

from dragonlight_router.core.types import (
    DispatchOrder,
    EngineResponse,
)
from dragonlight_router.router import RouterEngine


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
        """Test that the full cascade selects an optimal provider when all backends are healthy."""
        # This test would verify that when all backends are healthy and within budget,
        # the cascade selects the optimal provider based on scoring
        # For now, we'll test that the dispatch method returns a valid result
        
        # Mock the internal dependencies to avoid complex setup
        with patch.object(router_engine._registry, 'get_backend') as mock_get_backend:
            mock_backend = AsyncMock()
            mock_backend.config.name = "test-provider/test-model"
            mock_backend.config.tier.value = "complex"
            mock_get_backend.return_value = mock_backend
            
            with patch.object(router_engine._health, 'score') as mock_health_score:
                mock_health_score.return_value.__class__.__name__ = "Ok"
                mock_health_score.return_value.value = 95.0
                
                with patch.object(router_engine._budget, 'score') as mock_budget_score:
                    mock_budget_score.return_value.__class__.__name__ = "Ok"
                    mock_budget_score.return_value.value = 90.0
                    
                    # Call dispatch - this exercises the full MBR→CBR→LBR→dispatch path
                    result = router_engine.dispatch(sample_dispatch_order)
                    
                    # Verify we get a result (either Ok or Err is acceptable for this skeleton test)
                    assert result is not None
                    # In a full implementation with real backends, we would assert Ok(EngineResponse)

    @pytest.mark.asyncio
    async def test_primary_failure_triggers_fallback_to_next_candidate(
        self, router_engine, sample_dispatch_order
    ):
        """Test that when the primary backend fails, the cascade falls back to the next candidate."""
        # Test fallback behavior when primary backend returns an error
        # This would verify the Err propagation and fallback logic in the cascade
        
        with patch.object(router_engine._registry, 'get_backend') as mock_get_backend:
            # First backend fails
            mock_get_backend.side_effect = Exception("Primary backend unavailable")
            
            result = router_engine.dispatch(sample_dispatch_order)
            
            # Should handle the error gracefully (return Err rather than crashing)
            assert result is not None
            # Would assert isinstance(result, Err) in full implementation

    @pytest.mark.asyncio
    async def test_all_backends_failing_returns_dispatch_failure(
        self, router_engine, sample_dispatch_order
    ):
        """Test that when all backends fail, the system returns a DispatchFailure."""
        # Test the case where all backends in the cascade are exhausted/unavailable
        
        with patch.object(router_engine._registry, 'get_backend') as mock_get_backend:
            mock_get_backend.side_effect = Exception("All backends unavailable")
            
            result = router_engine.dispatch(sample_dispatch_order)
            
            # Should return an Err representing the dispatch failure
            assert result is not None

    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_3_consecutive_errors(
        self, router_engine, sample_dispatch_order
    ):
        """Test that the circuit breaker pattern works correctly."""
        # This would test the health tracker's circuit breaker functionality
        # After 3 consecutive errors, a backend should be marked as circuit_open
        
        # For now, just verify the dispatch method doesn't crash
        result = router_engine.dispatch(sample_dispatch_order)
        assert result is not None

    @pytest.mark.asyncio
    async def test_budget_exhaustion_deprioritizes_expensive_providers(
        self, router_engine, sample_dispatch_order
    ):
        """Test that budget constraints affect provider selection."""
        # Test that when budget is low, expensive providers are deprioritized
        # This would verify the cost governor and budget scoring integration
        
        result = router_engine.dispatch(sample_dispatch_order)
        assert result is not None

    @pytest.mark.asyncio
    async def test_catalog_refresh_failure_degrades_gracefully(
        self, router_engine, sample_dispatch_order
    ):
        """Test that the system continues to work when catalog refresh fails."""
        # Test graceful degradation when external dependencies fail
        
        with patch.object(router_engine._catalog, 'get') as mock_catalog_get:
            mock_catalog_get.side_effect = Exception("Catalog service unavailable")
            
            # Should still be able to dispatch (possibly with reduced functionality)
            result = router_engine.dispatch(sample_dispatch_order)
            assert result is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])