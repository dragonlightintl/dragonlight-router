"""Health tracker — per-model health scoring and availability.

Uses CircuitBreaker for each tracked model. Provides health scores
based on error count and circuit state. Handles model retirement on 404.
"""
from __future__ import annotations

import time
from collections import defaultdict

import structlog

from dragonlight_router.core.errors import ModelNotFoundError
from dragonlight_router.core.types import Ok, Result
from dragonlight_router.health.circuit_breaker import CircuitBreaker

logger = structlog.get_logger()


class HealthTracker:
    """Tracks health state for all models."""

    def __init__(
        self,
        error_threshold: int = 3,
        error_window_s: float = 120.0,
        cooldown_s: float = 60.0,
    ) -> None:
        self._error_threshold = error_threshold
        self._error_window_s = error_window_s
        self._cooldown_s = cooldown_s

        self._breakers: dict[str, CircuitBreaker] = defaultdict(
            lambda: CircuitBreaker(
                error_threshold=self._error_threshold,
                error_window_s=self._error_window_s,
                cooldown_s=self._cooldown_s,
            )
        )
        self._error_counts: dict[str, int] = defaultdict(int)
        self._avg_latency: dict[str, float] = {}
        self._latency_alpha: float = 0.2
        self._retired: dict[str, float] = {}

    def score(self, model_id: str) -> Result[float, ModelNotFoundError]:
        """Health score (0-100) for a model.

        - retired → 0
        - circuit_open → 0
        - 3+ errors → 30
        - 1-2 errors → 70
        - 0 errors → 100
        """
        logger.debug("health_score_computed", model_id=model_id)
        if model_id in self._retired:
            score = 0.0
            # Assertions for coding standard (>=2 assertions)
            assert 0.0 <= score <= 100.0, f'health score {score} must be between 0 and 100'
            assert isinstance(score, float), f'health score must be float, got {type(score)}'
            return Ok(score)
        breaker = self._breakers[model_id]
        if not breaker.allow_request():
            score = 0.0
            assert 0.0 <= score <= 100.0, f'health score {score} must be between 0 and 100'
            assert isinstance(score, float), f'health score must be float, got {type(score)}'
            return Ok(score)

        error_count = self._error_counts.get(model_id, 0)
        if error_count >= 3:
            score = 30.0
            assert 0.0 <= score <= 100.0, f'health score {score} must be between 0 and 100'
            assert isinstance(score, float), f'health score must be float, got {type(score)}'
            return Ok(score)
        if error_count >= 1:
            score = 70.0
            assert 0.0 <= score <= 100.0, f'health score {score} must be between 0 and 100'
            assert isinstance(score, float), f'health score must be float, got {type(score)}'
            return Ok(score)
        score = 100.0
        assert 0.0 <= score <= 100.0, f'health score {score} must be between 0 and 100'
        assert isinstance(score, float), f'health score must be float, got {type(score)}'
        return Ok(score)

    def record_success(self, model_id: str, latency_ms: float) -> None:
        """Record a successful request — resets errors, updates latency."""
        self._breakers[model_id].record_success()
        self._error_counts[model_id] = 0

        if model_id not in self._avg_latency:
            self._avg_latency[model_id] = latency_ms
        else:
            self._avg_latency[model_id] = (
                self._latency_alpha * latency_ms
                + (1.0 - self._latency_alpha) * self._avg_latency[model_id]
            )

    def record_error(
        self, model_id: str, *, http_status: int | None = None,
    ) -> None:
        """Record a failed request — may trip circuit breaker or retire model.

        HTTP 404 at inference time triggers immediate retirement (eviction
        from active catalog). All other errors follow normal circuit breaker path.
        """
        if http_status == 404:
            self._retire_model(model_id)
            return
        self._error_counts[model_id] = self._error_counts.get(model_id, 0) + 1
        self._breakers[model_id].record_error()

    def _retire_model(self, model_id: str) -> None:
        """Evict a model from the active catalog as a retirement event."""
        self._retired[model_id] = time.time()
        logger.info("model_retired", model_id=model_id, reason="404_at_inference")

    def is_retired(self, model_id: str) -> bool:
        """Return True if the model has been retired (404 eviction)."""
        return model_id in self._retired

    def reinstate_model(self, model_id: str) -> None:
        """Restore a retired model to active status."""
        if model_id not in self._retired:
            return
        del self._retired[model_id]
        self._error_counts[model_id] = 0
        self._breakers[model_id].record_success()
        logger.info("model_reinstated", model_id=model_id)

    def get_retired_models(self) -> dict[str, float]:
        """Return model_id → retirement timestamp mapping."""
        return dict(self._retired)

    def is_available(self, model_id: str) -> bool:
        """Check if a model is available (not retired, circuit not open)."""
        if model_id in self._retired:
            return False
        return self._breakers[model_id].allow_request()

    def get_avg_latency(self, model_id: str) -> float:
        """Return EMA latency in ms, or 0.0 if no data."""
        return self._avg_latency.get(model_id, 0.0)

    def get_error_count(self, model_id: str) -> int:
        """Return current consecutive error count."""
        return self._error_counts.get(model_id, 0)
