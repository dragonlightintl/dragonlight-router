"""Health tracker — per-model health scoring and availability.

Uses CircuitBreaker for each tracked model. Provides health scores
based on error count and circuit state. Handles model retirement on 404.

Supports two modes:
- In-memory (db_path=None): Original behavior using dicts.
  Each process instance tracks independently. Suitable for tests/benchmarks.
- SQLite-backed (db_path=Path): Retirement/suspension state is persisted
  to a shared health.db with WAL mode. Solves the multi-process problem
  where factory builds lose retirement state because save_state() is only
  called on server shutdown.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import structlog

from dragonlight_router.core.errors import ModelNotFoundError
from dragonlight_router.core.types import Ok, Result
from dragonlight_router.health.circuit_breaker import CircuitBreaker
from dragonlight_router.health.health_db import HealthDB

logger = structlog.get_logger()


class HealthTracker:
    """Tracks health state for all models.

    When ``db_path`` is provided, retirement and suspension state is
    persisted to a SQLite database so it survives process restarts and
    is visible across concurrent processes. Circuit breakers remain
    in-memory for fast-path scoring.

    When ``db_path`` is None, the original in-memory implementation is
    used for backward compatibility (tests, benchmarks).
    """

    def __init__(
        self,
        error_threshold: int = 3,
        error_window_s: float = 120.0,
        cooldown_s: float = 60.0,
        db_path: Path | None = None,
    ) -> None:
        self._error_threshold = error_threshold
        self._error_window_s = error_window_s
        self._cooldown_s = cooldown_s
        self._db_path = db_path
        self._db: HealthDB | None = None

        if db_path is not None:
            self._db = HealthDB(db_path)

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
        self._suspended: dict[str, float] = {}
        self._suspend_ttl_s: float = 300.0
        self._provider_403_counts: dict[str, int] = defaultdict(int)
        self._suspended_providers: dict[str, float] = {}
        self._provider_suspend_threshold: int = 2
        self._provider_suspend_ttl_s: float = 3600.0

    def score(self, model_id: str) -> Result[float, ModelNotFoundError]:
        """Health score (0-100) for a model.

        - retired → 0
        - circuit_open → 0
        - 3+ errors → 30
        - 1-2 errors → 70
        - 0 errors → 100
        """
        logger.debug("health_score_computed", model_id=model_id)
        if self._is_retired_or_suspended(model_id):
            score = 0.0
            # Assertions for coding standard (>=2 assertions)
            assert 0.0 <= score <= 100.0, f"health score {score} must be between 0 and 100"
            assert isinstance(score, float), f"health score must be float, got {type(score)}"
            return Ok(score)
        breaker = self._breakers[model_id]
        if not breaker.allow_request():
            score = 0.0
            assert 0.0 <= score <= 100.0, f"health score {score} must be between 0 and 100"
            assert isinstance(score, float), f"health score must be float, got {type(score)}"
            return Ok(score)

        error_count = self._error_counts.get(model_id, 0)
        if error_count >= 3:
            score = 30.0
            assert 0.0 <= score <= 100.0, f"health score {score} must be between 0 and 100"
            assert isinstance(score, float), f"health score must be float, got {type(score)}"
            return Ok(score)
        if error_count >= 1:
            score = 70.0
            assert 0.0 <= score <= 100.0, f"health score {score} must be between 0 and 100"
            assert isinstance(score, float), f"health score must be float, got {type(score)}"
            return Ok(score)
        score = 100.0
        assert 0.0 <= score <= 100.0, f"health score {score} must be between 0 and 100"
        assert isinstance(score, float), f"health score must be float, got {type(score)}"
        return Ok(score)

    def record_success(self, model_id: str, latency_ms: float) -> None:
        """Record a successful request — resets errors, updates latency."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        assert latency_ms >= 0, f"latency_ms must be non-negative, got {latency_ms}"
        self._breakers[model_id].record_success()
        self._error_counts[model_id] = 0
        if self._db is not None:
            self._db.reset_error_count(model_id)

        if model_id not in self._avg_latency:
            self._avg_latency[model_id] = latency_ms
        else:
            self._avg_latency[model_id] = (
                self._latency_alpha * latency_ms
                + (1.0 - self._latency_alpha) * self._avg_latency[model_id]
            )

    def record_error(
        self,
        model_id: str,
        *,
        http_status: int | None = None,
    ) -> None:
        """Record a failed request — may trip circuit breaker or retire model.

        HTTP 404 (not found) and 403 (forbidden/unauthorized) at inference
        time trigger immediate retirement (eviction from active catalog).
        All other errors follow normal circuit breaker path.

        When a HealthDB is available, retirement/suspension is persisted
        to SQLite in addition to the in-memory dicts.
        """
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        assert http_status is None or isinstance(http_status, int), (
            f"http_status must be None or int, got {type(http_status)}"
        )
        if http_status == 404:
            self._retire_model(model_id, http_status=http_status)
            if self._db is not None:
                self._db.retire_model(model_id, http_status=http_status)
            return
        if http_status == 403:
            self._suspend_model(model_id)
            if self._db is not None:
                self._db.suspend_model(model_id, ttl_s=self._suspend_ttl_s)
        self._error_counts[model_id] = self._error_counts.get(model_id, 0) + 1
        if self._db is not None:
            self._db.record_error(model_id)
        self._breakers[model_id].record_error()

    def _retire_model(self, model_id: str, *, http_status: int = 404) -> None:
        """Evict a model permanently (404 — model does not exist)."""
        self._retired[model_id] = time.time()
        reason = f"{http_status}_at_inference"
        logger.info("model_retired", model_id=model_id, reason=reason)

    def _suspend_model(self, model_id: str) -> None:
        """Temporarily suspend a model (403 — may be transient auth/budget).

        Also tracks per-provider 403 counts. When a provider accumulates
        enough 403s, the entire provider is suspended to avoid burning
        cascade slots on budget-exhausted providers.
        """
        self._suspended[model_id] = time.time()
        provider = model_id.split("/", 1)[0] if "/" in model_id else model_id
        self._provider_403_counts[provider] += 1
        logger.info(
            "model_suspended",
            model_id=model_id,
            provider=provider,
            provider_403_count=self._provider_403_counts[provider],
            ttl_s=self._suspend_ttl_s,
            reason="403_at_inference",
        )
        if self._provider_403_counts[provider] >= self._provider_suspend_threshold:
            self._suspended_providers[provider] = time.time()
            logger.warning(
                "provider_suspended",
                provider=provider,
                count=self._provider_403_counts[provider],
                ttl_s=self._provider_suspend_ttl_s,
            )

    def _is_retired_or_suspended(self, model_id: str) -> bool:
        """Check both in-memory and DB for retirement/suspension status.

        Checks the in-memory dicts first (fast path), then falls back
        to the DB if available. This ensures that state persisted by
        other processes is visible even if this process hasn't seen
        the retirement/suspension event.
        """
        # In-memory fast path
        if model_id in self._retired:
            return True
        if model_id in self._suspended:
            elapsed = time.time() - self._suspended[model_id]
            if elapsed < self._suspend_ttl_s:
                return True
            del self._suspended[model_id]
        # Provider-level suspension (budget exhaustion)
        provider = model_id.split("/", 1)[0] if "/" in model_id else model_id
        if provider in self._suspended_providers:
            elapsed = time.time() - self._suspended_providers[provider]
            if elapsed < self._provider_suspend_ttl_s:
                return True
            del self._suspended_providers[provider]
            self._provider_403_counts[provider] = 0
        # DB fallback — catches retirements from other processes
        if self._db is not None:
            return self._db.is_unavailable(model_id)
        return False

    def is_retired(self, model_id: str) -> bool:
        """Return True if permanently retired or temporarily suspended."""
        return self._is_retired_or_suspended(model_id)

    def reinstate_model(self, model_id: str) -> None:
        """Restore a retired or suspended model to active status."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        assert isinstance(self._retired, dict), "_retired must be a dict"
        was_unavailable = model_id in self._retired or model_id in self._suspended
        if model_id in self._retired:
            del self._retired[model_id]
        if model_id in self._suspended:
            del self._suspended[model_id]
        self._error_counts[model_id] = 0
        self._breakers[model_id].record_success()
        if self._db is not None:
            self._db.reinstate_model(model_id)
            self._db.reset_error_count(model_id)
            was_unavailable = True  # DB may have had it retired even if in-memory didn't
        if was_unavailable:
            logger.info("model_reinstated", model_id=model_id)

    def get_retired_models(self) -> dict[str, float]:
        """Return model_id → retirement timestamp mapping.

        Merges in-memory and DB state when DB is available.
        """
        result = dict(self._retired)
        if self._db is not None:
            db_retired = self._db.get_retired_models()
            for model_id, ts in db_retired.items():
                if model_id not in result:
                    result[model_id] = ts
        return result

    def is_available(self, model_id: str) -> bool:
        """Check if a model is available (not retired, not suspended, circuit not open)."""
        if self._is_retired_or_suspended(model_id):
            return False
        return self._breakers[model_id].allow_request()

    def get_avg_latency(self, model_id: str) -> float:
        """Return EMA latency in ms, or 0.0 if no data."""
        return self._avg_latency.get(model_id, 0.0)

    def get_error_count(self, model_id: str) -> int:
        """Return current consecutive error count."""
        return self._error_counts.get(model_id, 0)

    def get_state(self) -> dict[str, Any]:
        """Export health tracker state for persistence (HAZ-003/HAZ-012).

        Exports retired models and circuit breaker states so they survive
        process restarts. EMA latency data is intentionally excluded as
        it is stale on restart and will rebuild from live probes.

        When the DB is available, merges DB-persisted state with in-memory
        state for a complete snapshot.
        """
        assert isinstance(self._retired, dict), "_retired must be a dict"
        assert isinstance(self._error_counts, (dict, defaultdict)), "_error_counts must be a dict"

        breaker_states: dict[str, dict[str, Any]] = {}
        for model_id, breaker in self._breakers.items():
            breaker_states[model_id] = breaker.get_state()

        retired = dict(self._retired)
        suspended = dict(self._suspended)

        # Merge DB state so JSON export captures cross-process retirements
        if self._db is not None:
            db_retired = self._db.get_retired_models()
            for model_id, ts in db_retired.items():
                if model_id not in retired:
                    retired[model_id] = ts
            db_suspended = self._db.get_suspended_models()
            for model_id, ts in db_suspended.items():
                if model_id not in suspended:
                    suspended[model_id] = ts

        return {
            "retired": retired,
            "suspended": suspended,
            "error_counts": dict(self._error_counts),
            "breaker_states": breaker_states,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore health tracker state from persistence (HAZ-003/HAZ-012).

        Restores retired models and circuit breaker states. Error counts
        are restored but will be overwritten by live health check probes.

        When the DB is available, retirement/suspension restore from JSON
        is skipped because SQLite IS the persistence layer for those.
        Circuit breaker states are still restored into in-memory breakers.
        """
        assert isinstance(state, dict), "state must be a dict"

        if self._db is not None:
            # SQLite is the source of truth for retirements/suspensions.
            # Still restore breaker states into in-memory breakers for
            # fast-path scoring, and error counts for scoring thresholds.
            logger.info("health_state_restore_retirements_skipped_db_mode")
        else:
            # Restore retired models from JSON (in-memory mode only)
            retired = state.get("retired", {})
            for model_id, timestamp in retired.items():
                self._retired[model_id] = timestamp

            # Restore suspended models (prune expired suspensions on load)
            suspended = state.get("suspended", {})
            now = time.time()
            for model_id, timestamp in suspended.items():
                if now - timestamp < self._suspend_ttl_s:
                    self._suspended[model_id] = timestamp

        # Restore error counts (used for scoring thresholds in both modes)
        error_counts = state.get("error_counts", {})
        for model_id, count in error_counts.items():
            self._error_counts[model_id] = count

        # Restore circuit breaker states (in-memory for fast-path in both modes)
        breaker_states = state.get("breaker_states", {})
        for model_id, breaker_state in breaker_states.items():
            self._breakers[model_id].restore_state(breaker_state)

        logger.info(
            "health_state_restored",
            retired_count=len(state.get("retired", {})),
            breaker_count=len(breaker_states),
            db_mode=self._db is not None,
        )

    def availability_status(self) -> str:
        """Return router-level availability status (HAZ-003 mitigation).

        Assesses overall router availability based on circuit breaker
        states and retired model count:
        - "healthy": majority of tracked models are available
        - "degraded": some models unavailable but at least one is healthy
        - "unavailable": all tracked models are unavailable

        Returns "healthy" if no models are tracked (no state = fresh start).
        """
        assert isinstance(self._breakers, (dict, defaultdict)), "_breakers must be a dict"
        assert isinstance(self._retired, dict), "_retired must be a dict"

        all_retired = self.get_retired_models()
        all_models = set(self._breakers.keys()) | set(all_retired.keys())
        if not all_models:
            return "healthy"

        available_count = 0
        for model_id in all_models:
            if self._is_retired_or_suspended(model_id):
                continue
            if self._breakers[model_id].allow_request():
                available_count += 1

        total = len(all_models)
        if available_count == 0:
            return "unavailable"
        if available_count < total:
            return "degraded"
        return "healthy"
