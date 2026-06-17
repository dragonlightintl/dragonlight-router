"""RouterEngine — the main orchestrator class.

Provides dual interface:
- select_models(role) for factory-style consumers (returns ranked model IDs)
- dispatch(order) for engine-style consumers (full cascade dispatch)

Wires together: config, budget, health, catalog, matrix, scoring, interleaving.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any

import aiohttp
import structlog

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.catalog.cache import CatalogCache
from dragonlight_router.catalog.refresher import CatalogRefresher
from dragonlight_router.config.loader import load_config
from dragonlight_router.config.schema import RouterConfig
from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import ModelScore, ProviderConfig, RequestOutcome, DispatchOrder, EngineResponse, LatencySLO
from dragonlight_router.dispatch.cascade import dispatch as cascade_dispatch, route
from dragonlight_router.health.tracker import HealthTracker
from dragonlight_router.health.check_loop import HealthCheckLoop
from dragonlight_router.result import Ok, Err, Result
from dragonlight_router.roles.matrix import RoleMatrix
from dragonlight_router.selection.interleave import interleave_providers
from dragonlight_router.selection.scoring import (
    compute_composite_score,
)

logger = structlog.get_logger()

# DEVIATION QA-012: Module-level mutable singleton state.
# Justification: Thread-safe singleton pattern for RouterEngine requires module-level lock
# and instance reference. reset_router() provided for test isolation.
# Approved by: architect. Mitigations: reset_router() for test cleanup.
# Scope: _router_lock, _router_instance only. Expiration: permanent (design pattern).
_router_lock = threading.Lock()
_router_instance: RouterEngine | None = None


class RouterEngine:
    """Central router — serves both daos-engine and factory consumers."""

    def __init__(self, config_path: Path | None = None, **overrides: Any) -> None:
        assert config_path is None or isinstance(config_path, Path), "config_path must be a Path or None"

        self._config = self._load_config(config_path, overrides)
        self._config.state_dir.mkdir(parents=True, exist_ok=True)
        self._init_subsystems()
        self._init_health_check()

        assert isinstance(self._budget, BudgetTracker), "_budget must be a BudgetTracker instance"
        assert isinstance(self._health, HealthTracker), "_health must be a HealthTracker instance"
        assert isinstance(self._config, RouterConfig), "_config must be a RouterConfig instance"

    @staticmethod
    def _load_config(config_path: Path | None, overrides: dict[str, Any]) -> RouterConfig:
        """Load and apply overrides to router configuration."""
        config_result = load_config(config_path)
        if isinstance(config_result, Ok):
            config = config_result.value
        else:
            logger.error("config_load_failed", error=config_result.error.message)
            config = RouterConfig()

        if overrides:
            config_data = config.model_dump()
            config_data.update(overrides)
            config = RouterConfig(**config_data)

        assert isinstance(config, RouterConfig), "config must be a RouterConfig instance"
        return config

    def _init_subsystems(self) -> None:
        """Wire up budget, health, catalog, matrix, and registry subsystems."""
        state_dir = self._config.state_dir
        provider_configs = [
            ProviderConfig(
                name=p.name, base_url=p.base_url, catalog_url=p.catalog_url,
                env_key=p.env_key, model_prefix=p.model_prefix,
                rpm_limit=p.rate_limits.rpm, rpd_limit=p.rate_limits.rpd,
                tpm_limit=p.rate_limits.tpm, daily_token_cap=p.rate_limits.daily_token_cap,
            )
            for p in self._config.providers
        ]
        self._budget = BudgetTracker(providers=provider_configs)
        self._health = HealthTracker()
        self._catalog = CatalogCache(
            cache_path=state_dir / "provider_catalog.json",
            ttl_hours=self._config.catalog_ttl_hours,
        )
        self._refresher = CatalogRefresher()
        self._matrix = RoleMatrix(matrix_path=state_dir / "model_role_matrix.json")
        self._registry = BackendRegistry()
        self._provider_configs = {p.name: p for p in provider_configs}

    def _init_health_check(self) -> None:
        """Set up the background health check loop from registry state."""
        backends_dict = {}
        states_dict = {}
        for name, backend, state in self._registry.all_backends():
            backends_dict[name] = backend
            states_dict[name] = state

        latency_slos_dict = {name: LatencySLO(latency_ms=5000.0) for name in backends_dict}
        self._health_check_loop = HealthCheckLoop(
            backends=backends_dict, states=states_dict,
            latency_slos=latency_slos_dict, interval_s=30.0, timeout_s=10.0,
        )

    async def start_health_check_loop(self) -> None:
        """Start the background health check loop."""
        await self._health_check_loop.start()

    def select_models(
        self,
        role: str,
        *,
        top_n: int = 12,
        exclude_providers: frozenset[str] | None = None,
    ) -> list[str]:
        assert isinstance(role, str) and len(role) > 0, "role must be a non-empty string"
        assert isinstance(top_n, int) and top_n >= 0, "top_n must be a non-negative integer"
        assert exclude_providers is None or isinstance(exclude_providers, frozenset), "exclude_providers must be a frozenset or None"
        """Return ranked model IDs for a role. Factory's primary entry point."""
        self._matrix.reload_if_changed()

        # Get candidates from role matrix
        candidates = self._matrix.get_ranked_models(role)
        if not candidates:
            return []

        # Get live catalog for filtering — refresh if stale
        if self._catalog.is_stale():
            self._refresh_catalog()
        catalog_result = self._catalog.get()
        live_models: set[str] = set()
        fetched_providers: set[str] = set()
        if isinstance(catalog_result, Ok):
            catalog = catalog_result.value
            for provider_name, entries in catalog.items():
                fetched_providers.add(provider_name)
                for entry in entries:
                    live_models.add(entry.model_id)
        # If catalog_result is Err, we proceed with empty live_models/fetched_providers
        # This will cause all candidates to be filtered out, triggering a refresh on next call

        # Filter and score candidates
        filtered = self._filter_by_catalog(
            candidates, exclude_providers, live_models, fetched_providers
        )
        scored = self._score_candidates(filtered)
        return self._build_ranked_list(scored, top_n)

    def _filter_by_catalog(
        self,
        candidates: list[tuple[str, int]],
        exclude_providers: frozenset[str] | None,
        live_models: set[str],
        fetched_providers: set[str],
    ) -> list[tuple[str, int, str | None]]:
        """Filter candidates by provider exclusion and catalog membership.

        Precondition:
            - candidates is a list of (model_id, rank) tuples
            - live_models and fetched_providers are sets of strings
        Postcondition:
            - Returns list of (model_id, rank, provider) tuples that passed filters
        """
        assert isinstance(candidates, list), "candidates must be a list"
        assert all(
            isinstance(item, tuple) and len(item) == 2 for item in candidates
        ), "each candidate must be a (model_id, rank) tuple"
        assert isinstance(live_models, set), "live_models must be a set"
        assert isinstance(fetched_providers, set), "fetched_providers must be a set"

        filtered: list[tuple[str, int, str | None]] = []
        for model_id, rank in candidates:
            provider = self._resolve_provider(model_id)

            # Exclude providers if requested
            if exclude_providers and provider in exclude_providers:
                continue

            # Filter by catalog — only if this model's provider was fetched
            if provider in fetched_providers and model_id not in live_models:
                continue

            filtered.append((model_id, rank, provider))
        return filtered

    def _score_candidates(
        self,
        filtered: list[tuple[str, int, str | None]],
    ) -> list[ModelScore]:
        """Score filtered candidates using budget, health, and rank.

        Precondition:
            - filtered is a list of (model_id, rank, provider) tuples
        Postcondition:
            - Returns list of ModelScore objects sorted by composite score descending
        """
        assert isinstance(filtered, list), "filtered must be a list"
        assert all(
            isinstance(item, tuple) and len(item) == 3 for item in filtered
        ), "each item must be a (model_id, rank, provider) tuple"

        scored: list[ModelScore] = []
        for model_id, rank, provider in filtered:
            # Compute scores
            budget_result = self._budget.score(provider) if provider else Ok(100.0)
            budget_score = budget_result.value if isinstance(budget_result, Ok) else 100.0
            health_result = self._health.score(model_id)
            health_score = health_result.value if isinstance(health_result, Ok) else 100.0

            composite = compute_composite_score(
                rank=rank,
                budget_score=budget_score,
                health_score=health_score,
            )

            scored.append(
                ModelScore(
                    model_id=model_id,
                    provider=provider or "unknown",
                    rank=rank,
                    budget_score=budget_score,
                    health_score=health_score,
                    composite=composite,
                )
            )
        # Sort by composite score descending
        scored.sort(key=lambda m: m.composite, reverse=True)
        return scored

    def _build_ranked_list(
        self, scored: list[ModelScore], top_n: int
    ) -> list[str]:
        """Interleave scored candidates and return top_n model IDs.

        Precondition:
            - scored is a list of ModelScore objects
            - top_n is a non-negative integer
        Postcondition:
            - Returns list of model_id strings of length min(top_n, len(scored))
        """
        assert isinstance(scored, list), "scored must be a list"
        assert all(
            isinstance(item, ModelScore) for item in scored
        ), "each item must be a ModelScore"
        assert isinstance(top_n, int) and top_n >= 0, "top_n must be a non-negative integer"

        # Interleave providers
        interleaved = interleave_providers(
            scored,
            max_consecutive=self._config.max_consecutive_same_provider,
        )
        # Return top_n model IDs
        return [m.model_id for m in interleaved[:top_n]]

    def record_request(self, outcome: RequestOutcome) -> None:
        assert isinstance(outcome, RequestOutcome), "outcome must be a RequestOutcome"
        assert isinstance(outcome.provider, str) and len(outcome.provider) > 0, "outcome.provider must be a non-empty string"
        assert isinstance(outcome.model_id, str) and len(outcome.model_id) > 0, "outcome.model_id must be a non-empty string"
        assert isinstance(outcome.success, bool), "outcome.success must be a boolean"
        assert isinstance(outcome.tokens_used, int) and outcome.tokens_used >= 0, "outcome.tokens_used must be a non-negative integer"
        assert isinstance(outcome.latency_ms, float) and outcome.latency_ms >= 0.0, "outcome.latency_ms must be a non-negative float"
        """Record request outcome for budget/health tracking."""
        if outcome.success:
            self._health.record_success(outcome.model_id, outcome.latency_ms)
        else:
            self._health.record_error(outcome.model_id)
        self._budget.record_request(outcome.provider, outcome.tokens_used)

    def health_snapshot(self) -> dict[str, Any]:
        """Return health state of all tracked models, keyed by provider then model_id."""
        assert isinstance(self._registry, BackendRegistry), "_registry must be a BackendRegistry instance"
        # Build a provider → {model_id → {score, ...}} snapshot from HealthTracker
        snapshot: dict[str, Any] = {}
        # Collect all model_ids that HealthTracker knows about (error counts + latency)
        tracked_models = set(self._health._error_counts.keys()) | set(self._health._avg_latency.keys())
        for model_id in tracked_models:
            # Resolve provider via prefix matching
            provider = self._resolve_provider(model_id) or "unknown"
            if provider not in snapshot:
                snapshot[provider] = {}
            health_result = self._health.score(model_id)
            health_score = health_result.value if isinstance(health_result, Ok) else 0.0
            snapshot[provider][model_id] = {
                "score": health_score,
                "error_count": self._health.get_error_count(model_id),
                "avg_latency_ms": self._health.get_avg_latency(model_id),
                "is_retired": self._health.is_retired(model_id),
            }
        assert isinstance(snapshot, dict), "health_snapshot must return a dict"
        return snapshot
    def budget_snapshot(self) -> dict[str, Any]:
        """Return budget state of all providers."""
        snapshot: dict[str, Any] = {}
        for provider_name in self._provider_configs:
            score_result = self._budget.score(provider_name)
            score_value = score_result.value if isinstance(score_result, Ok) else 0.0
            snapshot[provider_name] = {
                "score": score_value,
                "has_capacity": self._budget.has_capacity(provider_name),
            }

        assert isinstance(snapshot, dict), "budget_snapshot must return a dict"
        for provider_name in snapshot:
            assert isinstance(snapshot[provider_name], dict), f"budget snapshot for {provider_name} must be a dict"
        return snapshot
    def _refresh_catalog(self) -> None:
        """Synchronously trigger async catalog refresh. Stores result in cache."""
        assert isinstance(self._config, RouterConfig), "_config must be a RouterConfig instance"
        assert isinstance(self._refresher, CatalogRefresher), "_refresher must be a CatalogRefresher instance"

        result = self._execute_catalog_refresh()
        if result is None:
            return
        self._catalog.set(result)
        logger.info("catalog_refreshed", providers=list(result.keys()))

    def _execute_catalog_refresh(self) -> dict | None:
        """Run the async catalog refresh and return the catalog dict, or None on failure.

        This is an I/O boundary (network calls to provider APIs via asyncio.run).
        The refresher delegates to multiple provider HTTP endpoints that can raise
        arbitrary exceptions (httpx, aiohttp, provider-specific errors). Graceful
        degradation is required: catalog refresh failure must never crash the router.

        DEVIATION QA-002: except Exception at I/O boundary.
        Justification: Provider adapters raise heterogeneous exception types including
          RuntimeError, OSError, httpx.HTTPError, aiohttp.ClientError, and provider SDK
          errors. Enumerating all possible types is infeasible and brittle.
        Approved by: architect. Mitigations: exception is logged, never silenced.
        Scope: this function only. Expiration: revisit when adapters adopt Result returns.
        """
        try:
            result = asyncio.run(self._refresher.refresh(self._config.providers))
        except Exception as exc:  # noqa: BLE001 — deviation QA-002
            logger.warning("catalog_refresh_failed", error=str(exc))
            return None

        if isinstance(result, Ok):
            return result.value if result.value else None
        if isinstance(result, dict):
            return result if result else None

        error_msg = getattr(getattr(result, "error", None), "message", str(result))
        logger.warning("catalog_refresh_failed", error=error_msg)
        return None

    def _resolve_provider(self, model_id: str) -> str | None:
        """Resolve a model_id to its provider name via prefix matching."""
        assert isinstance(model_id, str), "model_id must be a string"
        assert len(model_id) > 0, "model_id must not be empty"
        for p in self._config.providers:
            if model_id.startswith(p.model_prefix):
                return p.name
        return None

    async def dispatch(self, order: DispatchOrder) -> Result[EngineResponse, Exception]:
        """Execute full cascade dispatch for engine-style consumers."""
        assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"

        config_dict = self._config.model_dump()
        return await cascade_dispatch(
            order=order,
            registry=self._registry,
            budget_tracker=self._budget,
            health_tracker=self._health,
            config=config_dict,
        )


def get_router(
    config_path: str | Path | None = None,
    **overrides: Any,
) -> RouterEngine:
    """Thread-safe singleton. Zero-config works out of the box with canonical defaults."""
    global _router_instance
    if _router_instance is not None:
        return _router_instance
    with _router_lock:
        if _router_instance is not None:
            return _router_instance
        path = Path(config_path) if config_path else None
        _router_instance = RouterEngine(config_path=path, **overrides)
        return _router_instance


def reset_router() -> None:
    """Reset the singleton instance for test isolation (QA-012 mitigation)."""
    global _router_instance
    with _router_lock:
        _router_instance = None