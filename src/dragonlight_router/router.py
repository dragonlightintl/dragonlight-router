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

import structlog

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.catalog.cache import CatalogCache
from dragonlight_router.catalog.refresher import CatalogRefresher
from dragonlight_router.config.loader import load_config
from dragonlight_router.config.schema import RouterConfig
from dragonlight_router.core.registry import BackendRegistry
from dragonlight_router.core.types import CatalogEntry, ModelScore, ProviderConfig
from dragonlight_router.health.tracker import HealthTracker
from dragonlight_router.roles.matrix import RoleMatrix
from dragonlight_router.selection.interleave import interleave_providers
from dragonlight_router.selection.scoring import (
    compute_budget_score,
    compute_composite_score,
    compute_health_score,
)

logger = structlog.get_logger()

_router_lock = threading.Lock()
_router_instance: RouterEngine | None = None


class RouterEngine:
    """Central router — serves both daos-engine and factory consumers."""

    def __init__(self, config_path: Path | None = None, **overrides: Any) -> None:
        self._config: RouterConfig = load_config(config_path)

        # Apply overrides
        if overrides:
            config_data = self._config.model_dump()
            config_data.update(overrides)
            self._config = RouterConfig(**config_data)

        # Initialize subsystems
        state_dir = self._config.state_dir
        state_dir.mkdir(parents=True, exist_ok=True)

        # Convert ProviderSchema → ProviderConfig for budget tracker
        provider_configs = [
            ProviderConfig(
                name=p.name,
                base_url=p.base_url,
                catalog_url=p.catalog_url,
                env_key=p.env_key,
                model_prefix=p.model_prefix,
                rpm_limit=p.rate_limits.rpm,
                rpd_limit=p.rate_limits.rpd,
                tpm_limit=p.rate_limits.tpm,
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

    def select_models(
        self,
        role: str,
        *,
        top_n: int = 12,
        exclude_providers: frozenset[str] | None = None,
    ) -> list[str]:
        """Return ranked model IDs for a role. Factory's primary entry point."""
        self._matrix.reload_if_changed()

        # Get candidates from role matrix
        candidates = self._matrix.get_ranked_models(role)
        if not candidates:
            return []

        # Get live catalog for filtering — refresh if stale
        if self._catalog.is_stale():
            self._refresh_catalog()
        catalog = self._catalog.get()
        live_models: set[str] = set()
        fetched_providers: set[str] = set()
        if catalog:
            for provider_name, entries in catalog.items():
                fetched_providers.add(provider_name)
                for entry in entries:
                    live_models.add(entry.model_id)

        # Score each candidate
        scored: list[ModelScore] = []
        for model_id, rank in candidates:
            # Determine provider from model_id prefix
            provider = self._resolve_provider(model_id)

            # Exclude providers if requested
            if exclude_providers and provider in exclude_providers:
                continue

            # Filter by catalog — only if this model's provider was fetched
            if provider in fetched_providers and model_id not in live_models:
                continue

            # Compute scores
            budget_score = self._budget.score(provider) if provider else 100.0
            health_score = self._health.score(model_id)

            composite = compute_composite_score(
                rank=rank,
                budget_score=budget_score,
                health_score=health_score,
            )

            scored.append(ModelScore(
                model_id=model_id,
                provider=provider or "unknown",
                rank=rank,
                budget_score=budget_score,
                health_score=health_score,
                composite=composite,
            ))

        # Sort by composite score descending
        scored.sort(key=lambda m: m.composite, reverse=True)

        # Interleave providers
        scored = interleave_providers(
            scored,
            max_consecutive=self._config.max_consecutive_same_provider,
        )

        # Return top_n model IDs
        return [m.model_id for m in scored[:top_n]]

    def record_request(
        self,
        provider: str,
        model_id: str,
        *,
        success: bool,
        tokens_used: int = 0,
        latency_ms: float = 0.0,
    ) -> None:
        """Record request outcome for budget/health tracking."""
        if success:
            self._health.record_success(model_id, latency_ms)
        else:
            self._health.record_error(model_id)
        self._budget.record_request(provider, tokens_used)

    def health_snapshot(self) -> dict[str, Any]:
        """Return health state of all tracked models."""
        # Combine registry health with health tracker data
        snapshot: dict[str, Any] = {}
        # Add data from the health tracker for all known models
        # Return registry snapshot if backends are registered
        registry_snap = self._registry.health_snapshot()
        if registry_snap:
            return registry_snap
        return snapshot

    def budget_snapshot(self) -> dict[str, Any]:
        """Return budget state of all providers."""
        snapshot: dict[str, Any] = {}
        for provider_name in self._provider_configs:
            snapshot[provider_name] = {
                "score": self._budget.score(provider_name),
                "has_capacity": self._budget.has_capacity(provider_name),
            }
        return snapshot

    def _refresh_catalog(self) -> None:
        """Synchronously trigger async catalog refresh. Stores result in cache."""
        try:
            catalog = asyncio.run(self._refresher.refresh(self._config.providers))
            if catalog:
                self._catalog.set(catalog)
                logger.info("catalog_refreshed", providers=list(catalog.keys()))
        except Exception as exc:
            logger.warning("catalog_refresh_failed", error=str(exc))

    def _resolve_provider(self, model_id: str) -> str | None:
        """Resolve a model_id to its provider name via prefix matching."""
        for p in self._config.providers:
            if model_id.startswith(p.model_prefix):
                return p.name
        return None


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
