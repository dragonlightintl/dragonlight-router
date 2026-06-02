"""Catalog refresher — fetches model lists from providers concurrently.

Calls GET /v1/models on each provider's catalog_url (or base_url + /models).
Returns a unified catalog dict keyed by provider name.
"""
from __future__ import annotations

import asyncio

import httpx
import structlog

from dragonlight_router.config.schema import ProviderSchema
from dragonlight_router.core.errors import CatalogRefreshError
from dragonlight_router.core.types import CatalogEntry
from dragonlight_router.result import Ok, Result

logger = structlog.get_logger()


class CatalogRefresher:
    """Fetches model catalogs from all configured providers."""

    def __init__(self, timeout_s: float = 10.0) -> None:
        self._timeout = timeout_s

    async def refresh(
        self, providers: list[ProviderSchema]
    ) -> Result[dict[str, list[CatalogEntry]], CatalogRefreshError]:
        """Refresh catalogs from all providers concurrently.

        Returns partial results — providers that fail are logged but skipped.
        """
        tasks = [self._fetch_provider(p) for p in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        catalog: dict[str, list[CatalogEntry]] = {}
        for provider, result in zip(providers, results, strict=True):
            if isinstance(result, Exception):
                logger.warning(
                    "catalog_refresh_failed",
                    provider=provider.name,
                    error=str(result),
                )
                continue
            catalog[provider.name] = result

        assert not isinstance(result, Exception)  # For type checker
        return Ok(catalog)

    async def _fetch_provider(self, provider: ProviderSchema) -> list[CatalogEntry]:
        """Fetch model list from a single provider."""
        url = provider.catalog_url or f"{provider.base_url}/models"

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url)
            response.raise_for_status()

        data = response.json()
        models = data.get("data", [])

        entries: list[CatalogEntry] = []
        for model in models:
            model_id = model.get("id", "")
            created = model.get("created")
            entries.append(
                CatalogEntry(
                    model_id=f"{provider.model_prefix}{model_id}",
                    provider=provider.name,
                    created=created,
                )
            )

        return entries
