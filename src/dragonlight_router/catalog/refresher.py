"""Catalog refresher -- fetches model lists from providers concurrently.

Calls GET /v1/models on each provider's catalog_url (or base_url + /models).
Returns a unified catalog dict keyed by provider name.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Any

import httpx
import structlog

from dragonlight_router.config.schema import ProviderSchema
from dragonlight_router.core.errors import CatalogRefreshError
from dragonlight_router.core.types import CatalogEntry
from dragonlight_router.core.validation import validate_provider_url
from dragonlight_router.result import Ok, Result

logger = structlog.get_logger()


@dataclass(frozen=True)
class CatalogRefreshResult:
    """Bundles catalog data with auth failure tracking.

    auth_failures maps provider_name → HTTP status code (401 or 403) for
    providers whose API key was rejected. These are tracked separately from
    transient errors (timeouts, 5xx, connection failures) so callers can
    surface key-rotation alerts.
    """

    catalog: dict[str, list[CatalogEntry]]
    auth_failures: dict[str, int]


class CatalogRefresher:
    """Fetches model catalogs from all configured providers."""

    def __init__(self, timeout_s: float = 10.0) -> None:
        assert timeout_s > 0, f"timeout_s must be positive, got {timeout_s}"
        self._timeout = timeout_s

    async def refresh(
        self, providers: list[ProviderSchema]
    ) -> Result[CatalogRefreshResult, CatalogRefreshError]:
        """Refresh catalogs from all providers concurrently.

        Returns partial results -- providers that fail are logged but skipped.
        Auth failures (401/403) are tracked in auth_failures; transient errors
        (timeouts, 5xx, connection failures) are logged as warnings.
        """
        assert isinstance(providers, list), "providers must be a list"
        tasks = [self._fetch_provider(p) for p in providers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        catalog: dict[str, list[CatalogEntry]] = {}
        auth_failures: dict[str, int] = {}
        for provider, result in zip(providers, results, strict=True):
            if isinstance(result, BaseException):
                if isinstance(result, httpx.HTTPStatusError) and result.response.status_code in (
                    401,
                    403,
                ):
                    auth_failures[provider.name] = result.response.status_code
                    logger.warning(
                        "catalog_auth_failed",
                        provider=provider.name,
                        status=result.response.status_code,
                    )
                else:
                    logger.warning(
                        "catalog_refresh_failed",
                        provider=provider.name,
                        error=str(result),
                    )
            else:
                catalog[provider.name] = result

        assert isinstance(catalog, dict), "catalog must be a dict"
        return Ok(CatalogRefreshResult(catalog=catalog, auth_failures=auth_failures))

    async def _fetch_provider(self, provider: ProviderSchema) -> list[CatalogEntry]:
        """Fetch model list from a single provider."""
        assert isinstance(provider, ProviderSchema), "provider must be a ProviderSchema"
        url = provider.catalog_url or f"{provider.base_url}/models"

        # SEC-003: Validate URL before making HTTP request
        validate_provider_url(url)

        headers = self._build_auth_headers(provider)

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        data = response.json()
        models = data.get("data", [])

        entries = self._parse_models(models, provider)
        assert all(isinstance(e, CatalogEntry) for e in entries), "all entries must be CatalogEntry"
        return entries

    @staticmethod
    def _build_auth_headers(provider: ProviderSchema) -> dict[str, str]:
        """Resolve API key from environment and build auth headers."""
        if not provider.env_key:
            return {}
        api_key = os.environ.get(provider.env_key, "")
        if not api_key:
            return {}
        if provider.name == "anthropic":
            return {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
        return {"Authorization": f"Bearer {api_key}"}

    @staticmethod
    def _parse_models(models: list[dict[str, Any]], provider: ProviderSchema) -> list[CatalogEntry]:
        """Parse raw model dicts into CatalogEntry objects."""
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
