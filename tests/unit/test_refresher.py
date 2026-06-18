"""Tests for dragonlight_router.catalog.refresher — CatalogRefresher.

Spec traceability: TM-005 (Catalog refresh and provider model fetching)
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dragonlight_router.catalog.refresher import CatalogRefresher
from dragonlight_router.config.schema import ProviderSchema, RateLimitSchema
from dragonlight_router.core.types import CatalogEntry
from dragonlight_router.result import Ok


def make_provider(
    name: str = "groq",
    base_url: str = "https://api.groq.com/openai/v1",
    catalog_url: str | None = None,
    model_prefix: str = "groq/",
) -> ProviderSchema:
    """Build a minimal ProviderSchema for tests."""
    return ProviderSchema(
        name=name,
        base_url=base_url,
        catalog_url=catalog_url,
        env_key="GROQ_API_KEY",
        model_prefix=model_prefix,
        rate_limits=RateLimitSchema(rpm=30, rpd=None, tpm=None, daily_token_cap=None),
    )


def make_models_response(model_ids: list[str], created: int | None = None) -> dict:
    """Build a fake /v1/models response payload."""
    return {
        "data": [
            {"id": mid, "created": created}
            for mid in model_ids
        ]
    }


class TestCatalogRefresherInit:
    def test_timeout_stored(self):
        """[TM-005 AC-1] Constructor stores the timeout value."""
        refresher = CatalogRefresher(timeout_s=5.0)
        assert refresher._timeout == 5.0

    def test_default_timeout(self):
        """[TM-005 AC-1] Default timeout is 10 seconds."""
        refresher = CatalogRefresher()
        assert refresher._timeout == 10.0

    def test_invalid_timeout_raises(self):
        """[TM-005 AC-1] Non-positive timeout raises AssertionError."""
        with pytest.raises(AssertionError):
            CatalogRefresher(timeout_s=0)

    def test_negative_timeout_raises(self):
        """[TM-005 AC-1] Negative timeout raises AssertionError."""
        with pytest.raises(AssertionError):
            CatalogRefresher(timeout_s=-1.0)


class TestRefresh:
    @pytest.mark.asyncio
    async def test_refresh_returns_ok_with_catalog(self):
        """[TM-005 AC-2] refresh() returns Ok containing a dict keyed by provider name."""
        provider = make_provider(name="groq", model_prefix="groq/")
        refresher = CatalogRefresher()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = make_models_response(["llama-70b", "mixtral-8x7b"])

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        refresher_client = "dragonlight_router.catalog.refresher.httpx.AsyncClient"
        with patch(refresher_client, return_value=mock_client):
            result = await refresher.refresh([provider])

        assert isinstance(result, Ok)
        catalog = result.unwrap()
        assert "groq" in catalog
        assert len(catalog["groq"]) == 2

    @pytest.mark.asyncio
    async def test_refresh_multiple_providers(self):
        """[TM-005 AC-2] refresh() aggregates results from multiple providers."""
        groq = make_provider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            model_prefix="groq/",
        )
        nvidia = make_provider(
            name="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
            model_prefix="nvidia/",
        )

        call_count = 0

        async def fake_fetch(provider_arg):
            nonlocal call_count
            call_count += 1
            model_id = f"{provider_arg.model_prefix}model-a"
            return [CatalogEntry(model_id=model_id, provider=provider_arg.name)]

        refresher = CatalogRefresher()

        with patch.object(refresher, "_fetch_provider", side_effect=fake_fetch):
            result = await refresher.refresh([groq, nvidia])

        assert isinstance(result, Ok)
        catalog = result.unwrap()
        assert "groq" in catalog
        assert "nvidia" in catalog
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_refresh_partial_results_on_provider_failure(self):
        """[TM-005 AC-3] refresh() returns partial catalog when one provider fails."""
        groq = make_provider(name="groq", model_prefix="groq/")
        nvidia = make_provider(
            name="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
            model_prefix="nvidia/",
        )

        async def fake_fetch(provider_arg):
            if provider_arg.name == "nvidia":
                raise httpx.HTTPStatusError(
                    "503", request=MagicMock(), response=MagicMock()
                )
            return [CatalogEntry(model_id="groq/llama-70b", provider="groq")]

        refresher = CatalogRefresher()

        with patch.object(refresher, "_fetch_provider", side_effect=fake_fetch):
            result = await refresher.refresh([groq, nvidia])

        assert isinstance(result, Ok)
        catalog = result.unwrap()
        assert "groq" in catalog
        assert "nvidia" not in catalog

    @pytest.mark.asyncio
    async def test_refresh_empty_providers_returns_empty_catalog(self):
        """[TM-005 AC-2] refresh() with empty provider list returns Ok with empty dict."""
        refresher = CatalogRefresher()
        result = await refresher.refresh([])
        assert isinstance(result, Ok)
        assert result.unwrap() == {}

    @pytest.mark.asyncio
    async def test_refresh_all_providers_fail_returns_empty_catalog(self):
        """[TM-005 AC-3] refresh() returns empty catalog dict when all providers fail."""
        provider = make_provider(name="groq", model_prefix="groq/")

        async def fake_fetch(p):
            raise RuntimeError("connection refused")

        refresher = CatalogRefresher()

        with patch.object(refresher, "_fetch_provider", side_effect=fake_fetch):
            result = await refresher.refresh([provider])

        assert isinstance(result, Ok)
        assert result.unwrap() == {}


class TestFetchProvider:
    @pytest.mark.asyncio
    async def test_fetch_provider_happy_path(self):
        """[TM-005 AC-4] _fetch_provider() returns a list of CatalogEntry on success."""
        provider = make_provider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            model_prefix="groq/",
        )
        refresher = CatalogRefresher()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = make_models_response(
            ["llama-70b"], created=1700000000,
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        refresher_client = "dragonlight_router.catalog.refresher.httpx.AsyncClient"
        with patch(refresher_client, return_value=mock_client):
            entries = await refresher._fetch_provider(provider)

        assert len(entries) == 1
        assert entries[0].model_id == "groq/llama-70b"
        assert entries[0].provider == "groq"
        assert entries[0].created == 1700000000

    @pytest.mark.asyncio
    async def test_fetch_provider_uses_catalog_url_when_set(self):
        """[TM-005 AC-4] _fetch_provider() uses catalog_url if provided, not base_url/models."""
        provider = make_provider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            catalog_url="https://custom.groq.com/catalog",
            model_prefix="groq/",
        )
        refresher = CatalogRefresher()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        refresher_client = "dragonlight_router.catalog.refresher.httpx.AsyncClient"
        with patch(refresher_client, return_value=mock_client):
            await refresher._fetch_provider(provider)

        mock_client.get.assert_called_once_with("https://custom.groq.com/catalog")

    @pytest.mark.asyncio
    async def test_fetch_provider_falls_back_to_base_url_models(self):
        """[TM-005 AC-4] _fetch_provider() appends /models to base_url when catalog_url is None."""
        provider = make_provider(
            name="groq",
            base_url="https://api.groq.com/openai/v1",
            catalog_url=None,
            model_prefix="groq/",
        )
        refresher = CatalogRefresher()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        refresher_client = "dragonlight_router.catalog.refresher.httpx.AsyncClient"
        with patch(refresher_client, return_value=mock_client):
            await refresher._fetch_provider(provider)

        mock_client.get.assert_called_once_with("https://api.groq.com/openai/v1/models")

    @pytest.mark.asyncio
    async def test_fetch_provider_raises_on_http_error(self):
        """[TM-005 AC-5] _fetch_provider() propagates HTTPStatusError from raise_for_status."""
        provider = make_provider(name="groq", model_prefix="groq/")
        refresher = CatalogRefresher()

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "503 Service Unavailable", request=MagicMock(), response=MagicMock()
        )

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        refresher_client = "dragonlight_router.catalog.refresher.httpx.AsyncClient"
        with (
            patch(refresher_client, return_value=mock_client),
            pytest.raises(httpx.HTTPStatusError),
        ):
            await refresher._fetch_provider(provider)

    @pytest.mark.asyncio
    async def test_fetch_provider_timeout_passed_to_client(self):
        """[TM-005 AC-1] _fetch_provider() passes the configured timeout to httpx.AsyncClient."""
        provider = make_provider(name="groq", model_prefix="groq/")
        refresher = CatalogRefresher(timeout_s=7.5)

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        with patch(
            "dragonlight_router.catalog.refresher.httpx.AsyncClient", return_value=mock_client
        ) as mock_cls:
            await refresher._fetch_provider(provider)

        mock_cls.assert_called_once_with(timeout=7.5)

    @pytest.mark.asyncio
    async def test_fetch_provider_returns_catalog_entries(self):
        """[TM-005 AC-4] _fetch_provider() returns only CatalogEntry instances."""
        provider = make_provider(name="groq", model_prefix="groq/")
        refresher = CatalogRefresher()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = make_models_response(["a", "b", "c"])

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        refresher_client = "dragonlight_router.catalog.refresher.httpx.AsyncClient"
        with patch(refresher_client, return_value=mock_client):
            entries = await refresher._fetch_provider(provider)

        assert all(isinstance(e, CatalogEntry) for e in entries)
        assert len(entries) == 3


class TestParseModels:
    def test_parse_models_with_multiple_models(self):
        """[TM-005 AC-6] _parse_models() converts raw dicts to CatalogEntry list."""
        provider = make_provider(name="groq", model_prefix="groq/")
        models = [
            {"id": "llama-70b", "created": 1700000000},
            {"id": "mixtral-8x7b", "created": 1700000001},
        ]
        entries = CatalogRefresher._parse_models(models, provider)
        assert len(entries) == 2
        assert entries[0].model_id == "groq/llama-70b"
        assert entries[0].provider == "groq"
        assert entries[0].created == 1700000000
        assert entries[1].model_id == "groq/mixtral-8x7b"
        assert entries[1].created == 1700000001

    def test_parse_models_with_empty_list(self):
        """[TM-005 AC-6] _parse_models() returns empty list for empty input."""
        provider = make_provider(name="groq", model_prefix="groq/")
        entries = CatalogRefresher._parse_models([], provider)
        assert entries == []

    def test_parse_models_applies_model_prefix(self):
        """[TM-005 AC-6] _parse_models() prepends provider.model_prefix to model id."""
        provider = make_provider(name="nvidia", model_prefix="nvidia/")
        models = [{"id": "nemotron-70b"}]
        entries = CatalogRefresher._parse_models(models, provider)
        assert entries[0].model_id == "nvidia/nemotron-70b"

    def test_parse_models_created_defaults_to_none(self):
        """[TM-005 AC-6] _parse_models() sets created=None when not present in raw dict."""
        provider = make_provider(name="groq", model_prefix="groq/")
        models = [{"id": "llama-70b"}]
        entries = CatalogRefresher._parse_models(models, provider)
        assert entries[0].created is None

    def test_parse_models_missing_id_uses_empty_string(self):
        """[TM-005 AC-6] _parse_models() uses empty string for model id when 'id' key is absent."""
        provider = make_provider(name="groq", model_prefix="groq/")
        models = [{"created": 123}]
        entries = CatalogRefresher._parse_models(models, provider)
        assert entries[0].model_id == "groq/"

    def test_parse_models_sets_provider_name(self):
        """[TM-005 AC-6] _parse_models() sets the provider field from provider.name."""
        provider = make_provider(name="anthropic", model_prefix="anthropic/")
        models = [{"id": "claude-3-opus"}]
        entries = CatalogRefresher._parse_models(models, provider)
        assert entries[0].provider == "anthropic"

    def test_parse_models_returns_catalog_entry_instances(self):
        """[TM-005 AC-6] _parse_models() returns only CatalogEntry instances."""
        provider = make_provider(name="groq", model_prefix="groq/")
        models = [{"id": "a"}, {"id": "b"}]
        entries = CatalogRefresher._parse_models(models, provider)
        assert all(isinstance(e, CatalogEntry) for e in entries)
