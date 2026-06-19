"""Tests for dragonlight_router.catalog.refresher — CatalogRefresher.

Spec traceability: TM-005 (Catalog refresh and provider model fetching)
"""

from __future__ import annotations

import os
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from dragonlight_router.catalog.refresher import CatalogRefresher, CatalogRefreshResult
from dragonlight_router.config.schema import ProviderSchema, RateLimitSchema
from dragonlight_router.core.types import CatalogEntry
from dragonlight_router.result import Ok

pytestmark = pytest.mark.unit


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
    return {"data": [{"id": mid, "created": created} for mid in model_ids]}


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
        """[TM-005 AC-2] refresh() returns Ok containing a CatalogRefreshResult."""
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
        refresh_result = result.unwrap()
        assert isinstance(refresh_result, CatalogRefreshResult)
        assert "groq" in refresh_result.catalog
        assert len(refresh_result.catalog["groq"]) == 2
        assert refresh_result.auth_failures == {}

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
        refresh_result = result.unwrap()
        assert "groq" in refresh_result.catalog
        assert "nvidia" in refresh_result.catalog
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
                raise httpx.HTTPStatusError("503", request=MagicMock(), response=MagicMock())
            return [CatalogEntry(model_id="groq/llama-70b", provider="groq")]

        refresher = CatalogRefresher()

        with patch.object(refresher, "_fetch_provider", side_effect=fake_fetch):
            result = await refresher.refresh([groq, nvidia])

        assert isinstance(result, Ok)
        refresh_result = result.unwrap()
        assert "groq" in refresh_result.catalog
        assert "nvidia" not in refresh_result.catalog

    @pytest.mark.asyncio
    async def test_refresh_empty_providers_returns_empty_catalog(self):
        """[TM-005 AC-2] refresh() with empty providers returns empty result."""
        refresher = CatalogRefresher()
        result = await refresher.refresh([])
        assert isinstance(result, Ok)
        refresh_result = result.unwrap()
        assert refresh_result.catalog == {}
        assert refresh_result.auth_failures == {}

    @pytest.mark.asyncio
    async def test_refresh_all_providers_fail_returns_empty_catalog(self):
        """[TM-005 AC-3] refresh() returns empty catalog when all providers fail."""
        provider = make_provider(name="groq", model_prefix="groq/")

        async def fake_fetch(p):
            raise RuntimeError("connection refused")

        refresher = CatalogRefresher()

        with patch.object(refresher, "_fetch_provider", side_effect=fake_fetch):
            result = await refresher.refresh([provider])

        assert isinstance(result, Ok)
        refresh_result = result.unwrap()
        assert refresh_result.catalog == {}
        assert refresh_result.auth_failures == {}

    @pytest.mark.asyncio
    async def test_refresh_tracks_auth_failures(self):
        """[TM-005 AC-7] refresh() records 401 responses in auth_failures."""
        provider = make_provider(name="groq", model_prefix="groq/")

        mock_response = MagicMock()
        mock_response.status_code = 401

        async def fake_fetch(p):
            raise httpx.HTTPStatusError(
                "401 Unauthorized",
                request=MagicMock(),
                response=mock_response,
            )

        refresher = CatalogRefresher()

        with patch.object(refresher, "_fetch_provider", side_effect=fake_fetch):
            result = await refresher.refresh([provider])

        assert isinstance(result, Ok)
        refresh_result = result.unwrap()
        assert refresh_result.catalog == {}
        assert refresh_result.auth_failures == {"groq": 401}

    @pytest.mark.asyncio
    async def test_refresh_distinguishes_auth_from_transient(self):
        """[TM-005 AC-8] refresh() separates 401 auth failures from 500 transient errors."""
        groq = make_provider(name="groq", model_prefix="groq/")
        nvidia = make_provider(
            name="nvidia",
            base_url="https://integrate.api.nvidia.com/v1",
            model_prefix="nvidia/",
        )

        mock_401_response = MagicMock()
        mock_401_response.status_code = 401

        mock_500_response = MagicMock()
        mock_500_response.status_code = 500

        async def fake_fetch(provider_arg):
            if provider_arg.name == "groq":
                raise httpx.HTTPStatusError(
                    "401 Unauthorized",
                    request=MagicMock(),
                    response=mock_401_response,
                )
            raise httpx.HTTPStatusError(
                "500 Internal Server Error",
                request=MagicMock(),
                response=mock_500_response,
            )

        refresher = CatalogRefresher()

        with patch.object(refresher, "_fetch_provider", side_effect=fake_fetch):
            result = await refresher.refresh([groq, nvidia])

        assert isinstance(result, Ok)
        refresh_result = result.unwrap()
        assert refresh_result.catalog == {}
        # Only the 401 should appear in auth_failures
        assert refresh_result.auth_failures == {"groq": 401}
        assert "nvidia" not in refresh_result.auth_failures

    @pytest.mark.asyncio
    async def test_refresh_tracks_403_as_auth_failure(self):
        """[TM-005 AC-7] refresh() records 403 responses in auth_failures."""
        provider = make_provider(name="groq", model_prefix="groq/")

        mock_response = MagicMock()
        mock_response.status_code = 403

        async def fake_fetch(p):
            raise httpx.HTTPStatusError(
                "403 Forbidden",
                request=MagicMock(),
                response=mock_response,
            )

        refresher = CatalogRefresher()

        with patch.object(refresher, "_fetch_provider", side_effect=fake_fetch):
            result = await refresher.refresh([provider])

        assert isinstance(result, Ok)
        refresh_result = result.unwrap()
        assert refresh_result.auth_failures == {"groq": 403}


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
            ["llama-70b"],
            created=1700000000,
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

        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://custom.groq.com/catalog"

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

        mock_client.get.assert_called_once()
        call_args = mock_client.get.call_args
        assert call_args[0][0] == "https://api.groq.com/openai/v1/models"

    @pytest.mark.asyncio
    async def test_fetch_provider_sends_bearer_auth(self):
        """_fetch_provider() sends Authorization: Bearer header when env_key is set."""
        provider = make_provider(name="groq", model_prefix="groq/")
        refresher = CatalogRefresher()

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"data": []}

        mock_client = AsyncMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client.get = AsyncMock(return_value=mock_response)

        refresher_client = "dragonlight_router.catalog.refresher.httpx.AsyncClient"
        with (
            patch(refresher_client, return_value=mock_client),
            patch.dict(os.environ, {"GROQ_API_KEY": "test-key-123"}),
        ):
            await refresher._fetch_provider(provider)

        call_kwargs = mock_client.get.call_args[1]
        assert call_kwargs["headers"]["Authorization"] == "Bearer test-key-123"

    @pytest.mark.asyncio
    async def test_fetch_provider_sends_x_api_key_for_anthropic(self):
        """_fetch_provider() sends x-api-key header for Anthropic provider."""
        provider = ProviderSchema(
            name="anthropic",
            base_url="https://api.anthropic.com/v1",
            catalog_url=None,
            env_key="ANTHROPIC_API_KEY",
            model_prefix="anthropic/",
            rate_limits=RateLimitSchema(rpm=10, rpd=200, tpm=100000, daily_token_cap=None),
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
        with (
            patch(refresher_client, return_value=mock_client),
            patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-123"}),
        ):
            await refresher._fetch_provider(provider)

        call_kwargs = mock_client.get.call_args[1]
        assert call_kwargs["headers"]["x-api-key"] == "sk-ant-123"
        assert call_kwargs["headers"]["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in call_kwargs["headers"]

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
