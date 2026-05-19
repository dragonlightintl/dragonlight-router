"""Tests for catalog/cache.py and catalog/refresher.py."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from dragonlight_router.catalog.cache import CatalogCache
from dragonlight_router.core.types import CatalogEntry


class TestCatalogCache:
    def test_empty_cache_returns_none(self, tmp_path: Path):
        cache = CatalogCache(cache_path=tmp_path / "catalog.json", ttl_hours=24)
        assert cache.get() is None

    def test_set_and_get(self, tmp_path: Path):
        cache = CatalogCache(cache_path=tmp_path / "catalog.json", ttl_hours=24)
        catalog = {
            "groq": [CatalogEntry(model_id="llama-70b", provider="groq")],
            "nvidia": [CatalogEntry(model_id="nemotron-70b", provider="nvidia")],
        }
        cache.set(catalog)
        result = cache.get()
        assert result is not None
        assert "groq" in result
        assert result["groq"][0].model_id == "llama-70b"

    def test_stale_cache_returns_none(self, tmp_path: Path):
        cache = CatalogCache(cache_path=tmp_path / "catalog.json", ttl_hours=0)
        catalog = {"groq": [CatalogEntry(model_id="x", provider="groq")]}
        cache.set(catalog)
        # TTL of 0 means immediately stale
        time.sleep(0.01)
        assert cache.is_stale() is True
        assert cache.get() is None

    def test_fresh_cache_not_stale(self, tmp_path: Path):
        cache = CatalogCache(cache_path=tmp_path / "catalog.json", ttl_hours=24)
        catalog = {"groq": [CatalogEntry(model_id="x", provider="groq")]}
        cache.set(catalog)
        assert cache.is_stale() is False

    def test_missing_file_is_stale(self, tmp_path: Path):
        cache = CatalogCache(cache_path=tmp_path / "catalog.json", ttl_hours=24)
        assert cache.is_stale() is True

    def test_corrupt_file_returns_none(self, tmp_path: Path):
        path = tmp_path / "catalog.json"
        path.write_text("not valid json {{")
        cache = CatalogCache(cache_path=path, ttl_hours=24)
        assert cache.get() is None
