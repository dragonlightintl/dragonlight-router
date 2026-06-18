"""Tests for catalog/cache.py and catalog/refresher.py.

Spec traceability: TM-015 (Catalog cache and refresh)
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from dragonlight_router.catalog.cache import CatalogCache
from dragonlight_router.core.types import CatalogEntry
from dragonlight_router.result import Err, Ok


class TestCatalogCache:
    def test_empty_cache_returns_none(self, tmp_path: Path):
        """[TM-015 AC-1] Empty cache returns Err or None."""
        cache = CatalogCache(cache_path=tmp_path / "catalog.json", ttl_hours=24)
        result = cache.get()
        assert isinstance(result, Err) or result.unwrap_or(None) is None

    def test_set_and_get(self, tmp_path: Path):
        """[TM-015 AC-1] Set then get returns the stored catalog."""
        cache = CatalogCache(cache_path=tmp_path / "catalog.json", ttl_hours=24)
        catalog = {
            "groq": [CatalogEntry(model_id="llama-70b", provider="groq")],
            "nvidia": [CatalogEntry(model_id="nemotron-70b", provider="nvidia")],
        }
        cache.set(catalog)
        result = cache.get()
        assert isinstance(result, Ok), f"Expected Ok, got {result}"
        assert "groq" in result.unwrap()
        assert result.unwrap()["groq"][0].model_id == "llama-70b"

    def test_stale_cache_returns_none(self, tmp_path: Path):
        """[TM-015 AC-2] Stale cache (TTL expired) returns Err or None."""
        cache = CatalogCache(cache_path=tmp_path / "catalog.json", ttl_hours=0)
        catalog = {"groq": [CatalogEntry(model_id="x", provider="groq")]}
        cache.set(catalog)
        # TTL of 0 means immediately stale
        time.sleep(0.01)
        assert cache.is_stale() is True
        result = cache.get()
        assert isinstance(result, Err) or result.unwrap_or(None) is None

    def test_fresh_cache_not_stale(self, tmp_path: Path):
        """[TM-015 AC-2] Fresh cache is not stale."""
        cache = CatalogCache(cache_path=tmp_path / "catalog.json", ttl_hours=24)
        catalog = {"groq": [CatalogEntry(model_id="x", provider="groq")]}
        cache.set(catalog)
        assert cache.is_stale() is False

    def test_missing_file_is_stale(self, tmp_path: Path):
        """[TM-015 AC-2] Missing cache file is considered stale."""
        cache = CatalogCache(cache_path=tmp_path / "catalog.json", ttl_hours=24)
        assert cache.is_stale() is True

    def test_corrupt_file_returns_none(self, tmp_path: Path):
        """[TM-015 AC-3] Corrupt JSON file returns Err or None gracefully."""
        path = tmp_path / "catalog.json"
        path.write_text("not valid json {{")
        cache = CatalogCache(cache_path=path, ttl_hours=24)
        result = cache.get()
        assert isinstance(result, Err) or result.unwrap_or(None) is None

    def test_read_catalog_oserror_returns_err(self, tmp_path: Path):
        """[TM-015 AC-3] OSError during read returns Err (lines 46-48)."""
        from unittest.mock import patch

        path = tmp_path / "catalog.json"
        catalog = {"groq": [CatalogEntry(model_id="x", provider="groq")]}
        cache = CatalogCache(cache_path=path, ttl_hours=24)
        cache.set(catalog)

        with patch.object(type(path), "read_text", side_effect=OSError("io error")):
            result = cache._read_catalog()
        assert isinstance(result, Err)

    def test_set_write_failure_cleans_up_tmp(self, tmp_path: Path):
        """[TM-015 AC-3] Write failure during set() raises and cleans up tmp file."""
        from unittest.mock import patch

        path = tmp_path / "catalog.json"
        cache = CatalogCache(cache_path=path, ttl_hours=24)
        catalog = {"groq": [CatalogEntry(model_id="x", provider="groq")]}

        with (
            patch("os.rename", side_effect=OSError("rename failed")),
            pytest.raises(OSError),
        ):
            cache.set(catalog)

        tmp_files = list(path.parent.glob(".catalog_cache_*.tmp"))
        assert len(tmp_files) == 0
