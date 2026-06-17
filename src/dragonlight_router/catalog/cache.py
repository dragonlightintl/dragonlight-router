"""Catalog cache -- file-backed provider model catalog with TTL.

Stores the unified catalog as JSON with a timestamp.
Returns None when stale or missing (triggering a refresh).
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
import time
from pathlib import Path

import structlog

from dragonlight_router.core.errors import StaleCatalogError
from dragonlight_router.core.types import CatalogEntry, Err, Ok, Result

logger = structlog.get_logger()


class CatalogCache:
    """File-backed catalog cache with TTL-based expiration."""

    def __init__(self, cache_path: Path, ttl_hours: int = 24) -> None:
        """File-backed catalog cache with TTL-based expiration."""
        assert isinstance(cache_path, Path), "cache_path must be a Path instance"
        assert ttl_hours >= 0, "ttl_hours must be non-negative"

        self._path = cache_path
        self._ttl_s = ttl_hours * 3600.0

    def get(self) -> Result[dict[str, list[CatalogEntry]], StaleCatalogError]:
        """Load catalog from cache. Returns Err if stale, missing, or corrupt."""
        if self.is_stale():
            return Err(self._make_stale_error("Catalog cache is stale or missing"))

        return self._read_catalog()

    def _read_catalog(self) -> Result[dict[str, list[CatalogEntry]], StaleCatalogError]:
        """Read and deserialize catalog from the cache file."""
        try:
            text = self._path.read_text()
            data = json.loads(text)
        except (json.JSONDecodeError, OSError, KeyError) as exc:
            logger.warning("catalog_cache_read_failed", error=str(exc))
            return Err(self._make_stale_error(f"Failed to read catalog cache: {exc}"))

        catalog = self._deserialize(data.get("catalog", {}))
        self._validate_catalog(catalog)
        return Ok(catalog)

    def _read_cache_age(self) -> int:
        """Read the age of the cache file in seconds. Returns max TTL on failure."""
        if not self._path.exists():
            return int(self._ttl_s)
        try:
            text = self._path.read_text()
            data = json.loads(text)
            timestamp = data.get("timestamp", 0)
            return int(time.time() - timestamp)
        except (json.JSONDecodeError, OSError):
            return int(self._ttl_s)

    def _make_stale_error(self, message: str) -> StaleCatalogError:
        """Construct a StaleCatalogError with current cache age."""
        age = self._read_cache_age()
        return StaleCatalogError(
            provider="unified_cache",
            max_age_seconds=int(self._ttl_s),
            age_seconds=age,
            message=message,
        )

    @staticmethod
    def _validate_catalog(catalog: dict[str, list[CatalogEntry]]) -> None:
        """Assert postconditions on deserialized catalog structure."""
        assert isinstance(catalog, dict), "catalog must be a dict"
        for provider, entries in catalog.items():
            assert isinstance(provider, str), "provider key must be string"
            assert isinstance(entries, list), "entries must be list"

    def set(self, catalog: dict[str, list[CatalogEntry]]) -> None:
        """Atomically write catalog to cache with current timestamp."""
        assert isinstance(catalog, dict), "catalog must be a dict"
        self._validate_catalog(catalog)

        self._path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "timestamp": time.time(),
            "catalog": self._serialize(catalog),
        }

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent),
            prefix=".catalog_cache_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(payload, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.rename(tmp_path, str(self._path))
        except (json.JSONDecodeError, OSError, ValueError):
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

    def is_stale(self) -> bool:
        """Check if the cache is stale (missing, corrupt, or expired)."""
        assert isinstance(self._path, Path), "_path must be a Path instance"
        if not self._path.exists():
            return True

        try:
            text = self._path.read_text()
            data = json.loads(text)
            timestamp = data.get("timestamp", 0)
            age = time.time() - timestamp
            result = age > self._ttl_s
        except (json.JSONDecodeError, OSError):
            result = True

        assert isinstance(result, bool), "is_stale must return a bool"
        return result

    @staticmethod
    def _serialize(catalog: dict[str, list[CatalogEntry]]) -> dict[str, list[dict]]:
        """Convert catalog to JSON-serializable dict."""
        result: dict[str, list[dict]] = {}
        for provider, entries in catalog.items():
            result[provider] = [
                {"model_id": e.model_id, "provider": e.provider, "created": e.created}
                for e in entries
            ]
        return result

    @staticmethod
    def _deserialize(data: dict[str, list[dict]]) -> dict[str, list[CatalogEntry]]:
        """Convert JSON dict back to CatalogEntry objects."""
        result: dict[str, list[CatalogEntry]] = {}
        for provider, entries in data.items():
            result[provider] = [
                CatalogEntry(
                    model_id=e["model_id"],
                    provider=e["provider"],
                    created=e.get("created"),
                )
                for e in entries
            ]
        return result
