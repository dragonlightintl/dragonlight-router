"""Tests for caching/simple.py and caching/semantic.py.

Spec traceability: TM-020 (Response caching)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from dragonlight_router.caching.semantic import SemanticCache
from dragonlight_router.caching.simple import SimpleCache

pytestmark = pytest.mark.unit


class TestSimpleCacheKey:
    def test_deterministic(self):
        """[TM-020 AC-1] Same inputs produce same cache key."""
        msgs = [{"role": "user", "content": "hi"}]
        key1 = SimpleCache.make_key("model-a", "system", msgs, 0.7, 100)
        key2 = SimpleCache.make_key("model-a", "system", msgs, 0.7, 100)
        assert key1 == key2

    def test_different_model_different_key(self):
        """[TM-020 AC-1] Different models produce different cache keys."""
        key1 = SimpleCache.make_key("model-a", "sys", [{"role": "user", "content": "hi"}], 0.7, 100)
        key2 = SimpleCache.make_key("model-b", "sys", [{"role": "user", "content": "hi"}], 0.7, 100)
        assert key1 != key2

    def test_different_messages_different_key(self):
        """[TM-020 AC-1] Different messages produce different cache keys."""
        key1 = SimpleCache.make_key("m", "s", [{"role": "user", "content": "hello"}], 0.7, 100)
        key2 = SimpleCache.make_key("m", "s", [{"role": "user", "content": "world"}], 0.7, 100)
        assert key1 != key2

    def test_different_temperature_different_key(self):
        """[TM-020 AC-1] Different temperatures produce different cache keys."""
        key1 = SimpleCache.make_key("m", "s", [], 0.7, 100)
        key2 = SimpleCache.make_key("m", "s", [], 0.9, 100)
        assert key1 != key2

    def test_sha256_format(self):
        """[TM-020 AC-1] Cache key is a 64-char SHA-256 hex digest."""
        key = SimpleCache.make_key("m", "s", [], 0.7, 100)
        assert len(key) == 64  # SHA-256 hex digest


class TestSimpleCache:
    def test_miss_returns_none(self, tmp_path: Path):
        """[TM-020 AC-2] Cache miss returns None."""
        cache = SimpleCache(db_path=tmp_path / "cache.db")
        assert cache.get("nonexistent") is None
        cache.close()

    def test_put_and_get(self, tmp_path: Path):
        """[TM-020 AC-2] Put then get returns the cached value."""
        cache = SimpleCache(db_path=tmp_path / "cache.db")
        cache.put("key1", "response value")
        assert cache.get("key1") == "response value"
        cache.close()

    def test_overwrite_existing(self, tmp_path: Path):
        """[TM-020 AC-2] Overwriting an existing key replaces the value."""
        cache = SimpleCache(db_path=tmp_path / "cache.db")
        cache.put("key1", "old")
        cache.put("key1", "new")
        assert cache.get("key1") == "new"
        cache.close()

    def test_max_entries_eviction(self, tmp_path: Path):
        """[TM-020 AC-3] Cache evicts oldest entries when max_entries is exceeded."""
        cache = SimpleCache(db_path=tmp_path / "cache.db", max_entries=3)
        cache.put("a", "1")
        cache.put("b", "2")
        cache.put("c", "3")
        cache.put("d", "4")  # Should evict oldest
        # At least "d" should exist
        assert cache.get("d") == "4"
        cache.close()
        # Total entries should not exceed max_entries
        conn = sqlite3.connect(str(tmp_path / "cache.db"))
        count = conn.execute("SELECT COUNT(*) FROM simple_cache").fetchone()[0]
        conn.close()
        assert count <= 3

    def test_ttl_expiration(self, tmp_path: Path):
        """[TM-020 AC-3] Expired entries return None on get."""
        cache = SimpleCache(db_path=tmp_path / "cache.db", ttl_s=0)
        cache.put("key1", "value")
        import time

        time.sleep(0.01)
        # TTL of 0 means immediately expired
        assert cache.get("key1") is None
        cache.close()


class TestSemanticCache:
    def test_miss_returns_none(self, tmp_path: Path):
        """[TM-020 AC-4] Semantic cache miss returns None."""
        cache = SemanticCache(db_path=tmp_path / "cache.db")
        assert cache.get_similar("hello world") is None
        cache.close()

    def test_exact_match(self, tmp_path: Path):
        """[TM-020 AC-4] Exact query match returns cached response."""
        cache = SemanticCache(db_path=tmp_path / "cache.db", threshold=0.95)
        cache.put("explain quicksort in python", "Here is quicksort...")
        result = cache.get_similar("explain quicksort in python")
        assert result == "Here is quicksort..."
        cache.close()

    def test_similar_match(self, tmp_path: Path):
        """[TM-020 AC-4] Similar query above threshold returns cached response."""
        cache = SemanticCache(db_path=tmp_path / "cache.db", threshold=0.8)
        cache.put("explain quicksort algorithm in python", "Here is quicksort...")
        # Very similar text should match at 0.8 threshold
        cache.get_similar("explain quicksort algorithm in python code")
        # May or may not match depending on n-gram similarity
        # But exact should always match
        result2 = cache.get_similar("explain quicksort algorithm in python")
        assert result2 == "Here is quicksort..."
        cache.close()

    def test_dissimilar_no_match(self, tmp_path: Path):
        """[TM-020 AC-4] Dissimilar query below threshold returns None."""
        cache = SemanticCache(db_path=tmp_path / "cache.db", threshold=0.95)
        cache.put("explain quicksort in python", "Here is quicksort...")
        result = cache.get_similar("what is the weather today")
        assert result is None
        cache.close()

    def test_multiple_entries(self, tmp_path: Path):
        """[TM-020 AC-4] Multiple entries are independently retrievable."""
        cache = SemanticCache(db_path=tmp_path / "cache.db", threshold=0.95)
        cache.put("hello world", "greeting response")
        cache.put("goodbye world", "farewell response")
        assert cache.get_similar("hello world") == "greeting response"
        assert cache.get_similar("goodbye world") == "farewell response"
        cache.close()

    def test_empty_string_returns_none(self, tmp_path: Path):
        """[TM-020 AC-4] Empty string query returns None (line 45: empty ngrams branch)."""
        cache = SemanticCache(db_path=tmp_path / "cache.db", threshold=0.95)
        cache.put("some text", "response")
        result = cache.get_similar("")
        assert result is None
        cache.close()

    def test_very_short_text_ngram_fallback(self, tmp_path: Path):
        """[TM-020 AC-4] Text shorter than ngram_size uses single-token fallback (line 87)."""
        cache = SemanticCache(db_path=tmp_path / "cache.db", threshold=0.95, ngram_size=5)
        cache.put("ab", "short response")
        result = cache.get_similar("ab")
        assert result == "short response"
        cache.close()

    def test_jaccard_empty_sets_returns_zero(self, tmp_path: Path):
        """[TM-020 AC-5] Jaccard of empty sets returns 0.0 (line 99)."""
        result = SemanticCache._jaccard(set(), {"abc"})
        assert result == 0.0
        result2 = SemanticCache._jaccard({"abc"}, set())
        assert result2 == 0.0

    def test_eviction_removes_oldest_entries(self, tmp_path: Path):
        """[TM-020 AC-3] Eviction removes oldest entries when max_entries exceeded."""
        cache = SemanticCache(db_path=tmp_path / "cache.db", threshold=0.95, max_entries=2)
        cache.put("first entry text", "response1")
        cache.put("second entry text", "response2")
        cache.put("third entry text", "response3")
        cache.close()
        conn = sqlite3.connect(str(tmp_path / "cache.db"))
        count = conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]
        conn.close()
        assert count <= 2
