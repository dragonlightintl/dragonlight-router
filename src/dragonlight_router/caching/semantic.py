"""Semantic cache -- character n-gram similarity matching.

MVP implementation that stores text with precomputed n-gram sets.
On lookup, computes Jaccard similarity between query n-grams and
stored n-grams. Returns the best match above threshold.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from dragonlight_router.caching.store import get_connection, init_semantic_cache_schema


class SemanticCache:
    """N-gram similarity cache for near-duplicate detection."""

    # DEVIATION DCS-PARAM-001: __init__ takes 5 params (excl. self).
    # Justification: cache config requires db_path plus tuning knobs with defaults;
    # a config dataclass would add indirection. Approved by: architect.
    def __init__(
        self,
        db_path: Path,
        threshold: float = 0.95,
        ngram_size: int = 3,
        max_entries: int = 500,
    ) -> None:
        assert isinstance(db_path, Path), "db_path must be a Path instance"
        assert 0.0 <= threshold <= 1.0, f"threshold must be in [0, 1], got {threshold}"
        assert ngram_size > 0, f"ngram_size must be positive, got {ngram_size}"
        assert max_entries > 0, f"max_entries must be positive, got {max_entries}"
        self._db_path = db_path
        self._threshold = threshold
        self._ngram_size = ngram_size
        self._max_entries = max_entries
        self._conn = get_connection(db_path)
        init_semantic_cache_schema(self._conn)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def get_similar(self, text: str) -> str | None:
        """Find the most similar cached entry above threshold.

        Returns the cached response if similarity >= threshold, else None.
        """
        assert isinstance(text, str), "text must be a string"
        query_ngrams = self._compute_ngrams(text)
        if not query_ngrams:
            return None

        rows = self._conn.execute("SELECT text, response, ngrams FROM semantic_cache").fetchall()

        best_score = 0.0
        best_response: str | None = None

        for _text, response, ngrams_json in rows:
            stored_ngrams = set(json.loads(ngrams_json))
            similarity = self._jaccard(query_ngrams, stored_ngrams)
            if similarity >= self._threshold and similarity > best_score:
                best_score = similarity
                best_response = response

        assert best_score <= 1.0, f"similarity score must be <= 1.0, got {best_score}"
        return best_response

    def put(self, text: str, response: str) -> None:
        """Store text and its response with precomputed n-grams."""
        assert isinstance(text, str), "text must be a string"
        assert isinstance(response, str), "response must be a string"
        ngrams = self._compute_ngrams(text)
        ngrams_json = json.dumps(sorted(ngrams))
        now = time.time()

        self._conn.execute(
            """INSERT INTO semantic_cache (text, response, ngrams, created_at)
               VALUES (?, ?, ?, ?)""",
            (text, response, ngrams_json, now),
        )
        self._conn.commit()

        # Evict old entries
        self._evict_if_needed()

    def _compute_ngrams(self, text: str) -> set[str]:
        """Compute character n-grams from normalized text."""
        assert isinstance(text, str), "text must be a string"
        normalized = text.lower().strip()
        if len(normalized) < self._ngram_size:
            return {normalized} if normalized else set()

        ngrams: set[str] = set()
        for i in range(len(normalized) - self._ngram_size + 1):
            ngrams.add(normalized[i : i + self._ngram_size])
        assert all(len(ng) == self._ngram_size for ng in ngrams), (
            "all n-grams must have correct size"
        )
        return ngrams

    @staticmethod
    def _jaccard(set_a: set[str], set_b: set[str]) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        result = intersection / union if union > 0 else 0.0
        assert 0.0 <= result <= 1.0, f"Jaccard similarity must be in [0, 1], got {result}"
        return result

    def _evict_if_needed(self) -> None:
        """Remove oldest entries if count exceeds max_entries."""
        count = self._conn.execute("SELECT COUNT(*) FROM semantic_cache").fetchone()[0]

        if count <= self._max_entries:
            return

        excess = count - self._max_entries
        self._conn.execute(
            """DELETE FROM semantic_cache WHERE id IN (
                   SELECT id FROM semantic_cache ORDER BY created_at ASC LIMIT ?
               )""",
            (excess,),
        )
        self._conn.commit()
