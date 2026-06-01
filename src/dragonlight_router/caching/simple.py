"""Simple response cache — SHA-256 keyed, TTL-based expiration.

Deterministic cache for exact request matches. Uses a composite key
of model_id + system_prompt + messages + temperature + max_tokens.
"""
from __future__ import annotations
from typing import Any, Optional
import hashlib
import json
import time
from pathlib import Path

from dragonlight_router.caching.store import get_connection, init_simple_cache_schema


class SimpleCache:
    """SHA-256 keyed response cache with TTL and max entries."""

    def __init__(
        self,
        db_path: Path,
        max_entries: int = 1000,
        ttl_s: int = 3600,
    ) -> None:
        self._db_path = db_path
        self._max_entries = max_entries
        self._ttl_s = ttl_s
        self._conn = get_connection(db_path)
        init_simple_cache_schema(self._conn)

    def get(self, key: str) -> Optional[str]:
        """Retrieve a cached response by key. Returns None if missing or expired."""
        row = self._conn.execute(
            "SELECT value, created_at FROM simple_cache WHERE key = ?",
            (key,),
        ).fetchone()

        if row is None:
            return None

        value, created_at = row  # type: tuple[str, float]
        if time.time() - created_at > self._ttl_s:
            # Expired — delete and return None
            self._conn.execute("DELETE FROM simple_cache WHERE key = ?", (key,))
            self._conn.commit()
            return None

        return value

    def put(self, key: str, value: str) -> None:
        """Store a response. Evicts oldest entries if over max_entries."""
        now = time.time()

        self._conn.execute(
            """INSERT OR REPLACE INTO simple_cache (key, value, created_at)
               VALUES (?, ?, ?)""",
            (key, value, now),
        )
        self._conn.commit()

        # Evict if over max
        self._evict_if_needed()

    @staticmethod
    def make_key(
        model_id: str,
        system_prompt: str,
        messages: list[dict[str, Any]],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate a deterministic SHA-256 key from request parameters."""
        payload = json.dumps(
            {
                "model_id": model_id,
                "system_prompt": system_prompt,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def _evict_if_needed(self) -> None:
        """Remove oldest entries if count exceeds max_entries."""
        count = self._conn.execute(
            "SELECT COUNT(*) FROM simple_cache"
        ).fetchone()[0]

        if count > self._max_entries:
            excess = count - self._max_entries
            self._conn.execute(
                """DELETE FROM simple_cache WHERE key IN (
                       SELECT key FROM simple_cache ORDER BY created_at ASC LIMIT ?
                   )""",
                (excess,),
            )
            self._conn.commit()
