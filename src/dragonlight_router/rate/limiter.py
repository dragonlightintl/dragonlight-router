"""Rate limiter -- cross-process RPM coordination via SQLite WAL.

Provides a sliding-window rate limiter backed by SQLite WAL mode so that
multiple concurrent factory processes coordinate against a single RPM
budget per API key. Rate limits are enforced per API key (not per provider
name) because that's what providers actually rate-limit by. Without this,
N processes each applying their own per-process throttle can exceed the
provider's actual rate limit by Nx.

Follows the same SQLite WAL pattern as health/health_db.py and
budget/tracker.py: WAL mode, busy_timeout=5000, os.chmod(0o600),
short-lived connections (open/close per operation), assertions on inputs.
"""

from __future__ import annotations

import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Sliding window size in seconds.
_WINDOW_SECONDS = 60.0

# Default slot TTL in seconds (crash safety -- slots auto-expire even
# if the acquiring process crashes without calling release()).
_DEFAULT_SLOT_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class RateSlot:
    """Result of a rate slot acquisition attempt."""

    slot_id: str
    granted: bool
    provider: str
    expires_at: float  # unix timestamp
    retry_after_ms: int | None = None  # hint when denied


@dataclass(frozen=True)
class RateUsage:
    """Current RPM usage snapshot for a provider."""

    provider: str
    current_rpm: int
    limit_rpm: int | None
    oldest_slot_age_s: float


class RateLimiter:
    """Cross-process rate limiter backed by SQLite WAL.

    When ``provider_limits`` is provided, each provider is constrained to
    the specified RPM. When ``None`` or when a provider has no configured
    limit, all acquisitions succeed (no rate limiting enforced).

    Multiple processes sharing the same ``db_path`` coordinate through
    SQLite WAL mode, which allows concurrent readers and serialized writers.
    """

    def __init__(
        self,
        db_path: Path,
        provider_limits: dict[str, int] | None = None,
        key_limits: dict[str, int] | None = None,
    ) -> None:
        """Initialize the rate limiter.

        Args:
            db_path: Path to the SQLite database file.
            provider_limits: provider_name -> RPM limit (legacy, used as
                fallback when key_limits is not provided).
            key_limits: api_key_env -> RPM limit, e.g.
                {"NVIDIA_NIM_API_KEY": 60, "GROQ_API_KEY": 28}.
                Keyed by env var name (never the actual key value).
                This is the correct granularity since providers rate-limit
                per API key, not per provider name.
        """
        assert isinstance(db_path, Path), "db_path must be a Path"
        assert provider_limits is None or isinstance(provider_limits, dict), (
            "provider_limits must be a dict or None"
        )
        assert key_limits is None or isinstance(key_limits, dict), (
            "key_limits must be a dict or None"
        )
        self._db_path = db_path
        self._provider_limits = provider_limits or {}
        self._key_limits = key_limits or {}
        self._init_db()

    # ------------------------------------------------------------------
    # SQLite initialization
    # ------------------------------------------------------------------

    def _init_db(self) -> None:
        """Initialize the shared SQLite database with WAL mode."""
        assert self._db_path is not None, "_init_db called without db_path"
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rate_slots (
                    slot_id TEXT PRIMARY KEY,
                    rate_key TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    process_id INTEGER NOT NULL,
                    acquired_at REAL NOT NULL,
                    expires_at REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rate_slots_rate_key
                    ON rate_slots(rate_key)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_rate_slots_expires
                    ON rate_slots(expires_at)
            """)
            conn.commit()
        finally:
            conn.close()
        # SEC: Restrict database file permissions to owner-only (0600).
        os.chmod(self._db_path, 0o600)
        logger.info("rate_limiter_sqlite_initialized", db_path=str(self._db_path))

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection for the current thread/context."""
        assert self._db_path is not None, "_connect called without db_path"
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(
        self, provider: str, model_id: str, *, api_key_env: str | None = None,
    ) -> RateSlot:
        """Try to acquire a rate slot.

        Rate-limit lookup order:
        1. api_key_env in key_limits (correct granularity)
        2. provider in provider_limits (legacy fallback)
        3. No limit found → always grant

        Args:
            provider: Provider name (e.g. "nvidia_nim", "groq").
            model_id: Model identifier for logging/diagnostics.
            api_key_env: Environment variable name for the API key
                (e.g. "NVIDIA_NIM_API_KEY"). Used for per-key rate limiting.
        """
        assert isinstance(provider, str) and provider, "provider must be a non-empty string"
        assert isinstance(model_id, str) and model_id, "model_id must be a non-empty string"

        now = time.time()
        rate_key = api_key_env or provider
        limit = (
            self._key_limits.get(api_key_env) if api_key_env else None
        ) or self._provider_limits.get(provider)

        # No limit configured — always grant.
        if limit is None:
            slot_id = uuid.uuid4().hex
            expires_at = now + _DEFAULT_SLOT_TTL_SECONDS
            conn = self._connect()
            try:
                self._prune_expired(conn, now)
                conn.execute(
                    "INSERT INTO rate_slots (slot_id, rate_key, provider, model_id, process_id, acquired_at, expires_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (slot_id, rate_key, provider, model_id, os.getpid(), now, expires_at),
                )
                conn.commit()
            finally:
                conn.close()
            return RateSlot(
                slot_id=slot_id,
                granted=True,
                provider=provider,
                expires_at=expires_at,
            )

        assert isinstance(limit, int) and limit > 0, (
            f"RPM limit must be a positive integer, got {limit}"
        )

        # Use BEGIN IMMEDIATE for atomicity: prevents concurrent writers
        # from inserting between the count check and the insert.
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")

            # Prune expired slots within the transaction.
            self._prune_expired(conn, now)

            # Count active slots in the sliding window, keyed by rate_key.
            window_start = now - _WINDOW_SECONDS
            row = conn.execute(
                "SELECT COUNT(*) FROM rate_slots WHERE rate_key = ? AND acquired_at > ?",
                (rate_key, window_start),
            ).fetchone()
            current_count = row[0] if row else 0

            if current_count < limit:
                # Under limit — grant the slot.
                slot_id = uuid.uuid4().hex
                expires_at = now + _DEFAULT_SLOT_TTL_SECONDS
                conn.execute(
                    "INSERT INTO rate_slots"
                    " (slot_id, rate_key, provider, model_id, process_id, acquired_at, expires_at)"
                    " VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (slot_id, rate_key, provider, model_id, os.getpid(), now, expires_at),
                )
                conn.execute("COMMIT")
                logger.debug(
                    "rate_slot_acquired",
                    provider=provider,
                    model_id=model_id,
                    slot_id=slot_id,
                    current_rpm=current_count + 1,
                    limit_rpm=limit,
                )
                return RateSlot(
                    slot_id=slot_id,
                    granted=True,
                    provider=provider,
                    expires_at=expires_at,
                )
            else:
                # At/over limit — calculate retry hint.
                conn.execute("COMMIT")
                oldest_row = conn.execute(
                    "SELECT MIN(acquired_at) FROM rate_slots"
                    " WHERE rate_key = ? AND acquired_at > ?",
                    (rate_key, window_start),
                ).fetchone()
                retry_after_ms = 1000  # default 1 second
                if oldest_row and oldest_row[0] is not None:
                    oldest_acquired = oldest_row[0]
                    # The oldest slot will leave the window at oldest_acquired + 60s.
                    wait_s = (oldest_acquired + _WINDOW_SECONDS) - now
                    retry_after_ms = max(100, int(wait_s * 1000))

                logger.debug(
                    "rate_slot_denied",
                    provider=provider,
                    model_id=model_id,
                    current_rpm=current_count,
                    limit_rpm=limit,
                    retry_after_ms=retry_after_ms,
                )
                return RateSlot(
                    slot_id="",
                    granted=False,
                    provider=provider,
                    expires_at=0.0,
                    retry_after_ms=retry_after_ms,
                )
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except Exception:  # noqa: BLE001
                pass
            raise
        finally:
            conn.close()

    def release(self, slot_id: str) -> None:
        """Release a rate slot (call after the LLM API call completes).

        Args:
            slot_id: The slot_id returned by acquire(). Empty strings
                (from denied slots) are silently ignored.
        """
        assert isinstance(slot_id, str), "slot_id must be a string"
        if not slot_id:
            return
        conn = self._connect()
        try:
            conn.execute("DELETE FROM rate_slots WHERE slot_id = ?", (slot_id,))
            conn.commit()
        finally:
            conn.close()
        logger.debug("rate_slot_released", slot_id=slot_id)

    def get_usage(self, provider: str, *, api_key_env: str | None = None) -> RateUsage:
        """Current RPM usage for a rate key (for monitoring/logging).

        Args:
            provider: Provider name.
            api_key_env: If provided, look up usage by API key env var.
        """
        assert isinstance(provider, str) and provider, "provider must be a non-empty string"

        now = time.time()
        window_start = now - _WINDOW_SECONDS
        rate_key = api_key_env or provider
        limit = (
            self._key_limits.get(api_key_env) if api_key_env else None
        ) or self._provider_limits.get(provider)

        conn = self._connect()
        try:
            self._prune_expired(conn, now)

            row = conn.execute(
                "SELECT COUNT(*) FROM rate_slots WHERE rate_key = ? AND acquired_at > ?",
                (rate_key, window_start),
            ).fetchone()
            current_rpm = row[0] if row else 0

            oldest_row = conn.execute(
                "SELECT MIN(acquired_at) FROM rate_slots"
                " WHERE rate_key = ? AND acquired_at > ?",
                (rate_key, window_start),
            ).fetchone()
            oldest_slot_age_s = 0.0
            if oldest_row and oldest_row[0] is not None:
                oldest_slot_age_s = now - oldest_row[0]
        finally:
            conn.close()

        return RateUsage(
            provider=provider,
            current_rpm=current_rpm,
            limit_rpm=limit,
            oldest_slot_age_s=oldest_slot_age_s,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prune_expired(self, conn: sqlite3.Connection, now: float) -> None:
        """Delete all expired rate slots (crash safety)."""
        conn.execute("DELETE FROM rate_slots WHERE expires_at <= ?", (now,))
