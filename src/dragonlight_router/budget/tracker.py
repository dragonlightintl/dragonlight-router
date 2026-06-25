"""Budget tracker -- per-provider rate limit tracking with sliding windows.

Tracks RPM (requests per minute) via a sliding window of timestamps,
and RPD (requests per day) via a simple counter with daily reset.

Supports two modes:
- In-memory (db_path=None): Original behavior using deque sliding windows.
  Each process instance tracks independently. Suitable for tests/benchmarks.
- SQLite-backed (db_path=Path): All instances coordinate through a shared
  budget.db with WAL mode. Solves the multi-process 429 problem where N
  concurrent factory builds each think they have the full RPM budget.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import sqlite3
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import structlog

from dragonlight_router.core.errors import ProviderNotFoundError
from dragonlight_router.core.types import Ok, ProviderConfig, Result

logger = structlog.get_logger()

# Cleanup runs every N requests or on init.
_CLEANUP_EVERY_N_REQUESTS = 100


def invariant(condition: bool, message: str) -> None:
    """Inline invariant check -- enforced even under python -O."""
    if not condition:
        raise AssertionError(message)


class BudgetTracker:
    """Tracks rate limit budget for all configured providers.

    When ``db_path`` is provided, all rate-limit state is stored in a
    shared SQLite database (WAL mode) so that multiple router instances
    coordinate through a single source of truth.

    When ``db_path`` is None, the original in-memory sliding-window
    implementation is used for backward compatibility (tests, benchmarks).
    """

    def __init__(
        self,
        providers: list[ProviderConfig],
        db_path: Path | None = None,
    ) -> None:
        assert isinstance(providers, list), "providers must be a list"
        assert all(isinstance(p, ProviderConfig) for p in providers), (
            "all providers must be ProviderConfig instances"
        )
        self._providers: dict[str, ProviderConfig] = {p.name: p for p in providers}
        self._db_path = db_path
        self._shared_mode = db_path is not None

        if self._shared_mode:
            self._init_db()
            self._request_count_since_cleanup = 0
        else:
            # In-memory state (original implementation)
            self._rpm_windows: dict[str, deque[float]] = defaultdict(deque)
            self._rpd_counts: dict[str, int] = defaultdict(int)
            self._tpm_windows: dict[str, deque[tuple[float, int]]] = defaultdict(deque)
            self._daily_token_counts: dict[str, int] = defaultdict(int)
            self._day_reset_at: float = self._next_day_boundary()

        # HAZ-002: asyncio.Lock for atomic check-then-record under concurrency
        # Used for in-memory mode; SQLite mode uses DB-level atomicity.
        self._lock = asyncio.Lock()

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
                CREATE TABLE IF NOT EXISTS request_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    provider TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    tokens_used INTEGER NOT NULL DEFAULT 0
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_request_log_provider_ts
                    ON request_log(provider, timestamp)
            """)
            conn.commit()
            # Cleanup stale entries on init
            self._cleanup_old_entries(conn)
            conn.commit()
        finally:
            conn.close()
        # SEC: Restrict database file permissions to owner-only (0600).
        os.chmod(self._db_path, 0o600)
        logger.info("budget_tracker_sqlite_initialized", db_path=str(self._db_path))

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection for the current thread/context."""
        assert self._db_path is not None, "_connect called without db_path"
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _cleanup_old_entries(self, conn: sqlite3.Connection | None = None) -> None:
        """Remove request_log entries older than 24 hours."""
        cutoff = time.time() - 86400
        close_after = False
        if conn is None:
            conn = self._connect()
            close_after = True
        try:
            conn.execute("DELETE FROM request_log WHERE timestamp < ?", (cutoff,))
            conn.commit()
        finally:
            if close_after:
                conn.close()

    def _maybe_cleanup(self) -> None:
        """Trigger cleanup periodically based on request count."""
        self._request_count_since_cleanup += 1
        if self._request_count_since_cleanup >= _CLEANUP_EVERY_N_REQUESTS:
            self._request_count_since_cleanup = 0
            self._cleanup_old_entries()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, provider_name: str) -> Result[float, ProviderNotFoundError]:
        """Budget availability score (0-100) for a provider.

        Considers RPM, RPD, TPM, and daily token cap limits.
        """
        assert isinstance(provider_name, str), "provider_name must be a string"
        provider = self._providers.get(provider_name)
        if provider is None:
            logger.debug("provider_not_found", provider=provider_name)
            return Ok(100.0)

        ratios = self._compute_budget_ratios(provider_name, provider)
        score_value = min(ratios) * 100.0

        assert 0.0 <= score_value <= 100.0, f"score must be in [0, 100], got {score_value}"
        return Ok(score_value)

    def _compute_budget_ratios(self, provider_name: str, provider: ProviderConfig) -> list[float]:
        """Compute budget utilization ratios for all limit dimensions."""
        rpm_ratio = self._rpm_ratio(provider_name, provider)
        rpd_ratio = self._rpd_ratio(provider_name, provider)
        tpm_ratio = self._tpm_ratio(provider_name, provider)
        daily_token_ratio = self._daily_token_ratio(provider_name, provider)

        ratios = [rpm_ratio, rpd_ratio, tpm_ratio, daily_token_ratio]
        assert all(0.0 <= r <= 1.0 for r in ratios), f"all ratios must be in [0, 1], got {ratios}"
        return ratios

    def _rpm_ratio(self, provider_name: str, provider: ProviderConfig) -> float:
        """Compute RPM remaining ratio."""
        rpm_remaining = self._rpm_remaining(provider_name)
        return rpm_remaining / provider.rpm_limit if provider.rpm_limit > 0 else 1.0

    def _rpd_ratio(self, provider_name: str, provider: ProviderConfig) -> float:
        """Compute RPD remaining ratio."""
        if provider.rpd_limit is None or provider.rpd_limit == 0:
            return 1.0
        if self._shared_mode:
            rpd_used = self._db_rpd_count(provider_name)
            rpd_remaining = max(0, provider.rpd_limit - rpd_used)
        else:
            self._maybe_reset_daily()
            rpd_remaining = max(0, provider.rpd_limit - self._rpd_counts[provider_name])
        return rpd_remaining / provider.rpd_limit

    def _tpm_ratio(self, provider_name: str, provider: ProviderConfig) -> float:
        """Compute TPM remaining ratio."""
        if provider.tpm_limit is None or provider.tpm_limit <= 0:
            return 1.0
        tpm_remaining = self._tpm_remaining(provider_name)
        return tpm_remaining / provider.tpm_limit

    def _daily_token_ratio(self, provider_name: str, provider: ProviderConfig) -> float:
        """Compute daily token cap remaining ratio."""
        if provider.daily_token_cap is None or provider.daily_token_cap == 0:
            return 1.0
        if self._shared_mode:
            daily_tokens_used = self._db_daily_tokens(provider_name)
            remaining = max(0, provider.daily_token_cap - daily_tokens_used)
        else:
            self._maybe_reset_daily()
            remaining = max(0, provider.daily_token_cap - self._daily_token_counts[provider_name])
        return remaining / provider.daily_token_cap

    def record_request(self, provider_name: str, tokens_used: int = 0) -> None:
        """Record that a request was dispatched."""
        assert isinstance(provider_name, str), "provider_name must be a string"
        assert isinstance(tokens_used, int) and tokens_used >= 0, (
            f"tokens_used must be a non-negative integer, got {tokens_used}"
        )
        if self._shared_mode:
            self._db_record_request(provider_name, tokens_used)
            self._maybe_cleanup()
        else:
            now = time.time()
            self._rpm_windows[provider_name].append(now)
            self._rpd_counts[provider_name] += 1
            self._tpm_windows[provider_name].append((now, tokens_used))
            self._daily_token_counts[provider_name] += tokens_used

    async def check_and_reserve(self, provider_name: str, estimated_tokens: int = 0) -> bool:
        """Atomically check capacity and reserve budget.

        HAZ-002 mitigation: Prevents concurrent requests from passing budget
        checks simultaneously before either records its spend. Returns True
        if capacity was available and the reservation was recorded, False if
        the provider has no remaining capacity.

        In shared (SQLite) mode, atomicity is provided by a SQLite transaction.
        In in-memory mode, atomicity is provided by the asyncio.Lock.
        """
        assert isinstance(provider_name, str), "provider_name must be a string"
        assert isinstance(estimated_tokens, int) and estimated_tokens >= 0, (
            f"estimated_tokens must be a non-negative integer, got {estimated_tokens}"
        )
        if self._shared_mode:
            return self._db_check_and_reserve(provider_name, estimated_tokens)
        async with self._lock:
            if not self.has_capacity(provider_name):
                return False
            self.record_request(provider_name, estimated_tokens)
            return True

    def has_capacity(self, provider_name: str) -> bool:
        """Quick check: does this provider have RPM, RPD, TPM, and daily token headroom?"""
        assert isinstance(provider_name, str), "provider_name must be a string"
        assert provider_name, "provider_name must be non-empty"
        provider = self._providers.get(provider_name)
        if provider is None:
            return True
        if not self._rpm_remaining(provider_name):
            return False
        if provider.rpd_limit is not None and self._rpd_remaining(provider_name) <= 0:
            return False
        if (
            provider.tpm_limit is not None
            and provider.tpm_limit > 0
            and self._tpm_remaining(provider_name) <= 0
        ):
            return False
        return not (
            provider.daily_token_cap is not None
            and provider.daily_token_cap > 0
            and self._daily_token_remaining(provider_name) <= 0
        )

    def daily_spend_usd(self, provider_name: str, avg_cost_per_token: float = 0.0) -> float:
        """Estimated daily spend for a provider in USD.

        Calculates from tokens_today * avg_cost_per_token.
        """
        assert isinstance(provider_name, str), "provider_name must be a string"
        assert isinstance(avg_cost_per_token, (int, float)) and avg_cost_per_token >= 0, (
            f"avg_cost_per_token must be a non-negative number, got {avg_cost_per_token}"
        )
        if self._shared_mode:
            tokens_today = self._db_daily_tokens(provider_name)
        else:
            self._maybe_reset_daily()
            tokens_today = self._daily_token_counts.get(provider_name, 0)
        spend = tokens_today * avg_cost_per_token
        assert spend >= 0.0, f"daily spend must be non-negative, got {spend}"
        return spend

    def monthly_spend_usd(self, provider_name: str, avg_cost_per_token: float = 0.0) -> float:
        """Estimated monthly spend for a provider in USD.

        Approximates monthly spend as daily_spend * 30.
        """
        assert isinstance(provider_name, str), "provider_name must be a string"
        assert isinstance(avg_cost_per_token, (int, float)) and avg_cost_per_token >= 0, (
            f"avg_cost_per_token must be a non-negative number, got {avg_cost_per_token}"
        )
        daily = self.daily_spend_usd(provider_name, avg_cost_per_token)
        monthly = daily * 30.0
        assert monthly >= 0.0, f"monthly spend must be non-negative, got {monthly}"
        return monthly

    # ------------------------------------------------------------------
    # RPD remaining (shared vs. in-memory)
    # ------------------------------------------------------------------

    def _rpd_remaining(self, provider_name: str) -> int:
        """Remaining RPD in the current day."""
        provider = self._providers.get(provider_name)
        if provider is None or provider.rpd_limit is None:
            return 0
        if self._shared_mode:
            rpd_used = self._db_rpd_count(provider_name)
            remaining = max(0, provider.rpd_limit - rpd_used)
        else:
            self._maybe_reset_daily()
            remaining = max(0, provider.rpd_limit - self._rpd_counts[provider_name])
        assert remaining >= 0, f"Remaining RPD must be non-negative, got {remaining}"
        return remaining

    # ------------------------------------------------------------------
    # RPM remaining (shared vs. in-memory)
    # ------------------------------------------------------------------

    def _rpm_remaining(self, provider_name: str) -> int:
        """Remaining RPM in the current minute window."""
        provider = self._providers.get(provider_name)
        invariant(
            provider is not None,
            f"_rpm_remaining called for unknown provider: {provider_name}",
        )
        assert provider is not None, f"_rpm_remaining called for unknown provider: {provider_name}"
        limit = provider.rpm_limit
        if limit <= 0:
            return 1
        if self._shared_mode:
            rpm_used = self._db_rpm_count(provider_name)
            remaining = max(0, limit - rpm_used)
        else:
            now = time.time()
            cutoff = now - 60.0
            window = self._rpm_windows.get(provider_name, deque())
            while window and window[0] < cutoff:
                window.popleft()
            remaining = max(0, limit - len(window))
        assert remaining >= 0, f"Remaining RPM must be non-negative, got {remaining}"
        return remaining

    # ------------------------------------------------------------------
    # TPM remaining (shared vs. in-memory)
    # ------------------------------------------------------------------

    def _tpm_remaining(self, provider_name: str) -> int:
        """Remaining TPM in the current minute window."""
        provider = self._providers.get(provider_name)
        invariant(
            provider is not None,
            f"_tpm_remaining called for unknown provider: {provider_name}",
        )
        assert provider is not None, f"_tpm_remaining called for unknown provider: {provider_name}"
        limit = provider.tpm_limit
        if limit is None or limit <= 0:
            return 1
        if self._shared_mode:
            tpm_used = self._db_tpm_used(provider_name)
            remaining = max(0, limit - tpm_used)
        else:
            now = time.time()
            cutoff = now - 60.0
            window = self._tpm_windows.get(provider_name, deque())
            # Remove outdated entries
            while window and window[0][0] < cutoff:
                window.popleft()
            # Sum tokens used in the window
            tokens_used = sum(tokens for _, tokens in window)
            remaining = max(0, limit - tokens_used)
        assert remaining >= 0, f"Remaining TPM must be non-negative, got {remaining}"
        return remaining

    # ------------------------------------------------------------------
    # Daily token remaining (shared vs. in-memory)
    # ------------------------------------------------------------------

    def _daily_token_remaining(self, provider_name: str) -> int:
        """Remaining daily token cap for the provider."""
        provider = self._providers.get(provider_name)
        if provider is None or provider.daily_token_cap is None:
            return 0
        if self._shared_mode:
            daily_tokens_used = self._db_daily_tokens(provider_name)
            remaining = max(0, provider.daily_token_cap - daily_tokens_used)
        else:
            self._maybe_reset_daily()
            remaining = max(0, provider.daily_token_cap - self._daily_token_counts[provider_name])
        assert remaining >= 0, f"Remaining daily token cap must be non-negative, got {remaining}"
        return remaining

    # ------------------------------------------------------------------
    # In-memory daily reset (only used in in-memory mode)
    # ------------------------------------------------------------------

    def _maybe_reset_daily(self) -> None:
        """Reset daily counters if a day boundary has passed."""
        assert not self._shared_mode, "_maybe_reset_daily should not be called in shared mode"
        now = time.time()
        if now >= self._day_reset_at:
            self._rpd_counts.clear()
            self._daily_token_counts.clear()
            self._day_reset_at = self._next_day_boundary()
        assert self._day_reset_at > now, (
            f"Day reset time must be in the future, got {self._day_reset_at}"
        )

    @staticmethod
    def _next_day_boundary() -> float:
        """Compute the next UTC midnight timestamp."""
        tomorrow = dt.datetime.now(dt.UTC).replace(
            hour=0,
            minute=0,
            second=0,
            microsecond=0,
        ) + dt.timedelta(days=1)
        result = tomorrow.timestamp()
        assert result > time.time(), f"Next day boundary must be in the future, got {result}"
        return result

    # ------------------------------------------------------------------
    # State export/import (in-memory mode only; shared mode uses SQLite)
    # ------------------------------------------------------------------

    def get_state(self) -> dict[str, Any]:
        """Export serializable budget state for persistence (HAZ-012 mitigation).

        Returns daily counters and reset timestamp. Sliding windows (RPM/TPM)
        are intentionally excluded -- they represent sub-minute state that
        becomes stale immediately on restore.

        In shared mode, daily counters are read from SQLite.
        """
        if self._shared_mode:
            # In shared mode, daily counters live in SQLite.
            # Export them for cost reporting compatibility.
            rpd_counts: dict[str, int] = {}
            daily_token_counts: dict[str, int] = {}
            for provider_name in self._providers:
                rpd_counts[provider_name] = self._db_rpd_count(provider_name)
                daily_token_counts[provider_name] = self._db_daily_tokens(provider_name)
            return {
                "rpd_counts": rpd_counts,
                "daily_token_counts": daily_token_counts,
                "day_reset_at": self._next_day_boundary(),
            }

        assert isinstance(self._rpd_counts, (dict, defaultdict)), "_rpd_counts must be a dict"
        assert isinstance(self._day_reset_at, float), "_day_reset_at must be a float"
        return {
            "rpd_counts": dict(self._rpd_counts),
            "daily_token_counts": dict(self._daily_token_counts),
            "day_reset_at": self._day_reset_at,
        }

    def restore_state(self, state: dict[str, Any]) -> None:
        """Restore budget state from persistence (HAZ-012 mitigation).

        Only restores daily counters if the persisted reset boundary has not
        passed (i.e., we are still within the same UTC day). If the boundary
        has passed, counters start fresh.

        In shared mode, this is a no-op because SQLite IS the persistence.
        """
        if self._shared_mode:
            # SQLite is the source of truth; nothing to restore.
            logger.info("budget_state_restore_skipped_shared_mode")
            return

        assert isinstance(state, dict), "state must be a dict"
        assert "day_reset_at" not in state or isinstance(state["day_reset_at"], (int, float)), (
            "day_reset_at must be numeric if present"
        )
        persisted_reset = state.get("day_reset_at", 0.0)
        now = time.time()

        if now >= persisted_reset:
            # Day boundary passed since save -- start fresh
            logger.info("budget_state_stale_skipping_restore")
            return

        self._day_reset_at = persisted_reset
        rpd = state.get("rpd_counts", {})
        for provider_name, count in rpd.items():
            self._rpd_counts[provider_name] = count
        dtc = state.get("daily_token_counts", {})
        for provider_name, count in dtc.items():
            self._daily_token_counts[provider_name] = count
        logger.info(
            "budget_state_restored",
            providers_restored=len(rpd),
        )

    # ------------------------------------------------------------------
    # SQLite query helpers (shared mode only)
    # ------------------------------------------------------------------

    def _db_record_request(self, provider_name: str, tokens_used: int) -> None:
        """INSERT a request into the shared request_log."""
        now = time.time()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO request_log (provider, timestamp, tokens_used) VALUES (?, ?, ?)",
                (provider_name, now, tokens_used),
            )
            conn.commit()
        finally:
            conn.close()

    def _db_rpm_count(self, provider_name: str) -> int:
        """Count requests in the last 60 seconds for a provider."""
        cutoff = time.time() - 60.0
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE provider = ? AND timestamp > ?",
                (provider_name, cutoff),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def _db_tpm_used(self, provider_name: str) -> int:
        """Sum tokens used in the last 60 seconds for a provider."""
        cutoff = time.time() - 60.0
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens_used), 0)"
                " FROM request_log WHERE provider = ? AND timestamp > ?",
                (provider_name, cutoff),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def _db_rpd_count(self, provider_name: str) -> int:
        """Count requests since the start of the current UTC day."""
        start_of_day = self._start_of_day_utc()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE provider = ? AND timestamp >= ?",
                (provider_name, start_of_day),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def _db_daily_tokens(self, provider_name: str) -> int:
        """Sum tokens used since the start of the current UTC day."""
        start_of_day = self._start_of_day_utc()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens_used), 0)"
                " FROM request_log WHERE provider = ? AND timestamp >= ?",
                (provider_name, start_of_day),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def _db_check_and_reserve(self, provider_name: str, estimated_tokens: int) -> bool:
        """Atomically check capacity and reserve via a SQLite transaction.

        Opens a single connection, checks all limits within an IMMEDIATE
        transaction, and inserts if capacity exists. The IMMEDIATE lock
        prevents other writers from inserting between the check and the
        insert, providing cross-process atomicity.
        """
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Check capacity using the transactional connection
            if not self._has_capacity_in_txn(conn, provider_name):
                conn.execute("ROLLBACK")
                return False
            now = time.time()
            conn.execute(
                "INSERT INTO request_log (provider, timestamp, tokens_used) VALUES (?, ?, ?)",
                (provider_name, now, estimated_tokens),
            )
            conn.execute("COMMIT")
            self._maybe_cleanup()
            return True
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def _has_capacity_in_txn(self, conn: sqlite3.Connection, provider_name: str) -> bool:
        """Check capacity using an existing transactional connection.

        Same logic as has_capacity() but reads from the open transaction
        to ensure consistency within the check-and-reserve sequence.
        """
        provider = self._providers.get(provider_name)
        if provider is None:
            return True

        now = time.time()
        cutoff_minute = now - 60.0

        # RPM check
        if provider.rpm_limit > 0:
            row = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE provider = ? AND timestamp > ?",
                (provider_name, cutoff_minute),
            ).fetchone()
            rpm_used = row[0] if row else 0
            if rpm_used >= provider.rpm_limit:
                return False

        # RPD check
        if provider.rpd_limit is not None and provider.rpd_limit > 0:
            start_of_day = self._start_of_day_utc()
            row = conn.execute(
                "SELECT COUNT(*) FROM request_log WHERE provider = ? AND timestamp >= ?",
                (provider_name, start_of_day),
            ).fetchone()
            rpd_used = row[0] if row else 0
            if rpd_used >= provider.rpd_limit:
                return False

        # TPM check
        if provider.tpm_limit is not None and provider.tpm_limit > 0:
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens_used), 0)"
                " FROM request_log WHERE provider = ? AND timestamp > ?",
                (provider_name, cutoff_minute),
            ).fetchone()
            tpm_used = row[0] if row else 0
            if tpm_used >= provider.tpm_limit:
                return False

        # Daily token cap check
        if provider.daily_token_cap is not None and provider.daily_token_cap > 0:
            start_of_day = self._start_of_day_utc()
            row = conn.execute(
                "SELECT COALESCE(SUM(tokens_used), 0)"
                " FROM request_log WHERE provider = ? AND timestamp >= ?",
                (provider_name, start_of_day),
            ).fetchone()
            daily_tokens = row[0] if row else 0
            if daily_tokens >= provider.daily_token_cap:
                return False

        return True

    @staticmethod
    def _start_of_day_utc() -> float:
        """Return the timestamp of the start of the current UTC day."""
        now_utc = dt.datetime.now(dt.UTC)
        start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        return start.timestamp()
