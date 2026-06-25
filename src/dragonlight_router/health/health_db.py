"""Health database -- persistent health state via SQLite WAL.

Stores retired models, suspended models, error counts, and circuit
breaker states in a shared SQLite database with WAL mode. Solves the
multi-process problem where factory runs lose retirement/suspension
state because save_state() is only called on server shutdown.

Follows the same SQLite WAL pattern as budget/tracker.py.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

import structlog

logger = structlog.get_logger()


class HealthDB:
    """Persistent health state backed by SQLite WAL.

    Provides durable storage for retirement/suspension decisions so they
    survive process restarts and are visible across concurrent processes
    (factory builds, server, CLI).

    When ``db_path`` is provided, all retirement/suspension state is stored
    in a shared SQLite database (WAL mode) so that multiple router instances
    coordinate through a single source of truth.
    """

    def __init__(self, db_path: Path) -> None:
        assert isinstance(db_path, Path), "db_path must be a Path"
        self._db_path = db_path
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
                CREATE TABLE IF NOT EXISTS retired_models (
                    model_id TEXT PRIMARY KEY,
                    retired_at REAL NOT NULL,
                    reason TEXT NOT NULL,
                    http_status INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS suspended_models (
                    model_id TEXT PRIMARY KEY,
                    suspended_at REAL NOT NULL,
                    ttl_seconds REAL NOT NULL DEFAULT 300.0
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS error_counts (
                    model_id TEXT PRIMARY KEY,
                    count INTEGER NOT NULL DEFAULT 0,
                    last_error_at REAL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS breaker_states (
                    model_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL DEFAULT 'closed',
                    opened_at REAL NOT NULL DEFAULT 0.0,
                    error_timestamps TEXT NOT NULL DEFAULT '[]'
                )
            """)
            conn.commit()
        finally:
            conn.close()
        # SEC: Restrict database file permissions to owner-only (0600).
        os.chmod(self._db_path, 0o600)
        logger.info("health_db_sqlite_initialized", db_path=str(self._db_path))

    def _connect(self) -> sqlite3.Connection:
        """Open a new SQLite connection for the current thread/context."""
        assert self._db_path is not None, "_connect called without db_path"
        conn = sqlite3.connect(str(self._db_path), timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    # ------------------------------------------------------------------
    # Retirement operations
    # ------------------------------------------------------------------

    def retire_model(self, model_id: str, http_status: int = 404) -> None:
        """Permanently retire a model (e.g. 404 at inference time).

        INSERT OR REPLACE into retired_models.
        """
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        assert isinstance(http_status, int), "http_status must be an int"
        now = time.time()
        reason = f"{http_status}_at_inference"
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO retired_models (model_id, retired_at, reason, http_status)"
                " VALUES (?, ?, ?, ?)",
                (model_id, now, reason, http_status),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info("health_db_model_retired", model_id=model_id, reason=reason)

    def is_retired(self, model_id: str) -> bool:
        """Check if a model is permanently retired."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT 1 FROM retired_models WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    def reinstate_model(self, model_id: str) -> None:
        """Remove a model from the retired list, restoring it to active status."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        conn = self._connect()
        try:
            conn.execute(
                "DELETE FROM retired_models WHERE model_id = ?",
                (model_id,),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info("health_db_model_reinstated", model_id=model_id)

    def get_retired_models(self) -> dict[str, float]:
        """Return all retired model_id -> retirement timestamp."""
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT model_id, retired_at FROM retired_models"
            ).fetchall()
            return {row[0]: row[1] for row in rows}
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Suspension operations
    # ------------------------------------------------------------------

    def suspend_model(self, model_id: str, ttl_s: float = 300.0) -> None:
        """Temporarily suspend a model (e.g. 403 — may be transient auth/budget).

        INSERT OR REPLACE into suspended_models.
        """
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        assert isinstance(ttl_s, (int, float)) and ttl_s > 0, "ttl_s must be positive"
        now = time.time()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO suspended_models (model_id, suspended_at, ttl_seconds)"
                " VALUES (?, ?, ?)",
                (model_id, now, ttl_s),
            )
            conn.commit()
        finally:
            conn.close()
        logger.info(
            "health_db_model_suspended",
            model_id=model_id,
            ttl_s=ttl_s,
            reason="403_at_inference",
        )

    def is_suspended(self, model_id: str) -> bool:
        """Check if a model is currently suspended (non-expired).

        Prunes the specific entry if expired.
        """
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        now = time.time()
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT suspended_at, ttl_seconds FROM suspended_models WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                return False
            suspended_at, ttl_seconds = row[0], row[1]
            if now - suspended_at < ttl_seconds:
                return True
            # Expired — prune it
            conn.execute(
                "DELETE FROM suspended_models WHERE model_id = ?",
                (model_id,),
            )
            conn.commit()
            return False
        finally:
            conn.close()

    def is_unavailable(self, model_id: str) -> bool:
        """Check if a model is unavailable (retired OR actively suspended)."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        return self.is_retired(model_id) or self.is_suspended(model_id)

    def get_suspended_models(self) -> dict[str, float]:
        """Return active (non-expired) suspended model_id -> suspension timestamp."""
        now = time.time()
        conn = self._connect()
        try:
            rows = conn.execute(
                "SELECT model_id, suspended_at, ttl_seconds FROM suspended_models"
            ).fetchall()
            active: dict[str, float] = {}
            expired_ids: list[str] = []
            for model_id, suspended_at, ttl_seconds in rows:
                if now - suspended_at < ttl_seconds:
                    active[model_id] = suspended_at
                else:
                    expired_ids.append(model_id)
            # Prune expired entries
            if expired_ids:
                conn.executemany(
                    "DELETE FROM suspended_models WHERE model_id = ?",
                    [(mid,) for mid in expired_ids],
                )
                conn.commit()
            return active
        finally:
            conn.close()

    def prune_expired_suspensions(self) -> int:
        """Delete all expired suspensions. Returns count pruned."""
        now = time.time()
        conn = self._connect()
        try:
            # Expired when: now - suspended_at >= ttl_seconds
            # i.e. suspended_at + ttl_seconds <= now
            cursor = conn.execute(
                "DELETE FROM suspended_models WHERE (suspended_at + ttl_seconds) <= ?",
                (now,),
            )
            conn.commit()
            pruned = cursor.rowcount
            assert isinstance(pruned, int) and pruned >= 0, (
                f"rowcount must be non-negative int, got {pruned}"
            )
            return pruned
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Error count operations
    # ------------------------------------------------------------------

    def record_error(self, model_id: str) -> None:
        """Increment error count for a model."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        now = time.time()
        conn = self._connect()
        try:
            conn.execute(
                "INSERT INTO error_counts (model_id, count, last_error_at)"
                " VALUES (?, 1, ?)"
                " ON CONFLICT(model_id) DO UPDATE SET count = count + 1, last_error_at = ?",
                (model_id, now, now),
            )
            conn.commit()
        finally:
            conn.close()

    def get_error_count(self, model_id: str) -> int:
        """Read error count for a model."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT count FROM error_counts WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def reset_error_count(self, model_id: str) -> None:
        """Reset error count for a model (e.g. on success)."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        conn = self._connect()
        try:
            conn.execute(
                "DELETE FROM error_counts WHERE model_id = ?",
                (model_id,),
            )
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Breaker state operations
    # ------------------------------------------------------------------

    def save_breaker_state(self, model_id: str, state: dict) -> None:
        """Persist circuit breaker state for a model."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        assert isinstance(state, dict), "state must be a dict"
        conn = self._connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO breaker_states"
                " (model_id, state, opened_at, error_timestamps)"
                " VALUES (?, ?, ?, ?)",
                (
                    model_id,
                    state.get("state", "closed"),
                    state.get("opened_at", 0.0),
                    json.dumps(state.get("error_timestamps", [])),
                ),
            )
            conn.commit()
        finally:
            conn.close()

    def load_breaker_state(self, model_id: str) -> dict | None:
        """Load circuit breaker state for a model. Returns None if not found."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty string"
        conn = self._connect()
        try:
            row = conn.execute(
                "SELECT state, opened_at, error_timestamps FROM breaker_states WHERE model_id = ?",
                (model_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "state": row[0],
                "opened_at": row[1],
                "error_timestamps": json.loads(row[2]),
            }
        finally:
            conn.close()
