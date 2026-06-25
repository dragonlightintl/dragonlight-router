"""SQLite profile storage for Model Spectrography.

Persists spectrography profiles and probe results in a SQLite database with WAL
mode for concurrent-read safety. Provides the durable backing store that
complements the YAML config profiles (which are the hot path for the router).

The database stores:
- Profiles: model flavor fingerprints (task, domain, quality_speed scores)
- Probe results: individual probe evaluation records for audit and re-analysis
- Run metadata: spectrography run history for lifecycle tracking

Spec reference: model-spectrography-v0.1.0-spec.md, IBR-FLV-07.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_SPECTROGRAPH,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    ModelSpectrographProfile,
    SpectrographScore,
)
from dragonlight_router.spectrography.analyzer import ProbeResult

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_VERSION: int = 1

_SCHEMA_SQL: str = """\
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS spectrography_runs (
    run_id       TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    completed_at TEXT,
    judge_model  TEXT NOT NULL,
    model_count  INTEGER NOT NULL DEFAULT 0,
    probe_count  INTEGER NOT NULL DEFAULT 0,
    error_count  INTEGER NOT NULL DEFAULT 0,
    status       TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS spectrography_profiles (
    model_id     TEXT NOT NULL,
    dimension    TEXT NOT NULL,
    dim_key      TEXT NOT NULL,
    score        REAL NOT NULL,
    confidence   REAL NOT NULL DEFAULT 0.0,
    sample_count INTEGER NOT NULL DEFAULT 0,
    run_id       TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    PRIMARY KEY (model_id, dimension, dim_key)
);

CREATE TABLE IF NOT EXISTS spectrography_probe_results (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL,
    model_id         TEXT NOT NULL,
    probe_id         TEXT NOT NULL,
    task_type        TEXT NOT NULL,
    domain           TEXT NOT NULL,
    quality_speed    TEXT NOT NULL,
    normalized_score REAL NOT NULL,
    judge_scores     TEXT,
    is_self_eval     INTEGER NOT NULL DEFAULT 0,
    error            TEXT,
    created_at       TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_profiles_model
    ON spectrography_profiles(model_id);

CREATE INDEX IF NOT EXISTS idx_profiles_run
    ON spectrography_profiles(run_id);

CREATE INDEX IF NOT EXISTS idx_probe_results_run
    ON spectrography_probe_results(run_id);

CREATE INDEX IF NOT EXISTS idx_probe_results_model
    ON spectrography_probe_results(model_id);

CREATE INDEX IF NOT EXISTS idx_probe_results_model_probe
    ON spectrography_probe_results(model_id, probe_id);
"""


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------


class SpectrographyStore:
    """SQLite-backed store for spectrography profiles and results.

    Uses WAL journal mode for concurrent read access during router operation.
    """

    def __init__(self, db_path: Path) -> None:
        assert isinstance(db_path, Path), "db_path must be a Path instance"
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        """Open the database connection, create schema if needed."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA_SQL)
        self._ensure_schema_version()
        logger.info(
            "spectrography_store_opened",
            path=str(self._db_path),
        )

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def _ensure_schema_version(self) -> None:
        """Insert schema version if missing."""
        assert self._conn is not None, "database not open"
        cursor = self._conn.execute("SELECT COUNT(*) FROM schema_version")
        count = cursor.fetchone()[0]
        if count == 0:
            self._conn.execute(
                "INSERT INTO schema_version (version) VALUES (?)",
                (_SCHEMA_VERSION,),
            )
            self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        """Return the active connection. Raises if not open."""
        assert self._conn is not None, "database not open -- call open() first"
        return self._conn

    # -----------------------------------------------------------------------
    # Run tracking
    # -----------------------------------------------------------------------

    def record_run_start(
        self,
        run_id: str,
        judge_model: str,
        model_count: int,
        probe_count: int,
    ) -> None:
        """Record the start of a spectrography run."""
        assert isinstance(run_id, str) and run_id, "run_id must be non-empty"
        assert isinstance(judge_model, str) and judge_model, "judge_model must be non-empty"

        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """INSERT OR REPLACE INTO spectrography_runs
               (run_id, started_at, judge_model, model_count, probe_count, status)
               VALUES (?, ?, ?, ?, ?, 'running')""",
            (run_id, now, judge_model, model_count, probe_count),
        )
        self.conn.commit()

    def record_run_complete(
        self,
        run_id: str,
        error_count: int,
        status: str = "complete",
    ) -> None:
        """Record the completion of a spectrography run."""
        assert isinstance(run_id, str) and run_id, "run_id must be non-empty"
        assert status in ("complete", "partial", "failed"), f"invalid status: {status}"

        now = datetime.now(UTC).isoformat()
        self.conn.execute(
            """UPDATE spectrography_runs
               SET completed_at = ?, error_count = ?, status = ?
               WHERE run_id = ?""",
            (now, error_count, status, run_id),
        )
        self.conn.commit()

    # -----------------------------------------------------------------------
    # Probe result storage
    # -----------------------------------------------------------------------

    def store_probe_result(self, result: ProbeResult, run_id: str) -> None:
        """Store a single probe result."""
        assert isinstance(result, ProbeResult), "result must be a ProbeResult"
        assert isinstance(run_id, str) and run_id, "run_id must be non-empty"

        now = datetime.now(UTC).isoformat()
        judge_scores_json = (
            json.dumps(result.judge_scores) if result.judge_scores is not None else None
        )

        self.conn.execute(
            """INSERT INTO spectrography_probe_results
               (run_id, model_id, probe_id, task_type, domain, quality_speed,
                normalized_score, judge_scores, is_self_eval, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                result.model_id,
                result.probe_id,
                result.task_type,
                result.domain,
                result.quality_speed,
                result.normalized_score,
                judge_scores_json,
                1 if result.is_self_eval else 0,
                result.error,
                now,
            ),
        )
        self.conn.commit()

    def store_probe_results_batch(
        self,
        results: list[ProbeResult],
        run_id: str,
    ) -> None:
        """Store multiple probe results in a single transaction."""
        assert isinstance(results, list), "results must be a list"
        assert isinstance(run_id, str) and run_id, "run_id must be non-empty"

        now = datetime.now(UTC).isoformat()
        rows = []
        for r in results:
            judge_json = json.dumps(r.judge_scores) if r.judge_scores is not None else None
            rows.append((
                run_id,
                r.model_id,
                r.probe_id,
                r.task_type,
                r.domain,
                r.quality_speed,
                r.normalized_score,
                judge_json,
                1 if r.is_self_eval else 0,
                r.error,
                now,
            ))

        self.conn.executemany(
            """INSERT INTO spectrography_probe_results
               (run_id, model_id, probe_id, task_type, domain, quality_speed,
                normalized_score, judge_scores, is_self_eval, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()
        logger.info("probe_results_batch_stored", count=len(rows), run_id=run_id)

    # -----------------------------------------------------------------------
    # Profile storage
    # -----------------------------------------------------------------------

    def store_profile(
        self,
        profile: ModelSpectrographProfile,
        run_id: str,
    ) -> None:
        """Store or update a model's spectrography profile."""
        assert isinstance(profile, ModelSpectrographProfile), (
            "profile must be ModelSpectrographProfile"
        )
        assert isinstance(run_id, str) and run_id, "run_id must be non-empty"

        now = datetime.now(UTC).isoformat()
        rows: list[tuple[str, str, str, float, float, int, str, str]] = []

        for key, fs in profile.task_scores.items():
            rows.append((
                profile.model_id, "task", key,
                fs.score, fs.confidence, fs.sample_count, run_id, now,
            ))
        for key, fs in profile.domain_scores.items():
            rows.append((
                profile.model_id, "domain", key,
                fs.score, fs.confidence, fs.sample_count, run_id, now,
            ))
        for key, fs in profile.qs_scores.items():
            rows.append((
                profile.model_id, "qs", key,
                fs.score, fs.confidence, fs.sample_count, run_id, now,
            ))

        self.conn.executemany(
            """INSERT OR REPLACE INTO spectrography_profiles
               (model_id, dimension, dim_key, score, confidence, sample_count,
                run_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()

    def store_profiles_batch(
        self,
        profiles: dict[str, ModelSpectrographProfile],
        run_id: str,
    ) -> None:
        """Store multiple model profiles in a single transaction."""
        assert isinstance(profiles, dict), "profiles must be a dict"
        assert isinstance(run_id, str) and run_id, "run_id must be non-empty"

        now = datetime.now(UTC).isoformat()
        rows: list[tuple[str, str, str, float, float, int, str, str]] = []

        for profile in profiles.values():
            for key, fs in profile.task_scores.items():
                rows.append((
                    profile.model_id, "task", key,
                    fs.score, fs.confidence, fs.sample_count, run_id, now,
                ))
            for key, fs in profile.domain_scores.items():
                rows.append((
                    profile.model_id, "domain", key,
                    fs.score, fs.confidence, fs.sample_count, run_id, now,
                ))
            for key, fs in profile.qs_scores.items():
                rows.append((
                    profile.model_id, "qs", key,
                    fs.score, fs.confidence, fs.sample_count, run_id, now,
                ))

        self.conn.executemany(
            """INSERT OR REPLACE INTO spectrography_profiles
               (model_id, dimension, dim_key, score, confidence, sample_count,
                run_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        self.conn.commit()
        logger.info(
            "profiles_batch_stored",
            model_count=len(profiles),
            row_count=len(rows),
            run_id=run_id,
        )

    # -----------------------------------------------------------------------
    # Profile retrieval
    # -----------------------------------------------------------------------

    def load_profile(self, model_id: str) -> ModelSpectrographProfile | None:
        """Load a single model's profile from the database.

        Returns None if no profile exists for this model.
        """
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty"

        cursor = self.conn.execute(
            """SELECT dimension, dim_key, score, confidence, sample_count, updated_at
               FROM spectrography_profiles
               WHERE model_id = ?""",
            (model_id,),
        )
        rows = cursor.fetchall()
        if not rows:
            return None

        return _rows_to_profile(model_id, rows)

    def load_all_profiles(self) -> dict[str, ModelSpectrographProfile]:
        """Load all model profiles from the database."""
        cursor = self.conn.execute(
            """SELECT model_id, dimension, dim_key, score, confidence,
                      sample_count, updated_at
               FROM spectrography_profiles
               ORDER BY model_id""",
        )
        rows = cursor.fetchall()
        if not rows:
            return {}

        # Group by model_id
        by_model: dict[str, list[sqlite3.Row]] = {}
        for row in rows:
            mid = row["model_id"]
            by_model.setdefault(mid, []).append(row)

        profiles: dict[str, ModelSpectrographProfile] = {}
        for mid, model_rows in by_model.items():
            profile = _rows_to_profile(mid, model_rows)
            if profile is not None:
                profiles[mid] = profile

        return profiles

    # -----------------------------------------------------------------------
    # Query helpers
    # -----------------------------------------------------------------------

    def get_probe_results_for_model(
        self,
        model_id: str,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Retrieve probe results for a model, most recent first."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty"
        assert limit > 0, "limit must be positive"

        cursor = self.conn.execute(
            """SELECT run_id, probe_id, task_type, domain, quality_speed,
                      normalized_score, judge_scores, is_self_eval, error,
                      created_at
               FROM spectrography_probe_results
               WHERE model_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (model_id, limit),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_run_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve recent spectrography run metadata."""
        assert limit > 0, "limit must be positive"

        cursor = self.conn.execute(
            """SELECT run_id, started_at, completed_at, judge_model,
                      model_count, probe_count, error_count, status
               FROM spectrography_runs
               ORDER BY started_at DESC
               LIMIT ?""",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]

    def get_latest_profile_timestamp(self, model_id: str) -> str | None:
        """Return the most recent updated_at for a model's profile, or None."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty"

        cursor = self.conn.execute(
            """SELECT MAX(updated_at) as latest
               FROM spectrography_profiles
               WHERE model_id = ?""",
            (model_id,),
        )
        row = cursor.fetchone()
        if row is None or row["latest"] is None:
            return None
        return str(row["latest"])

    def has_recent_profile(
        self,
        model_id: str,
        max_age_days: int = 30,
    ) -> bool:
        """Check if a model has a profile newer than max_age_days."""
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty"
        assert max_age_days > 0, "max_age_days must be positive"

        latest = self.get_latest_profile_timestamp(model_id)
        if latest is None:
            return False

        updated_at = datetime.fromisoformat(latest)
        if updated_at.tzinfo is None:
            updated_at = updated_at.replace(tzinfo=UTC)

        age_days = (datetime.now(UTC) - updated_at).total_seconds() / 86400.0
        return age_days <= max_age_days


# ---------------------------------------------------------------------------
# Row-to-profile conversion
# ---------------------------------------------------------------------------


def _rows_to_profile(
    model_id: str,
    rows: list[sqlite3.Row],
) -> ModelSpectrographProfile | None:
    """Convert database rows into a ModelSpectrographProfile."""
    assert isinstance(model_id, str) and model_id, "model_id must be non-empty"
    assert len(rows) > 0, "rows must not be empty"

    task_scores: dict[str, SpectrographScore] = {}
    domain_scores: dict[str, SpectrographScore] = {}
    qs_scores: dict[str, SpectrographScore] = {}
    latest_updated = ""

    for row in rows:
        dimension = row["dimension"]
        dim_key = row["dim_key"]
        fs = SpectrographScore(
            score=float(row["score"]),
            confidence=float(row["confidence"]),
            sample_count=int(row["sample_count"]),
        )
        updated = str(row["updated_at"])
        if updated > latest_updated:
            latest_updated = updated

        if dimension == "task":
            task_scores[dim_key] = fs
        elif dimension == "domain":
            domain_scores[dim_key] = fs
        elif dimension == "qs":
            qs_scores[dim_key] = fs

    # Fill missing dimensions with neutral defaults
    for key in IBR_TASK_TYPES:
        if key not in task_scores:
            task_scores[key] = IBR_NEUTRAL_SPECTROGRAPH
    for key in IBR_DOMAINS:
        if key not in domain_scores:
            domain_scores[key] = IBR_NEUTRAL_SPECTROGRAPH
    for key in IBR_QUALITY_SPEED:
        if key not in qs_scores:
            qs_scores[key] = IBR_NEUTRAL_SPECTROGRAPH

    return ModelSpectrographProfile(
        model_id=model_id,
        version=_SCHEMA_VERSION,
        updated_at=latest_updated,
        task_scores=task_scores,
        domain_scores=domain_scores,
        qs_scores=qs_scores,
    )
