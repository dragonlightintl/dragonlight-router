"""Feedback-loop learning engine for IBR flavor profiles (v0.2.0).

Persists quality feedback in SQLite (WAL mode) and applies EMA updates
to learned model flavor profiles.  Feedback-learned scores overlay
operator-declared profiles but never lower a dimension below 80% of the
operator-declared value (IBR-FLV-03).

Spec reference: intent-based-router-v0.1.0-spec.md section 3.2, Method 2.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import structlog

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_FLAVOR,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    ClassifiedIntent,
    FlavorScore,
    ModelFlavorProfile,
)

logger = structlog.get_logger(__name__)

# EMA smoothing factor — low alpha means slow, stable convergence.
_EMA_ALPHA: float = 0.1

# Number of observations needed for full confidence.
_FULL_CONFIDENCE_SAMPLES: int = 50

# Floor ratio — feedback cannot lower below this fraction of operator value.
_FLOOR_RATIO: float = 0.8


# ---------------------------------------------------------------------------
# FeedbackStore — SQLite-backed storage for learned flavor profiles
# ---------------------------------------------------------------------------


class FeedbackStore:
    """SQLite-backed storage for feedback-learned flavor profiles.

    Thread-safe via WAL mode.  Each call to ``record_feedback`` updates
    the running EMA for one model across all three intent dimensions.
    """

    def __init__(self, db_path: Path) -> None:
        assert isinstance(db_path, Path), "db_path must be a Path instance"
        self._db_path = db_path
        self._conn = self._open_connection()
        self._ensure_schema()

    def _open_connection(self) -> sqlite3.Connection:
        """Open a SQLite connection with WAL mode for safe concurrency."""
        assert self._db_path.parent.exists(), (
            f"Parent directory must exist: {self._db_path.parent}"
        )
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def _ensure_schema(self) -> None:
        """Create the feedback_scores table if it does not exist."""
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS feedback_scores (
                model_id       TEXT NOT NULL,
                dimension_type TEXT NOT NULL,
                dimension_value TEXT NOT NULL,
                score          REAL NOT NULL DEFAULT 0.5,
                confidence     REAL NOT NULL DEFAULT 0.0,
                sample_count   INTEGER NOT NULL DEFAULT 0,
                updated_at     TEXT NOT NULL,
                PRIMARY KEY (model_id, dimension_type, dimension_value)
            )
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Record feedback
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        model_id: str,
        classified_intent: ClassifiedIntent,
        quality_rating: int,
        operator_profile: ModelFlavorProfile | None = None,
    ) -> None:
        """Update the learned profile for *model_id* via EMA.

        Normalizes quality_rating (1-5) to [0.0, 1.0], then updates all
        three dimensions (task_type, domain, quality_speed) from the
        ClassifiedIntent.  Floor enforcement (IBR-FLV-03) is applied when
        an operator_profile is provided.
        """
        assert isinstance(model_id, str) and model_id, "model_id must be non-empty"
        assert isinstance(classified_intent, ClassifiedIntent), (
            "classified_intent must be ClassifiedIntent"
        )
        assert isinstance(quality_rating, int) and 1 <= quality_rating <= 5, (
            "quality_rating must be int in [1, 5]"
        )

        observation = quality_rating / 5.0

        dims = [
            ("task", classified_intent.task_type),
            ("domain", classified_intent.domain),
            ("qs", classified_intent.quality_speed),
        ]
        for dim_type, dim_value in dims:
            floor = _resolve_floor(dim_type, dim_value, operator_profile)
            self._update_dimension(model_id, dim_type, dim_value, observation, floor)

        logger.debug(
            "feedback_recorded",
            model_id=model_id,
            quality_rating=quality_rating,
            observation=round(observation, 4),
        )

    def _update_dimension(
        self,
        model_id: str,
        dim_type: str,
        dim_value: str,
        observation: float,
        floor: float,
    ) -> None:
        """Apply EMA update for one dimension with floor enforcement."""
        assert 0.0 <= observation <= 1.0, "observation must be in [0.0, 1.0]"

        row = self._conn.execute(
            "SELECT score, sample_count FROM feedback_scores "
            "WHERE model_id = ? AND dimension_type = ? AND dimension_value = ?",
            (model_id, dim_type, dim_value),
        ).fetchone()

        if row is not None:
            old_score, old_count = row
            new_score = _EMA_ALPHA * observation + (1.0 - _EMA_ALPHA) * old_score
            new_count = old_count + 1
        else:
            new_score = observation
            new_count = 1

        new_score = _apply_floor(new_score, floor)
        confidence = min(1.0, new_count / _FULL_CONFIDENCE_SAMPLES)
        now = datetime.now(UTC).isoformat()

        self._conn.execute(
            "INSERT INTO feedback_scores "
            "(model_id, dimension_type, dimension_value, "
            "score, confidence, sample_count, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(model_id, dimension_type, dimension_value) "
            "DO UPDATE SET score=?, confidence=?, "
            "sample_count=?, updated_at=?",
            (model_id, dim_type, dim_value,
             new_score, confidence, new_count, now,
             new_score, confidence, new_count, now),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Load learned profiles
    # ------------------------------------------------------------------

    def get_learned_profiles(self) -> dict[str, ModelFlavorProfile]:
        """Load all feedback-learned profiles from SQLite.

        Returns a dict of model_id -> ModelFlavorProfile built from the
        stored dimension scores.  Dimensions without feedback data are
        filled with neutral defaults.
        """
        rows = self._conn.execute(
            "SELECT model_id, dimension_type, dimension_value, "
            "score, confidence, sample_count, updated_at "
            "FROM feedback_scores"
        ).fetchall()

        if not rows:
            return {}

        return _build_profiles_from_rows(rows)

    def close(self) -> None:
        """Close the SQLite connection."""
        self._conn.close()


# ---------------------------------------------------------------------------
# Pure helpers (stateless, testable in isolation)
# ---------------------------------------------------------------------------


def _resolve_floor(
    dim_type: str,
    dim_value: str,
    operator_profile: ModelFlavorProfile | None,
) -> float:
    """Compute the floor score for a dimension from the operator profile.

    Returns 0.0 when no operator profile is available (no floor enforced).
    """
    if operator_profile is None:
        return 0.0

    score_map = _get_score_map(dim_type, operator_profile)
    fs = score_map.get(dim_value, IBR_NEUTRAL_FLAVOR)
    return _FLOOR_RATIO * fs.score


def _get_score_map(
    dim_type: str,
    profile: ModelFlavorProfile,
) -> dict[str, FlavorScore]:
    """Return the correct score dict for a dimension type."""
    assert dim_type in ("task", "domain", "qs"), f"Unknown dim_type: {dim_type}"
    if dim_type == "task":
        return profile.task_scores
    if dim_type == "domain":
        return profile.domain_scores
    return profile.qs_scores


def _apply_floor(score: float, floor: float) -> float:
    """Enforce floor: score must not drop below floor (IBR-FLV-03).

    Also clamps the result to [0.0, 1.0].
    """
    assert 0.0 <= floor <= 1.0, f"floor out of bounds: {floor}"
    result = max(score, floor)
    result = max(0.0, min(1.0, result))
    assert 0.0 <= result <= 1.0, f"result out of bounds: {result}"
    return result


def _build_profiles_from_rows(
    rows: list[tuple[str, str, str, float, float, int, str]],
) -> dict[str, ModelFlavorProfile]:
    """Group SQLite rows into ModelFlavorProfile instances."""
    assert len(rows) > 0, "rows must not be empty"

    # Group rows by model_id
    grouped: dict[str, list[tuple[str, str, str, float, float, int, str]]] = {}
    for row in rows:
        model_id = row[0]
        if model_id not in grouped:
            grouped[model_id] = []
        grouped[model_id].append(row)

    profiles: dict[str, ModelFlavorProfile] = {}
    for model_id, model_rows in grouped.items():
        profiles[model_id] = _build_single_profile(model_id, model_rows)

    assert len(profiles) == len(grouped), "profile count must match group count"
    return profiles


def _build_single_profile(
    model_id: str,
    rows: list[tuple[str, str, str, float, float, int, str]],
) -> ModelFlavorProfile:
    """Build one ModelFlavorProfile from its feedback rows."""
    assert isinstance(model_id, str) and model_id, "model_id must be non-empty"

    task_scores = dict.fromkeys(IBR_TASK_TYPES, IBR_NEUTRAL_FLAVOR)
    domain_scores = dict.fromkeys(IBR_DOMAINS, IBR_NEUTRAL_FLAVOR)
    qs_scores = dict.fromkeys(IBR_QUALITY_SPEED, IBR_NEUTRAL_FLAVOR)

    latest_updated = ""
    for _mid, dim_type, dim_value, score, confidence, sample_count, updated_at in rows:
        fs = FlavorScore(score=score, confidence=confidence, sample_count=sample_count)
        if dim_type == "task" and dim_value in task_scores:
            task_scores[dim_value] = fs
        elif dim_type == "domain" and dim_value in domain_scores:
            domain_scores[dim_value] = fs
        elif dim_type == "qs" and dim_value in qs_scores:
            qs_scores[dim_value] = fs
        if updated_at > latest_updated:
            latest_updated = updated_at

    return ModelFlavorProfile(
        model_id=model_id,
        version=1,
        updated_at=latest_updated or datetime.now(UTC).isoformat(),
        task_scores=task_scores,
        domain_scores=domain_scores,
        qs_scores=qs_scores,
    )
