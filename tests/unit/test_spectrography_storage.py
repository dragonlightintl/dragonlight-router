"""Tests for spectrography/storage.py — SQLite profile persistence.

Tests the WAL-mode SQLite store for spectrography profiles, probe results,
and run metadata.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_SPECTROGRAPH,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    ModelSpectrographProfile,
    SpectrographScore,
)
from dragonlight_router.spectrography.analyzer import ProbeResult
from dragonlight_router.spectrography.storage import SpectrographyStore

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_probe_result(
    model_id: str = "test/model-a",
    probe_id: str = "disc-test-001",
    task_type: str = "generation",
    domain: str = "code",
    quality_speed: str = "quality",
    normalized_score: float = 0.8,
    error: str | None = None,
) -> ProbeResult:
    """Build a ProbeResult with sensible defaults."""
    return ProbeResult(
        model_id=model_id,
        probe_id=probe_id,
        task_type=task_type,
        domain=domain,
        quality_speed=quality_speed,
        normalized_score=normalized_score,
        judge_scores={"accuracy": 4, "completeness": 3, "clarity": 4, "relevance": 5},
        is_self_eval=False,
        error=error,
    )


def _make_profile(
    model_id: str = "test/model-a",
    score: float = 0.7,
    confidence: float = 0.8,
    sample_count: int = 5,
) -> ModelSpectrographProfile:
    """Build a ModelSpectrographProfile with uniform scores."""
    fs = SpectrographScore(score=score, confidence=confidence, sample_count=sample_count)
    return ModelSpectrographProfile(
        model_id=model_id,
        version=1,
        updated_at=datetime.now(UTC).isoformat(),
        task_scores=dict.fromkeys(IBR_TASK_TYPES, fs),
        domain_scores=dict.fromkeys(IBR_DOMAINS, fs),
        qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, fs),
    )


@pytest.fixture()
def store(tmp_path):
    """Create and open a temporary SpectrographyStore."""
    db_path = tmp_path / "test_spectrography.db"
    s = SpectrographyStore(db_path)
    s.open()
    yield s
    s.close()


# ===========================================================================
# Connection / Schema tests
# ===========================================================================


class TestStoreInit:
    """Tests for SpectrographyStore initialization."""

    def test_creates_database_file(self, tmp_path):
        db_path = tmp_path / "new.db"
        store = SpectrographyStore(db_path)
        store.open()
        assert db_path.exists()
        store.close()

    def test_creates_parent_directories(self, tmp_path):
        db_path = tmp_path / "deep" / "nested" / "store.db"
        store = SpectrographyStore(db_path)
        store.open()
        assert db_path.exists()
        store.close()

    def test_uses_wal_mode(self, tmp_path):
        db_path = tmp_path / "wal_test.db"
        store = SpectrographyStore(db_path)
        store.open()
        cursor = store.conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        store.close()

    def test_schema_version_is_set(self, store):
        cursor = store.conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 1

    def test_idempotent_open(self, tmp_path):
        """Opening an existing database does not error."""
        db_path = tmp_path / "reopen.db"
        store = SpectrographyStore(db_path)
        store.open()
        store.close()
        # Reopen
        store2 = SpectrographyStore(db_path)
        store2.open()
        cursor = store2.conn.execute("SELECT version FROM schema_version")
        version = cursor.fetchone()[0]
        assert version == 1
        store2.close()


# ===========================================================================
# Run tracking tests
# ===========================================================================


class TestRunTracking:
    """Tests for run start/complete recording."""

    def test_record_run_start(self, store):
        store.record_run_start("run-001", "gemini/gemini-2.5-pro", 5, 80)
        runs = store.get_run_history()
        assert len(runs) == 1
        assert runs[0]["run_id"] == "run-001"
        assert runs[0]["status"] == "running"
        assert runs[0]["model_count"] == 5
        assert runs[0]["probe_count"] == 80

    def test_record_run_complete(self, store):
        store.record_run_start("run-002", "gemini/gemini-2.5-pro", 3, 40)
        store.record_run_complete("run-002", error_count=2, status="complete")
        runs = store.get_run_history()
        assert runs[0]["status"] == "complete"
        assert runs[0]["error_count"] == 2
        assert runs[0]["completed_at"] is not None

    def test_partial_status(self, store):
        store.record_run_start("run-003", "judge/model", 1, 10)
        store.record_run_complete("run-003", error_count=0, status="partial")
        runs = store.get_run_history()
        assert runs[0]["status"] == "partial"

    def test_multiple_runs_ordered_by_recency(self, store):
        store.record_run_start("run-a", "judge", 1, 1)
        store.record_run_start("run-b", "judge", 2, 2)
        store.record_run_start("run-c", "judge", 3, 3)
        runs = store.get_run_history()
        # Most recent first
        assert runs[0]["run_id"] == "run-c"
        assert len(runs) == 3

    def test_history_limit(self, store):
        for i in range(10):
            store.record_run_start(f"run-{i:03d}", "judge", 1, 1)
        runs = store.get_run_history(limit=3)
        assert len(runs) == 3


# ===========================================================================
# Probe result storage tests
# ===========================================================================


class TestProbeResultStorage:
    """Tests for storing and retrieving probe results."""

    def test_store_single_result(self, store):
        result = _make_probe_result()
        store.store_probe_result(result, "run-001")
        results = store.get_probe_results_for_model("test/model-a")
        assert len(results) == 1
        assert results[0]["probe_id"] == "disc-test-001"
        assert results[0]["normalized_score"] == pytest.approx(0.8)

    def test_store_result_with_error(self, store):
        result = _make_probe_result(error="test_error")
        store.store_probe_result(result, "run-001")
        results = store.get_probe_results_for_model("test/model-a")
        assert results[0]["error"] == "test_error"

    def test_store_result_preserves_judge_scores(self, store):
        result = _make_probe_result()
        store.store_probe_result(result, "run-001")
        results = store.get_probe_results_for_model("test/model-a")
        judge_scores = json.loads(results[0]["judge_scores"])
        assert judge_scores["accuracy"] == 4
        assert judge_scores["relevance"] == 5

    def test_batch_store(self, store):
        results = [
            _make_probe_result(model_id="m1", probe_id="p1"),
            _make_probe_result(model_id="m1", probe_id="p2"),
            _make_probe_result(model_id="m2", probe_id="p1"),
        ]
        store.store_probe_results_batch(results, "run-001")
        m1_results = store.get_probe_results_for_model("m1")
        assert len(m1_results) == 2
        m2_results = store.get_probe_results_for_model("m2")
        assert len(m2_results) == 1

    def test_results_ordered_by_recency(self, store):
        r1 = _make_probe_result(probe_id="p-old")
        r2 = _make_probe_result(probe_id="p-new")
        store.store_probe_result(r1, "run-001")
        store.store_probe_result(r2, "run-002")
        results = store.get_probe_results_for_model("test/model-a")
        # Most recent first
        assert results[0]["probe_id"] == "p-new"

    def test_results_limit(self, store):
        for i in range(20):
            r = _make_probe_result(probe_id=f"disc-p-{i:03d}")
            store.store_probe_result(r, "run-001")
        results = store.get_probe_results_for_model("test/model-a", limit=5)
        assert len(results) == 5

    def test_no_results_returns_empty(self, store):
        results = store.get_probe_results_for_model("nonexistent/model")
        assert results == []


# ===========================================================================
# Profile storage tests
# ===========================================================================


class TestProfileStorage:
    """Tests for storing and retrieving model profiles."""

    def test_store_and_load_single_profile(self, store):
        profile = _make_profile("test/model-a", score=0.85)
        store.store_profile(profile, "run-001")
        loaded = store.load_profile("test/model-a")
        assert loaded is not None
        assert loaded.model_id == "test/model-a"
        for t in IBR_TASK_TYPES:
            assert loaded.task_scores[t].score == pytest.approx(0.85)
        for d in IBR_DOMAINS:
            assert loaded.domain_scores[d].score == pytest.approx(0.85)
        for q in IBR_QUALITY_SPEED:
            assert loaded.qs_scores[q].score == pytest.approx(0.85)

    def test_load_nonexistent_profile_returns_none(self, store):
        result = store.load_profile("no/such/model")
        assert result is None

    def test_store_replaces_existing_profile(self, store):
        p1 = _make_profile("test/m1", score=0.3)
        store.store_profile(p1, "run-001")
        p2 = _make_profile("test/m1", score=0.9)
        store.store_profile(p2, "run-002")
        loaded = store.load_profile("test/m1")
        assert loaded is not None
        assert loaded.task_scores["generation"].score == pytest.approx(0.9)

    def test_batch_store_profiles(self, store):
        profiles = {
            "m1": _make_profile("m1", score=0.6),
            "m2": _make_profile("m2", score=0.8),
            "m3": _make_profile("m3", score=0.4),
        }
        store.store_profiles_batch(profiles, "run-001")
        all_profiles = store.load_all_profiles()
        assert len(all_profiles) == 3
        assert all_profiles["m1"].task_scores["generation"].score == pytest.approx(0.6)
        assert all_profiles["m2"].task_scores["generation"].score == pytest.approx(0.8)
        assert all_profiles["m3"].task_scores["generation"].score == pytest.approx(0.4)

    def test_load_all_empty_returns_empty(self, store):
        result = store.load_all_profiles()
        assert result == {}

    def test_profile_preserves_confidence_and_sample_count(self, store):
        profile = _make_profile("test/m1", score=0.7, confidence=0.9, sample_count=42)
        store.store_profile(profile, "run-001")
        loaded = store.load_profile("test/m1")
        assert loaded is not None
        fs = loaded.task_scores["generation"]
        assert fs.confidence == pytest.approx(0.9)
        assert fs.sample_count == 42

    def test_profile_fills_missing_dimensions_with_neutral(self, store):
        """A profile with only some dimensions stored should fill others with neutral."""
        # Manually insert partial data
        store.conn.execute(
            """INSERT INTO spectrography_profiles
               (model_id, dimension, dim_key, score, confidence, sample_count,
                run_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("partial/model", "task", "generation", 0.9, 0.8, 10, "run-x",
             datetime.now(UTC).isoformat()),
        )
        store.conn.commit()
        loaded = store.load_profile("partial/model")
        assert loaded is not None
        # Generation should have the stored value
        assert loaded.task_scores["generation"].score == pytest.approx(0.9)
        # Other task types should be neutral
        assert loaded.task_scores["analysis"] == IBR_NEUTRAL_SPECTROGRAPH
        # Domain and QS should all be neutral
        for d in IBR_DOMAINS:
            assert loaded.domain_scores[d] == IBR_NEUTRAL_SPECTROGRAPH


# ===========================================================================
# Timestamp / staleness helpers
# ===========================================================================


class TestTimestampHelpers:
    """Tests for profile timestamp queries."""

    def test_latest_profile_timestamp(self, store):
        profile = _make_profile("test/m1")
        store.store_profile(profile, "run-001")
        ts = store.get_latest_profile_timestamp("test/m1")
        assert ts is not None
        assert len(ts) > 0

    def test_latest_timestamp_missing_model(self, store):
        ts = store.get_latest_profile_timestamp("no/model")
        assert ts is None

    def test_has_recent_profile_true(self, store):
        profile = _make_profile("test/m1")
        store.store_profile(profile, "run-001")
        assert store.has_recent_profile("test/m1", max_age_days=30) is True

    def test_has_recent_profile_false_for_missing(self, store):
        assert store.has_recent_profile("no/model", max_age_days=30) is False

    def test_has_recent_profile_false_for_old(self, store):
        # Insert a profile with an old timestamp
        old_time = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        store.conn.execute(
            """INSERT INTO spectrography_profiles
               (model_id, dimension, dim_key, score, confidence, sample_count,
                run_id, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ("old/model", "task", "generation", 0.5, 0.5, 1, "old-run", old_time),
        )
        store.conn.commit()
        assert store.has_recent_profile("old/model", max_age_days=30) is False
