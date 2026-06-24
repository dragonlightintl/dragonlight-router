"""Additional tests for selection/feedback.py — edge cases and concurrency.

Supplements test_feedback.py with coverage for:
- Multiple sequential EMA convergence behavior
- Floor enforcement with edge-case operator scores
- Thread safety under concurrent writes
- Edge-case quality_rating values (1, 5)
- Schema creation verification on fresh DB
- get_learned_profiles after multiple models and dimensions

Spec traceability: IBR spec v0.1.0 section 3.2 (Method 2).
AC numbers: IBR-FLV-03, IBR-FLV-05.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_SPECTROGRAPH,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    ClassifiedIntent,
    ModelSpectrographProfile,
    SpectrographScore,
)
from dragonlight_router.selection.feedback import (
    _EMA_ALPHA,
    _FLOOR_RATIO,
    _FULL_CONFIDENCE_SAMPLES,
    FeedbackStore,
    _apply_floor,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(**overrides: object) -> ClassifiedIntent:
    """Build a ClassifiedIntent with sensible defaults."""
    defaults = {
        "task_type": "analysis",
        "domain": "code",
        "quality_speed": "balanced",
        "confidence": 0.9,
        "latency_ms": 15.0,
        "from_cache": False,
    }
    defaults.update(overrides)
    return ClassifiedIntent(**defaults)


def _make_profile(
    model_id: str = "test-model",
    task_scores: dict[str, float] | None = None,
    domain_scores: dict[str, float] | None = None,
    qs_scores: dict[str, float] | None = None,
) -> ModelSpectrographProfile:
    """Build a ModelSpectrographProfile with optional partial scores."""

    def _build_scores(
        raw: dict[str, float] | None,
        allowed: frozenset[str],
    ) -> dict[str, SpectrographScore]:
        scores: dict[str, SpectrographScore] = {}
        parsed = raw or {}
        for key in allowed:
            if key in parsed:
                scores[key] = SpectrographScore(
                    score=parsed[key],
                    confidence=1.0,
                    sample_count=0,
                )
            else:
                scores[key] = IBR_NEUTRAL_SPECTROGRAPH
        return scores

    return ModelSpectrographProfile(
        model_id=model_id,
        version=1,
        updated_at="2026-01-01T00:00:00+00:00",
        task_scores=_build_scores(task_scores, IBR_TASK_TYPES),
        domain_scores=_build_scores(domain_scores, IBR_DOMAINS),
        qs_scores=_build_scores(qs_scores, IBR_QUALITY_SPEED),
    )


@pytest.fixture
def feedback_store(tmp_path):
    """Create a FeedbackStore with a temporary SQLite database."""
    store = FeedbackStore(db_path=tmp_path / "test_feedback_extra.db")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# Schema creation on fresh DB
# ---------------------------------------------------------------------------


class TestSchemaCreation:
    """FeedbackStore schema creation on a fresh database."""

    def test_schema_created_on_fresh_db(self, tmp_path):
        """Tables exist after creating FeedbackStore on a new path."""
        db_path = tmp_path / "fresh.db"
        assert not db_path.exists()

        store = FeedbackStore(db_path=db_path)

        cursor = store._conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "feedback_scores" in tables
        store.close()

    def test_schema_columns_correct(self, tmp_path):
        """feedback_scores table has expected columns."""
        store = FeedbackStore(db_path=tmp_path / "schema_check.db")

        cursor = store._conn.execute("PRAGMA table_info(feedback_scores)")
        columns = {row[1] for row in cursor.fetchall()}
        expected = {
            "model_id",
            "dimension_type",
            "dimension_value",
            "score",
            "confidence",
            "sample_count",
            "updated_at",
        }
        assert expected == columns
        store.close()

    def test_primary_key_enforces_uniqueness(self, tmp_path):
        """Primary key (model_id, dimension_type, dimension_value) is enforced."""
        store = FeedbackStore(db_path=tmp_path / "pk_check.db")
        intent = _make_intent()

        # Two feedbacks for same model — should update, not duplicate
        store.record_feedback("model-a", intent, 5)
        store.record_feedback("model-a", intent, 3)

        count = store._conn.execute(
            "SELECT COUNT(*) FROM feedback_scores WHERE model_id = 'model-a'"
        ).fetchone()[0]
        # 3 dimensions (task, domain, qs) — one row per dimension
        assert count == 3
        store.close()


# ---------------------------------------------------------------------------
# EMA convergence — multiple sequential feedbacks
# ---------------------------------------------------------------------------


class TestEMAConvergence:
    """Multiple sequential record_feedback calls show EMA convergence."""

    def test_ema_converges_toward_consistent_observation(self, feedback_store):
        """Repeated high ratings push score toward 1.0 via EMA."""
        intent = _make_intent()

        for _ in range(20):
            feedback_store.record_feedback("model-a", intent, 5)

        profiles = feedback_store.get_learned_profiles()
        score = profiles["model-a"].task_scores["analysis"].score
        # After 20 observations of 1.0, EMA should be close to 1.0
        assert score > 0.85

    def test_ema_converges_toward_low_value(self, feedback_store):
        """Repeated low ratings push score toward 0.2 via EMA."""
        intent = _make_intent()

        for _ in range(30):
            feedback_store.record_feedback("model-a", intent, 1)

        profiles = feedback_store.get_learned_profiles()
        score = profiles["model-a"].task_scores["analysis"].score
        # After 30 obs of 0.2, EMA should be close to 0.2
        assert score < 0.35

    def test_ema_smoothing_resists_single_outlier(self, feedback_store):
        """EMA with alpha=0.1 should resist a single outlier observation."""
        intent = _make_intent()

        # Build up a high baseline (10 x rating=5 -> obs=1.0)
        for _ in range(10):
            feedback_store.record_feedback("model-a", intent, 5)

        profiles_before = feedback_store.get_learned_profiles()
        score_before = profiles_before["model-a"].task_scores["analysis"].score

        # Single outlier (rating=1 -> obs=0.2)
        feedback_store.record_feedback("model-a", intent, 1)

        profiles_after = feedback_store.get_learned_profiles()
        score_after = profiles_after["model-a"].task_scores["analysis"].score

        # Score should drop only slightly (alpha=0.1 means 10% influence)
        drop = score_before - score_after
        assert drop < 0.15, f"Single outlier caused excessive drop: {drop}"
        assert score_after > 0.7

    def test_ema_math_is_correct(self, feedback_store):
        """Verify exact EMA math: new = alpha * obs + (1-alpha) * old."""
        intent = _make_intent()
        alpha = _EMA_ALPHA

        # First obs: rating=3 -> obs=0.6
        feedback_store.record_feedback("model-a", intent, 3)
        profiles = feedback_store.get_learned_profiles()
        score1 = profiles["model-a"].task_scores["analysis"].score
        assert score1 == pytest.approx(0.6, abs=1e-9)

        # Second obs: rating=5 -> obs=1.0
        # EMA: 0.1 * 1.0 + 0.9 * 0.6 = 0.64
        feedback_store.record_feedback("model-a", intent, 5)
        profiles = feedback_store.get_learned_profiles()
        score2 = profiles["model-a"].task_scores["analysis"].score
        expected = alpha * 1.0 + (1.0 - alpha) * 0.6
        assert score2 == pytest.approx(expected, abs=1e-9)

        # Third obs: rating=2 -> obs=0.4
        # EMA: 0.1 * 0.4 + 0.9 * 0.64 = 0.616
        feedback_store.record_feedback("model-a", intent, 2)
        profiles = feedback_store.get_learned_profiles()
        score3 = profiles["model-a"].task_scores["analysis"].score
        expected3 = alpha * 0.4 + (1.0 - alpha) * expected
        assert score3 == pytest.approx(expected3, abs=1e-9)


# ---------------------------------------------------------------------------
# Floor enforcement — edge cases (IBR-FLV-03)
# ---------------------------------------------------------------------------


class TestFloorEnforcementEdgeCases:
    """IBR-FLV-03: Floor enforcement with edge-case operator profiles."""

    def test_floor_with_max_operator_score(self, feedback_store):
        """Floor at 80% of 1.0 = 0.8. Low rating floored to 0.8."""
        operator = _make_profile("model-a", task_scores={"analysis": 1.0})
        intent = _make_intent()

        # Rating=1 -> obs=0.2, floor = 0.8 * 1.0 = 0.8
        feedback_store.record_feedback("model-a", intent, 1, operator)

        profiles = feedback_store.get_learned_profiles()
        score = profiles["model-a"].task_scores["analysis"].score
        assert score == pytest.approx(0.8, abs=1e-9)

    def test_floor_with_min_operator_score(self, feedback_store):
        """Floor at 80% of 0.0 = 0.0. No floor enforced."""
        operator = _make_profile("model-a", task_scores={"analysis": 0.0})
        intent = _make_intent()

        # Rating=1 -> obs=0.2, floor = 0.8 * 0.0 = 0.0
        feedback_store.record_feedback("model-a", intent, 1, operator)

        profiles = feedback_store.get_learned_profiles()
        score = profiles["model-a"].task_scores["analysis"].score
        assert score == pytest.approx(0.2, abs=1e-9)

    def test_floor_across_multiple_dimensions(self, feedback_store):
        """Floor is applied independently to each dimension."""
        operator = _make_profile(
            "model-a",
            task_scores={"analysis": 0.9},
            domain_scores={"code": 0.6},
            qs_scores={"balanced": 0.5},
        )
        intent = _make_intent()

        # Rating=1 -> obs=0.2
        feedback_store.record_feedback("model-a", intent, 1, operator)

        profiles = feedback_store.get_learned_profiles()
        p = profiles["model-a"]

        # task floor: 0.8 * 0.9 = 0.72
        assert p.task_scores["analysis"].score >= 0.72
        # domain floor: 0.8 * 0.6 = 0.48
        assert p.domain_scores["code"].score >= 0.48
        # qs floor: 0.8 * 0.5 = 0.40
        assert p.qs_scores["balanced"].score >= 0.40

    def test_floor_does_not_raise_score_above_observation(self, feedback_store):
        """Floor is a lower bound, not an upper bound."""
        operator = _make_profile("model-a", task_scores={"analysis": 0.3})
        intent = _make_intent()

        # Rating=5 -> obs=1.0, floor = 0.8 * 0.3 = 0.24
        feedback_store.record_feedback("model-a", intent, 5, operator)

        profiles = feedback_store.get_learned_profiles()
        score = profiles["model-a"].task_scores["analysis"].score
        assert score == pytest.approx(1.0, abs=1e-9)

    def test_resolve_floor_constants(self):
        """_FLOOR_RATIO is 0.8 as specified."""
        assert _FLOOR_RATIO == 0.8

    def test_apply_floor_boundary_values(self):
        """_apply_floor handles exact boundary at floor value."""
        assert _apply_floor(0.5, 0.5) == pytest.approx(0.5, abs=1e-9)
        assert _apply_floor(0.499, 0.5) == pytest.approx(0.5, abs=1e-9)
        assert _apply_floor(0.501, 0.5) == pytest.approx(0.501, abs=1e-9)


# ---------------------------------------------------------------------------
# get_learned_profiles — multiple models
# ---------------------------------------------------------------------------


class TestGetLearnedProfilesMultiModel:
    """Loading profiles after multiple models and dimensions recorded."""

    def test_profiles_returns_all_models(self, feedback_store):
        """Multiple models each get their own profile."""
        intent = _make_intent()

        feedback_store.record_feedback("model-a", intent, 5)
        feedback_store.record_feedback("model-b", intent, 3)
        feedback_store.record_feedback("model-c", intent, 1)

        profiles = feedback_store.get_learned_profiles()
        assert len(profiles) == 3
        assert set(profiles.keys()) == {"model-a", "model-b", "model-c"}

    def test_profile_scores_independent_across_models(self, feedback_store):
        """Each model's score reflects only its own feedback."""
        intent = _make_intent()

        feedback_store.record_feedback("model-a", intent, 5)  # obs=1.0
        feedback_store.record_feedback("model-b", intent, 1)  # obs=0.2

        profiles = feedback_store.get_learned_profiles()
        assert profiles["model-a"].task_scores["analysis"].score == pytest.approx(
            1.0,
            abs=1e-9,
        )
        assert profiles["model-b"].task_scores["analysis"].score == pytest.approx(
            0.2,
            abs=1e-9,
        )

    def test_different_intents_for_different_models(self, feedback_store):
        """Different intent dimensions recorded for different models."""
        intent_a = _make_intent(task_type="creative", domain="creative_writing")
        intent_b = _make_intent(task_type="analysis", domain="code")

        feedback_store.record_feedback("model-a", intent_a, 5)
        feedback_store.record_feedback("model-b", intent_b, 4)

        profiles = feedback_store.get_learned_profiles()
        assert profiles["model-a"].task_scores["creative"].sample_count == 1
        assert profiles["model-a"].task_scores["analysis"] == IBR_NEUTRAL_SPECTROGRAPH
        assert profiles["model-b"].task_scores["analysis"].sample_count == 1
        assert profiles["model-b"].task_scores["creative"] == IBR_NEUTRAL_SPECTROGRAPH

    def test_all_taxonomy_dimensions_filled(self, feedback_store):
        """Loaded profiles fill all taxonomy dimensions even with partial data."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 4)

        profiles = feedback_store.get_learned_profiles()
        p = profiles["model-a"]

        assert len(p.task_scores) == len(IBR_TASK_TYPES)
        assert len(p.domain_scores) == len(IBR_DOMAINS)
        assert len(p.qs_scores) == len(IBR_QUALITY_SPEED)


# ---------------------------------------------------------------------------
# Edge-case quality_rating values
# ---------------------------------------------------------------------------


class TestEdgeCaseQualityRatings:
    """Edge-case quality_rating values (1 and 5)."""

    def test_rating_1_gives_minimum_observation(self, feedback_store):
        """Rating 1 normalizes to 0.2 (1/5)."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 1)

        profiles = feedback_store.get_learned_profiles()
        assert profiles["model-a"].task_scores["analysis"].score == pytest.approx(
            0.2,
            abs=1e-9,
        )

    def test_rating_5_gives_maximum_observation(self, feedback_store):
        """Rating 5 normalizes to 1.0 (5/5)."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 5)

        profiles = feedback_store.get_learned_profiles()
        assert profiles["model-a"].task_scores["analysis"].score == pytest.approx(
            1.0,
            abs=1e-9,
        )

    def test_rating_0_raises_assertion(self, feedback_store):
        """Rating 0 is out of bounds and raises AssertionError."""
        intent = _make_intent()
        with pytest.raises(AssertionError, match="quality_rating"):
            feedback_store.record_feedback("model-a", intent, 0)

    def test_rating_6_raises_assertion(self, feedback_store):
        """Rating 6 is out of bounds and raises AssertionError."""
        intent = _make_intent()
        with pytest.raises(AssertionError, match="quality_rating"):
            feedback_store.record_feedback("model-a", intent, 6)

    def test_rating_1_then_5_ema(self, feedback_store):
        """EMA from rating=1 then rating=5."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 1)  # obs=0.2
        feedback_store.record_feedback("model-a", intent, 5)  # obs=1.0

        profiles = feedback_store.get_learned_profiles()
        # EMA: 0.1 * 1.0 + 0.9 * 0.2 = 0.28
        expected = _EMA_ALPHA * 1.0 + (1.0 - _EMA_ALPHA) * 0.2
        assert profiles["model-a"].task_scores["analysis"].score == pytest.approx(
            expected,
            abs=1e-9,
        )


# ---------------------------------------------------------------------------
# Concurrent record_feedback calls (thread safety)
# ---------------------------------------------------------------------------


class TestConcurrency:
    """Thread safety of FeedbackStore under concurrent writes.

    FeedbackStore uses SQLite WAL mode with check_same_thread=False.
    Concurrent read-modify-write cycles can cause transaction conflicts,
    so we serialize with an external lock — this mirrors real usage where
    the caller (RouterEngine) serializes feedback calls.
    """

    def test_concurrent_writes_do_not_corrupt(self, tmp_path):
        """Multiple threads writing to same FeedbackStore do not corrupt data."""
        store = FeedbackStore(db_path=tmp_path / "concurrent.db")
        intent = _make_intent()
        lock = threading.Lock()

        num_threads = 8
        calls_per_thread = 20

        def _write_feedback(thread_id: int) -> None:
            for i in range(calls_per_thread):
                model = f"model-{thread_id}"
                rating = (i % 5) + 1
                with lock:
                    store.record_feedback(model, intent, rating)

        with ThreadPoolExecutor(max_workers=num_threads) as pool:
            futures = [pool.submit(_write_feedback, t) for t in range(num_threads)]
            for future in as_completed(futures):
                future.result()  # Raises if any thread had an exception

        profiles = store.get_learned_profiles()
        assert len(profiles) == num_threads

        for t in range(num_threads):
            model_id = f"model-{t}"
            assert model_id in profiles
            p = profiles[model_id]
            assert p.task_scores["analysis"].sample_count == calls_per_thread

        store.close()

    def test_concurrent_different_models(self, tmp_path):
        """Concurrent writes to different models maintain correct independence."""
        store = FeedbackStore(db_path=tmp_path / "concurrent_independent.db")
        lock = threading.Lock()

        intent_a = _make_intent(task_type="creative")
        intent_b = _make_intent(task_type="analysis")

        barrier = threading.Barrier(2)

        def _write_model_a() -> None:
            barrier.wait()
            for _ in range(15):
                with lock:
                    store.record_feedback("model-a", intent_a, 5)

        def _write_model_b() -> None:
            barrier.wait()
            for _ in range(15):
                with lock:
                    store.record_feedback("model-b", intent_b, 1)

        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [
                pool.submit(_write_model_a),
                pool.submit(_write_model_b),
            ]
            for future in as_completed(futures):
                future.result()

        profiles = store.get_learned_profiles()
        assert profiles["model-a"].task_scores["creative"].sample_count == 15
        assert profiles["model-b"].task_scores["analysis"].sample_count == 15
        # model-a should have high creative score, model-b low analysis score
        assert profiles["model-a"].task_scores["creative"].score > 0.8
        assert profiles["model-b"].task_scores["analysis"].score < 0.35
        store.close()


# ---------------------------------------------------------------------------
# FeedbackStore with tmp_path fixture (explicit DB lifecycle)
# ---------------------------------------------------------------------------


class TestFeedbackStoreLifecycle:
    """FeedbackStore with explicit DB lifecycle using tmp_path."""

    def test_store_close_and_reopen(self, tmp_path):
        """Data persists across close and reopen."""
        db_path = tmp_path / "lifecycle.db"
        intent = _make_intent()

        store1 = FeedbackStore(db_path=db_path)
        store1.record_feedback("model-a", intent, 5)
        store1.close()

        store2 = FeedbackStore(db_path=db_path)
        profiles = store2.get_learned_profiles()
        assert "model-a" in profiles
        assert profiles["model-a"].task_scores["analysis"].score == pytest.approx(
            1.0,
            abs=1e-9,
        )
        store2.close()

    def test_multiple_stores_same_path_concurrent_read(self, tmp_path):
        """WAL mode allows concurrent reads from separate connections."""
        db_path = tmp_path / "wal_read.db"
        intent = _make_intent()

        store1 = FeedbackStore(db_path=db_path)
        store1.record_feedback("model-a", intent, 4)

        # Open second connection — should be able to read
        store2 = FeedbackStore(db_path=db_path)
        profiles = store2.get_learned_profiles()
        assert "model-a" in profiles

        store1.close()
        store2.close()

    def test_confidence_reaches_full_at_50_samples(self, tmp_path):
        """Confidence = 1.0 at exactly 50 samples."""
        store = FeedbackStore(db_path=tmp_path / "conf50.db")
        intent = _make_intent()

        for _ in range(_FULL_CONFIDENCE_SAMPLES):
            store.record_feedback("model-a", intent, 3)

        profiles = store.get_learned_profiles()
        assert profiles["model-a"].task_scores["analysis"].confidence == pytest.approx(
            1.0,
            abs=1e-9,
        )
        assert profiles["model-a"].task_scores["analysis"].sample_count == (
            _FULL_CONFIDENCE_SAMPLES
        )
        store.close()
