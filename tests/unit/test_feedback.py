"""Tests for selection/feedback.py — feedback-loop learning engine.

Spec traceability: IBR spec v0.1.0 section 3.2 (Method 2).
AC numbers: IBR-FLV-03, IBR-FLV-05.
"""

from __future__ import annotations

import pytest

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_FLAVOR,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    ClassifiedIntent,
    FlavorScore,
    ModelFlavorProfile,
)
from dragonlight_router.selection.feedback import (
    FeedbackStore,
    _apply_floor,
    _build_profiles_from_rows,
    _build_single_profile,
    _get_score_map,
    _resolve_floor,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_intent(**overrides) -> ClassifiedIntent:
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
) -> ModelFlavorProfile:
    """Build a ModelFlavorProfile with optional partial scores."""

    def _build_scores(
        raw: dict[str, float] | None,
        allowed: frozenset[str],
    ) -> dict[str, FlavorScore]:
        scores: dict[str, FlavorScore] = {}
        parsed = raw or {}
        for key in allowed:
            if key in parsed:
                scores[key] = FlavorScore(
                    score=parsed[key],
                    confidence=1.0,
                    sample_count=0,
                )
            else:
                scores[key] = IBR_NEUTRAL_FLAVOR
        return scores

    return ModelFlavorProfile(
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
    store = FeedbackStore(db_path=tmp_path / "test_feedback.db")
    yield store
    store.close()


# ---------------------------------------------------------------------------
# FeedbackStore — schema and initialization
# ---------------------------------------------------------------------------


class TestFeedbackStoreInit:
    """FeedbackStore initialization and schema creation."""

    def test_creates_db_file(self, tmp_path):
        """Database file is created on init."""
        db_path = tmp_path / "feedback.db"
        store = FeedbackStore(db_path=db_path)
        assert db_path.exists()
        store.close()

    def test_schema_has_feedback_scores_table(self, feedback_store):
        """feedback_scores table exists after init."""
        cursor = feedback_store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='feedback_scores'"
        )
        assert cursor.fetchone() is not None

    def test_wal_mode_enabled(self, feedback_store):
        """WAL journal mode is set for concurrency safety."""
        result = feedback_store._conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0] == "wal"

    def test_idempotent_schema_creation(self, tmp_path):
        """Calling _ensure_schema twice does not error."""
        db_path = tmp_path / "feedback.db"
        store = FeedbackStore(db_path=db_path)
        store._ensure_schema()  # second call should be safe
        store.close()


# ---------------------------------------------------------------------------
# FeedbackStore — record_feedback
# ---------------------------------------------------------------------------


class TestRecordFeedback:
    """EMA updates via record_feedback."""

    def test_first_observation_uses_raw_value(self, feedback_store):
        """First observation for a dimension uses the raw normalized value."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 5)

        profiles = feedback_store.get_learned_profiles()
        assert "model-a" in profiles
        profile = profiles["model-a"]
        # quality_rating=5 -> observation=1.0
        assert profile.task_scores["analysis"].score == pytest.approx(
            1.0,
            abs=1e-9,
        )

    def test_ema_update_second_observation(self, feedback_store):
        """Second observation applies EMA: 0.1 * new + 0.9 * old."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 5)  # obs = 1.0
        feedback_store.record_feedback("model-a", intent, 1)  # obs = 0.2

        profiles = feedback_store.get_learned_profiles()
        profile = profiles["model-a"]
        # EMA: 0.1 * 0.2 + 0.9 * 1.0 = 0.92
        assert profile.task_scores["analysis"].score == pytest.approx(
            0.92,
            abs=1e-9,
        )

    def test_sample_count_increments(self, feedback_store):
        """Sample count increments with each observation."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 3)
        feedback_store.record_feedback("model-a", intent, 4)
        feedback_store.record_feedback("model-a", intent, 5)

        profiles = feedback_store.get_learned_profiles()
        profile = profiles["model-a"]
        assert profile.task_scores["analysis"].sample_count == 3

    def test_confidence_grows_with_samples(self, feedback_store):
        """Confidence = min(1.0, sample_count / 50)."""
        intent = _make_intent()
        # Record 10 observations
        for _ in range(10):
            feedback_store.record_feedback("model-a", intent, 4)

        profiles = feedback_store.get_learned_profiles()
        profile = profiles["model-a"]
        assert profile.task_scores["analysis"].confidence == pytest.approx(
            10 / 50,
            abs=1e-9,
        )

    def test_confidence_caps_at_one(self, feedback_store):
        """Confidence caps at 1.0 after 50+ samples."""
        intent = _make_intent()
        for _ in range(60):
            feedback_store.record_feedback("model-a", intent, 4)

        profiles = feedback_store.get_learned_profiles()
        profile = profiles["model-a"]
        assert profile.task_scores["analysis"].confidence == pytest.approx(
            1.0,
            abs=1e-9,
        )

    def test_all_three_dimensions_updated(self, feedback_store):
        """record_feedback updates task, domain, and qs dimensions."""
        intent = _make_intent(
            task_type="creative",
            domain="technical",
            quality_speed="quality",
        )
        feedback_store.record_feedback("model-a", intent, 5)

        profiles = feedback_store.get_learned_profiles()
        profile = profiles["model-a"]
        assert profile.task_scores["creative"].sample_count == 1
        assert profile.domain_scores["technical"].sample_count == 1
        assert profile.qs_scores["quality"].sample_count == 1

    def test_quality_rating_normalization(self, feedback_store):
        """Quality ratings 1-5 normalize to 0.2-1.0."""
        for rating in (1, 2, 3, 4, 5):
            store = FeedbackStore(
                db_path=feedback_store._db_path.parent / f"norm_{rating}.db",
            )
            intent = _make_intent()
            store.record_feedback(f"m-{rating}", intent, rating)
            profiles = store.get_learned_profiles()
            expected = rating / 5.0
            assert profiles[f"m-{rating}"].task_scores["analysis"].score == (
                pytest.approx(expected, abs=1e-9)
            )
            store.close()

    def test_multiple_models_independent(self, feedback_store):
        """Feedback for different models does not interfere."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 5)
        feedback_store.record_feedback("model-b", intent, 1)

        profiles = feedback_store.get_learned_profiles()
        assert profiles["model-a"].task_scores["analysis"].score == pytest.approx(
            1.0,
            abs=1e-9,
        )
        assert profiles["model-b"].task_scores["analysis"].score == pytest.approx(
            0.2,
            abs=1e-9,
        )


# ---------------------------------------------------------------------------
# Floor enforcement (IBR-FLV-03)
# ---------------------------------------------------------------------------


class TestFloorEnforcement:
    """IBR-FLV-03: Feedback cannot lower below 80% of operator-declared value."""

    def test_floor_prevents_score_dropping_below_threshold(
        self,
        feedback_store,
    ):
        """Low rating cannot drop score below 80% of operator value."""
        operator = _make_profile(
            "model-a",
            task_scores={"analysis": 0.9},
        )
        intent = _make_intent()
        # rating=1 -> observation=0.2, floor = 0.8 * 0.9 = 0.72
        feedback_store.record_feedback("model-a", intent, 1, operator)

        profiles = feedback_store.get_learned_profiles()
        # First obs: raw = 0.2, floored to 0.72
        assert profiles["model-a"].task_scores["analysis"].score >= 0.72

    def test_floor_allows_scores_above_threshold(self, feedback_store):
        """Scores above the floor pass through unchanged."""
        operator = _make_profile(
            "model-a",
            task_scores={"analysis": 0.5},
        )
        intent = _make_intent()
        # rating=5 -> observation=1.0, floor = 0.8 * 0.5 = 0.4
        feedback_store.record_feedback("model-a", intent, 5, operator)

        profiles = feedback_store.get_learned_profiles()
        assert profiles["model-a"].task_scores["analysis"].score == pytest.approx(
            1.0,
            abs=1e-9,
        )

    def test_no_floor_without_operator_profile(self, feedback_store):
        """Without operator profile, no floor enforcement."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 1)

        profiles = feedback_store.get_learned_profiles()
        # rating=1 -> observation=0.2, no floor
        assert profiles["model-a"].task_scores["analysis"].score == pytest.approx(
            0.2,
            abs=1e-9,
        )


# ---------------------------------------------------------------------------
# get_learned_profiles
# ---------------------------------------------------------------------------


class TestGetLearnedProfiles:
    """Loading learned profiles from SQLite."""

    def test_empty_db_returns_empty_dict(self, feedback_store):
        """No feedback recorded returns empty dict."""
        profiles = feedback_store.get_learned_profiles()
        assert profiles == {}

    def test_profiles_have_all_dimensions(self, feedback_store):
        """Loaded profiles fill all taxonomy dimensions."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 4)

        profiles = feedback_store.get_learned_profiles()
        profile = profiles["model-a"]
        assert len(profile.task_scores) == len(IBR_TASK_TYPES)
        assert len(profile.domain_scores) == len(IBR_DOMAINS)
        assert len(profile.qs_scores) == len(IBR_QUALITY_SPEED)

    def test_unfeedback_dimensions_get_neutral(self, feedback_store):
        """Dimensions without feedback get neutral defaults."""
        intent = _make_intent(task_type="analysis")
        feedback_store.record_feedback("model-a", intent, 4)

        profiles = feedback_store.get_learned_profiles()
        profile = profiles["model-a"]
        # "creative" was never observed
        assert profile.task_scores["creative"] == IBR_NEUTRAL_FLAVOR

    def test_updated_at_set(self, feedback_store):
        """Profile updated_at is set to a non-empty timestamp."""
        intent = _make_intent()
        feedback_store.record_feedback("model-a", intent, 3)

        profiles = feedback_store.get_learned_profiles()
        assert profiles["model-a"].updated_at != ""


# ---------------------------------------------------------------------------
# Pure helper tests
# ---------------------------------------------------------------------------


class TestPureHelpers:
    """Stateless helper functions."""

    def test_apply_floor_raises_score(self):
        """_apply_floor raises score to floor when below."""
        assert _apply_floor(0.3, 0.5) == pytest.approx(0.5, abs=1e-9)

    def test_apply_floor_no_op_when_above(self):
        """_apply_floor is a no-op when score >= floor."""
        assert _apply_floor(0.8, 0.5) == pytest.approx(0.8, abs=1e-9)

    def test_apply_floor_clamps_to_unit_interval(self):
        """_apply_floor result is always in [0.0, 1.0]."""
        assert 0.0 <= _apply_floor(0.0, 0.0) <= 1.0
        assert 0.0 <= _apply_floor(1.0, 1.0) <= 1.0

    def test_resolve_floor_no_operator(self):
        """_resolve_floor returns 0.0 when no operator profile."""
        assert _resolve_floor("task", "analysis", None) == 0.0

    def test_resolve_floor_with_operator(self):
        """_resolve_floor returns 0.8 * operator score."""
        profile = _make_profile(task_scores={"analysis": 0.9})
        floor = _resolve_floor("task", "analysis", profile)
        assert floor == pytest.approx(0.72, abs=1e-9)

    def test_resolve_floor_missing_dimension(self):
        """Floor for undeclared dimension uses neutral (0.5 * 0.8 = 0.4)."""
        profile = _make_profile()  # all neutral
        floor = _resolve_floor("task", "analysis", profile)
        assert floor == pytest.approx(0.4, abs=1e-9)

    def test_get_score_map_task(self):
        """_get_score_map returns task_scores for 'task'."""
        profile = _make_profile(task_scores={"analysis": 0.9})
        scores = _get_score_map("task", profile)
        assert scores["analysis"].score == 0.9

    def test_get_score_map_domain(self):
        """_get_score_map returns domain_scores for 'domain'."""
        profile = _make_profile(domain_scores={"code": 0.8})
        scores = _get_score_map("domain", profile)
        assert scores["code"].score == 0.8

    def test_get_score_map_qs(self):
        """_get_score_map returns qs_scores for 'qs'."""
        profile = _make_profile(qs_scores={"speed": 0.7})
        scores = _get_score_map("qs", profile)
        assert scores["speed"].score == 0.7

    def test_get_score_map_invalid_raises(self):
        """_get_score_map raises AssertionError for invalid dim_type."""
        profile = _make_profile()
        with pytest.raises(AssertionError, match="Unknown dim_type"):
            _get_score_map("invalid", profile)

    def test_build_single_profile_all_dimensions(self):
        """_build_single_profile fills all taxonomy dimensions."""
        rows = [
            ("m1", "task", "analysis", 0.8, 0.5, 25, "2026-01-01"),
            ("m1", "domain", "code", 0.7, 0.4, 20, "2026-01-01"),
            ("m1", "qs", "quality", 0.9, 0.6, 30, "2026-01-01"),
        ]
        profile = _build_single_profile("m1", rows)
        assert profile.task_scores["analysis"].score == 0.8
        assert profile.domain_scores["code"].score == 0.7
        assert profile.qs_scores["quality"].score == 0.9

    def test_build_profiles_from_rows_groups_by_model(self):
        """_build_profiles_from_rows groups rows by model_id."""
        rows = [
            ("m1", "task", "analysis", 0.8, 0.5, 25, "2026-01-01"),
            ("m2", "task", "creative", 0.9, 0.6, 30, "2026-01-01"),
        ]
        profiles = _build_profiles_from_rows(rows)
        assert len(profiles) == 2
        assert "m1" in profiles
        assert "m2" in profiles
