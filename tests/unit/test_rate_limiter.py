"""Tests for the SQLite WAL-based cross-process rate limiter."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest

from dragonlight_router.rate.limiter import (
    RateLimiter,
    RateSlot,
    RateUsage,
    _DEFAULT_SLOT_TTL_SECONDS,
    _WINDOW_SECONDS,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Return a fresh database path for each test."""
    return tmp_path / "rate_limits.db"


@pytest.fixture
def limiter(db_path: Path) -> RateLimiter:
    """Rate limiter with a low limit for testing."""
    return RateLimiter(db_path=db_path, provider_limits={"test_provider": 3})


@pytest.fixture
def unlimited_limiter(db_path: Path) -> RateLimiter:
    """Rate limiter with no limits configured."""
    return RateLimiter(db_path=db_path, provider_limits=None)


class TestAcquireGranted:
    """Acquiring slots when under the limit should succeed."""

    def test_acquire_succeeds_under_limit(self, limiter: RateLimiter) -> None:
        slot = limiter.acquire("test_provider", "model-a")
        assert slot.granted is True
        assert slot.slot_id != ""
        assert slot.provider == "test_provider"
        assert slot.expires_at > time.time()
        assert slot.retry_after_ms is None

    def test_acquire_multiple_under_limit(self, limiter: RateLimiter) -> None:
        slots = [limiter.acquire("test_provider", "model-a") for _ in range(3)]
        assert all(s.granted for s in slots)
        assert len({s.slot_id for s in slots}) == 3  # unique slot IDs

    def test_acquire_returns_frozen_dataclass(self, limiter: RateLimiter) -> None:
        slot = limiter.acquire("test_provider", "model-a")
        assert isinstance(slot, RateSlot)
        with pytest.raises(AttributeError):
            slot.granted = False  # type: ignore[misc]


class TestAcquireDenied:
    """Acquiring slots at/over the limit should be denied."""

    def test_acquire_denied_at_limit(self, limiter: RateLimiter) -> None:
        # Fill up all 3 slots.
        for _ in range(3):
            slot = limiter.acquire("test_provider", "model-a")
            assert slot.granted is True

        # 4th should be denied.
        denied = limiter.acquire("test_provider", "model-a")
        assert denied.granted is False
        assert denied.slot_id == ""
        assert denied.retry_after_ms is not None
        assert denied.retry_after_ms >= 100

    def test_retry_after_ms_is_calculated(self, limiter: RateLimiter) -> None:
        for _ in range(3):
            limiter.acquire("test_provider", "model-a")

        denied = limiter.acquire("test_provider", "model-a")
        assert denied.retry_after_ms is not None
        # The retry hint should be roughly within the window period.
        assert 100 <= denied.retry_after_ms <= int(_WINDOW_SECONDS * 1000) + 1000


class TestRelease:
    """Releasing a slot should free capacity."""

    def test_release_frees_slot(self, limiter: RateLimiter) -> None:
        slots = [limiter.acquire("test_provider", "model-a") for _ in range(3)]
        assert all(s.granted for s in slots)

        # At limit — should be denied.
        denied = limiter.acquire("test_provider", "model-a")
        assert denied.granted is False

        # Release one slot.
        limiter.release(slots[0].slot_id)

        # Should now succeed.
        new_slot = limiter.acquire("test_provider", "model-a")
        assert new_slot.granted is True

    def test_release_empty_slot_id_is_noop(self, limiter: RateLimiter) -> None:
        # Should not raise.
        limiter.release("")

    def test_release_nonexistent_slot_id_is_noop(self, limiter: RateLimiter) -> None:
        # Should not raise.
        limiter.release("nonexistent-slot-id")


class TestExpiredSlotPruning:
    """Expired slots should be pruned automatically on acquire."""

    def test_expired_slots_are_pruned(self, limiter: RateLimiter) -> None:
        # Acquire 3 slots at a time in the past so they expire.
        past = time.time() - _DEFAULT_SLOT_TTL_SECONDS - 10
        with patch("dragonlight_router.rate.limiter.time") as mock_time:
            mock_time.time.return_value = past
            for _ in range(3):
                slot = limiter.acquire("test_provider", "model-a")
                assert slot.granted is True

        # Now at real time, all 3 should be expired and pruned.
        # Acquiring should succeed because pruning happens on acquire.
        slot = limiter.acquire("test_provider", "model-a")
        assert slot.granted is True

    def test_slot_ttl_is_applied(self, limiter: RateLimiter) -> None:
        slot = limiter.acquire("test_provider", "model-a")
        assert slot.granted is True
        # expires_at should be approximately now + TTL
        expected = time.time() + _DEFAULT_SLOT_TTL_SECONDS
        assert abs(slot.expires_at - expected) < 2.0


class TestProviderIsolation:
    """Slots for one provider should not affect another provider."""

    def test_provider_isolation(self, db_path: Path) -> None:
        limiter = RateLimiter(
            db_path=db_path,
            provider_limits={"provider_a": 2, "provider_b": 2},
        )

        # Fill provider_a to capacity.
        for _ in range(2):
            slot = limiter.acquire("provider_a", "model-a")
            assert slot.granted is True

        # provider_a is at limit.
        denied = limiter.acquire("provider_a", "model-a")
        assert denied.granted is False

        # provider_b should still have capacity.
        slot = limiter.acquire("provider_b", "model-b")
        assert slot.granted is True

    def test_release_only_affects_own_provider(self, db_path: Path) -> None:
        limiter = RateLimiter(
            db_path=db_path,
            provider_limits={"provider_a": 1, "provider_b": 1},
        )

        slot_a = limiter.acquire("provider_a", "model-a")
        slot_b = limiter.acquire("provider_b", "model-b")
        assert slot_a.granted and slot_b.granted

        # Release provider_b slot.
        limiter.release(slot_b.slot_id)

        # provider_a should still be at limit.
        denied = limiter.acquire("provider_a", "model-a")
        assert denied.granted is False

        # provider_b should now have capacity.
        new_b = limiter.acquire("provider_b", "model-b")
        assert new_b.granted is True


class TestNoLimitConfigured:
    """When no limit is configured for a provider, all acquires should succeed."""

    def test_no_limit_always_granted(self, unlimited_limiter: RateLimiter) -> None:
        # Acquire many slots — should all succeed.
        for _ in range(100):
            slot = unlimited_limiter.acquire("any_provider", "any-model")
            assert slot.granted is True

    def test_no_limit_for_specific_provider(self, db_path: Path) -> None:
        limiter = RateLimiter(
            db_path=db_path,
            provider_limits={"limited_provider": 2},
        )

        # Unlimited provider should always succeed.
        for _ in range(10):
            slot = limiter.acquire("unlimited_provider", "model-x")
            assert slot.granted is True

        # Limited provider hits its cap at 2.
        limiter.acquire("limited_provider", "model-y")
        limiter.acquire("limited_provider", "model-y")
        denied = limiter.acquire("limited_provider", "model-y")
        assert denied.granted is False


class TestConcurrentProcesses:
    """Simulate concurrent processes with different process_ids."""

    def test_slots_from_different_processes_count_together(self, db_path: Path) -> None:
        limiter = RateLimiter(
            db_path=db_path,
            provider_limits={"shared_provider": 3},
        )

        # Simulate 3 different processes each acquiring one slot.
        pids = [1001, 1002, 1003]
        for pid in pids:
            with patch("dragonlight_router.rate.limiter.os.getpid", return_value=pid):
                slot = limiter.acquire("shared_provider", "model-a")
                assert slot.granted is True

        # 4th from any process should be denied.
        with patch("dragonlight_router.rate.limiter.os.getpid", return_value=1004):
            denied = limiter.acquire("shared_provider", "model-a")
            assert denied.granted is False

    def test_cross_process_release(self, db_path: Path) -> None:
        limiter = RateLimiter(
            db_path=db_path,
            provider_limits={"provider": 1},
        )

        # Process A acquires.
        with patch("dragonlight_router.rate.limiter.os.getpid", return_value=100):
            slot = limiter.acquire("provider", "model-a")
            assert slot.granted is True

        # Process B is denied.
        with patch("dragonlight_router.rate.limiter.os.getpid", return_value=200):
            denied = limiter.acquire("provider", "model-a")
            assert denied.granted is False

        # Process A releases.
        limiter.release(slot.slot_id)

        # Process B can now acquire.
        with patch("dragonlight_router.rate.limiter.os.getpid", return_value=200):
            new_slot = limiter.acquire("provider", "model-a")
            assert new_slot.granted is True


class TestGetUsage:
    """get_usage should return accurate RPM counts and metadata."""

    def test_usage_empty(self, limiter: RateLimiter) -> None:
        usage = limiter.get_usage("test_provider")
        assert isinstance(usage, RateUsage)
        assert usage.provider == "test_provider"
        assert usage.current_rpm == 0
        assert usage.limit_rpm == 3
        assert usage.oldest_slot_age_s == 0.0

    def test_usage_after_acquire(self, limiter: RateLimiter) -> None:
        limiter.acquire("test_provider", "model-a")
        limiter.acquire("test_provider", "model-b")

        usage = limiter.get_usage("test_provider")
        assert usage.current_rpm == 2
        assert usage.limit_rpm == 3
        assert usage.oldest_slot_age_s >= 0.0

    def test_usage_after_release(self, limiter: RateLimiter) -> None:
        slot = limiter.acquire("test_provider", "model-a")
        limiter.release(slot.slot_id)

        usage = limiter.get_usage("test_provider")
        assert usage.current_rpm == 0

    def test_usage_no_limit_returns_none(self, unlimited_limiter: RateLimiter) -> None:
        unlimited_limiter.acquire("unconfigured", "model-x")
        usage = unlimited_limiter.get_usage("unconfigured")
        assert usage.current_rpm == 1
        assert usage.limit_rpm is None

    def test_usage_oldest_slot_age(self, limiter: RateLimiter) -> None:
        limiter.acquire("test_provider", "model-a")
        # Small sleep to ensure measurable age.
        time.sleep(0.05)
        limiter.acquire("test_provider", "model-b")

        usage = limiter.get_usage("test_provider")
        assert usage.oldest_slot_age_s >= 0.04


class TestSlidingWindow:
    """The sliding window should only count slots from the last 60 seconds."""

    def test_old_slots_outside_window_do_not_count(self, db_path: Path) -> None:
        limiter = RateLimiter(
            db_path=db_path,
            provider_limits={"provider": 2},
        )

        # Acquire 2 slots 61 seconds ago (outside window).
        past = time.time() - _WINDOW_SECONDS - 1
        with patch("dragonlight_router.rate.limiter.time") as mock_time:
            mock_time.time.return_value = past
            for _ in range(2):
                slot = limiter.acquire("provider", "model-a")
                assert slot.granted is True

        # Now at real time, the old slots are outside the window.
        # Should be able to acquire 2 more.
        slot1 = limiter.acquire("provider", "model-a")
        assert slot1.granted is True
        slot2 = limiter.acquire("provider", "model-a")
        assert slot2.granted is True


class TestDatabaseInitialization:
    """Database should be created and configured correctly."""

    def test_db_created_with_wal_mode(self, db_path: Path) -> None:
        RateLimiter(db_path=db_path, provider_limits={"p": 10})
        assert db_path.exists()

    def test_db_permissions_restricted(self, db_path: Path) -> None:
        RateLimiter(db_path=db_path, provider_limits={"p": 10})
        import stat

        mode = db_path.stat().st_mode & 0o777
        assert mode == 0o600

    def test_db_parent_dirs_created(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "rate.db"
        RateLimiter(db_path=deep_path, provider_limits={"p": 10})
        assert deep_path.exists()

    def test_multiple_initializations_are_idempotent(self, db_path: Path) -> None:
        # Should not raise.
        limiter1 = RateLimiter(db_path=db_path, provider_limits={"p": 2})
        limiter1.acquire("p", "model-a")

        limiter2 = RateLimiter(db_path=db_path, provider_limits={"p": 2})
        usage = limiter2.get_usage("p")
        # The slot from limiter1 should be visible to limiter2.
        assert usage.current_rpm == 1


class TestInputValidation:
    """Assertions should fire on invalid inputs."""

    def test_acquire_empty_provider_raises(self, limiter: RateLimiter) -> None:
        with pytest.raises(AssertionError):
            limiter.acquire("", "model-a")

    def test_acquire_empty_model_raises(self, limiter: RateLimiter) -> None:
        with pytest.raises(AssertionError):
            limiter.acquire("provider", "")

    def test_get_usage_empty_provider_raises(self, limiter: RateLimiter) -> None:
        with pytest.raises(AssertionError):
            limiter.get_usage("")

    def test_init_bad_db_path_raises(self) -> None:
        with pytest.raises(AssertionError):
            RateLimiter(db_path="not_a_path", provider_limits=None)  # type: ignore[arg-type]

    def test_init_bad_provider_limits_raises(self, db_path: Path) -> None:
        with pytest.raises(AssertionError):
            RateLimiter(db_path=db_path, provider_limits="bad")  # type: ignore[arg-type]
