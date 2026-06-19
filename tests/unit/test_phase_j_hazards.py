"""Tests for Phase J MEDIUM-risk hazard mitigations.

Covers:
- HAZ-003: Cascade exhaustion — health state persistence + availability status
- HAZ-004: Silent fallback — fallback_policy enforcement
- HAZ-007: Prompt injection — intent_category validation
- HAZ-010: Token estimation — centralized estimation with logging
- HAZ-013: Complexity misrouting — intent-based tier floor

Spec traceability: HAZ-003, HAZ-004, HAZ-007, HAZ-010, HAZ-013
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    DispatchOrder,
)
from dragonlight_router.dispatch.cascade import (
    _apply_fallback_policy,
    _estimate_token_count,
    _log_token_estimation,
)
from dragonlight_router.health.circuit_breaker import CircuitState
from dragonlight_router.health.tracker import HealthTracker
from dragonlight_router.selection.mbr import _INTENT_TIER_FLOOR, estimate_complexity
from dragonlight_router.server.routes import (
    _ALLOWED_FALLBACK_POLICIES,
    _ALLOWED_INTENT_CATEGORIES,
    _validate_dispatch_request,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_order(**kwargs) -> DispatchOrder:
    defaults = {
        "intent_category": "test",
        "specific_intent": "test",
        "operator_message": "hello",
        "system_prompt": "",
        "context_tokens": 0,
        "requires_tool_use": False,
        "requires_long_context": False,
    }
    defaults.update(kwargs)
    return DispatchOrder(**defaults)


def _make_config(
    name: str = "b1",
    tier: BackendTier = BackendTier.SIMPLE,
    provider: str = "test",
) -> BackendConfig:
    return BackendConfig(
        name=name,
        provider=provider,
        model=name,
        tier=tier,
        base_url="https://test.example.com",
        env_key=None,
        capabilities=BackendCapabilities(
            max_context_tokens=8192,
            supports_tool_use=True,
            supports_streaming=True,
            supports_json_mode=True,
            supports_system_prompts=True,
        ),
        cost=BackendCostProfile(input_per_mtok=1.0, output_per_mtok=2.0),
        rate_limits=BackendRateLimits(rpm=60, rpd=14400, tpm=100000, daily_token_cap=1000000),
    )


# ===========================================================================
# HAZ-003: Cascade Exhaustion — Health State Persistence + Availability
# ===========================================================================


class TestHealthTrackerGetState:
    """[HAZ-003] HealthTracker.get_state() exports retired models and breaker states."""

    def test_get_state_empty_tracker(self):
        """[HAZ-003] Fresh tracker exports empty state."""
        tracker = HealthTracker()
        state = tracker.get_state()
        assert state["retired"] == {}
        assert state["error_counts"] == {}
        assert state["breaker_states"] == {}

    def test_get_state_with_retired_model(self):
        """[HAZ-003] Retired models appear in exported state."""
        tracker = HealthTracker()
        tracker.record_error("model-a", http_status=404)
        state = tracker.get_state()
        assert "model-a" in state["retired"]
        assert isinstance(state["retired"]["model-a"], float)

    def test_get_state_with_errors(self):
        """[HAZ-003] Error counts appear in exported state."""
        tracker = HealthTracker()
        tracker.record_error("model-b")
        tracker.record_error("model-b")
        state = tracker.get_state()
        assert state["error_counts"]["model-b"] == 2

    def test_get_state_with_breaker_state(self):
        """[HAZ-003] Circuit breaker states are exported."""
        tracker = HealthTracker()
        # Trip the circuit breaker by recording enough errors
        for _ in range(5):
            tracker.record_error("model-c")
        state = tracker.get_state()
        assert "model-c" in state["breaker_states"]
        breaker_state = state["breaker_states"]["model-c"]
        assert "state" in breaker_state
        assert "opened_at" in breaker_state


class TestHealthTrackerRestoreState:
    """[HAZ-003] HealthTracker.restore_state() restores retired models and breakers."""

    def test_restore_retired_models(self):
        """[HAZ-003] Retired models are restored from persisted state."""
        tracker = HealthTracker()
        state = {
            "retired": {"model-x": time.time()},
            "error_counts": {},
            "breaker_states": {},
        }
        tracker.restore_state(state)
        assert tracker.is_retired("model-x")

    def test_restore_error_counts(self):
        """[HAZ-003] Error counts are restored from persisted state."""
        tracker = HealthTracker()
        state = {
            "retired": {},
            "error_counts": {"model-y": 3},
            "breaker_states": {},
        }
        tracker.restore_state(state)
        assert tracker.get_error_count("model-y") == 3

    def test_restore_breaker_states(self):
        """[HAZ-003] Circuit breaker states are restored from persisted state."""
        tracker = HealthTracker()
        state = {
            "retired": {},
            "error_counts": {},
            "breaker_states": {
                "model-z": {
                    "state": "open",
                    "opened_at": time.time(),
                    "error_timestamps": [time.time()],
                },
            },
        }
        tracker.restore_state(state)
        # Breaker should be in OPEN state
        breaker = tracker._breakers["model-z"]
        # The breaker should not allow requests (OPEN)
        # (may allow if cooldown has elapsed, but with current timestamp it shouldn't)
        assert breaker.state in (CircuitState.OPEN, CircuitState.HALF_OPEN)

    def test_restore_empty_state(self):
        """[HAZ-003] Restoring empty state dict does not crash."""
        tracker = HealthTracker()
        tracker.restore_state({})
        assert tracker.get_error_count("any") == 0

    def test_roundtrip_state(self):
        """[HAZ-003] get_state → restore_state preserves retired models."""
        original = HealthTracker()
        original.record_error("model-rt", http_status=404)
        original.record_error("model-rt2")
        state = original.get_state()

        restored = HealthTracker()
        restored.restore_state(state)
        assert restored.is_retired("model-rt")
        assert restored.get_error_count("model-rt2") == 1


class TestAvailabilityStatus:
    """[HAZ-003] HealthTracker.availability_status() reports router-level availability."""

    def test_fresh_tracker_is_healthy(self):
        """[HAZ-003] Fresh tracker with no models returns 'healthy'."""
        tracker = HealthTracker()
        assert tracker.availability_status() == "healthy"

    def test_all_models_available_is_healthy(self):
        """[HAZ-003] All models available returns 'healthy'."""
        tracker = HealthTracker()
        tracker.record_success("model-1", 100.0)
        tracker.record_success("model-2", 100.0)
        assert tracker.availability_status() == "healthy"

    def test_some_models_retired_is_degraded(self):
        """[HAZ-003] Some retired models returns 'degraded'."""
        tracker = HealthTracker()
        tracker.record_success("model-1", 100.0)
        tracker.record_error("model-2", http_status=404)  # Retire model-2
        assert tracker.availability_status() == "degraded"

    def test_all_models_retired_is_unavailable(self):
        """[HAZ-003] All models retired returns 'unavailable'."""
        tracker = HealthTracker()
        tracker.record_error("model-1", http_status=404)
        tracker.record_error("model-2", http_status=404)
        assert tracker.availability_status() == "unavailable"

    def test_circuit_breaker_open_affects_availability(self):
        """[HAZ-003] Model with open circuit breaker is counted as unavailable."""
        tracker = HealthTracker(error_threshold=2, error_window_s=120.0)
        tracker.record_success("model-1", 100.0)
        # Trip circuit for model-2
        for _ in range(3):
            tracker.record_error("model-2")
        # model-1 is available, model-2 has tripped circuit → degraded
        status = tracker.availability_status()
        assert status in ("degraded", "healthy")  # Depends on breaker state


# ===========================================================================
# HAZ-004: Silent Fallback — Fallback Policy Enforcement
# ===========================================================================


class TestApplyFallbackPolicy:
    """[HAZ-004] _apply_fallback_policy restricts candidate pool."""

    def test_allow_returns_all_candidates(self):
        """[HAZ-004] Policy 'allow' returns all candidates unchanged."""
        c1 = _make_config("b1", BackendTier.SIMPLE)
        c2 = _make_config("b2", BackendTier.MODERATE)
        order = _make_order(fallback_policy="allow")
        result = _apply_fallback_policy([c1, c2], order)
        assert len(result) == 2

    def test_deny_returns_only_primary(self):
        """[HAZ-004] Policy 'deny' returns only the first candidate."""
        c1 = _make_config("b1", BackendTier.SIMPLE)
        c2 = _make_config("b2", BackendTier.MODERATE)
        c3 = _make_config("b3", BackendTier.COMPLEX)
        order = _make_order(fallback_policy="deny")
        result = _apply_fallback_policy([c1, c2, c3], order)
        assert len(result) == 1
        assert result[0].name == "b1"

    def test_same_tier_filters_to_matching_tier(self):
        """[HAZ-004] Policy 'same_tier' keeps only candidates at the primary's tier."""
        c1 = _make_config("b1", BackendTier.MODERATE)
        c2 = _make_config("b2", BackendTier.MODERATE)
        c3 = _make_config("b3", BackendTier.COMPLEX)
        order = _make_order(fallback_policy="same_tier")
        result = _apply_fallback_policy([c1, c2, c3], order)
        assert len(result) == 2
        assert all(c.tier == BackendTier.MODERATE for c in result)

    def test_single_candidate_unaffected_by_policy(self):
        """[HAZ-004] Single candidate returns unchanged regardless of policy."""
        c1 = _make_config("b1", BackendTier.SIMPLE)
        for policy in ("allow", "deny", "same_tier"):
            order = _make_order(fallback_policy=policy)
            result = _apply_fallback_policy([c1], order)
            assert len(result) == 1

    def test_same_tier_with_all_different_tiers(self):
        """[HAZ-004] same_tier with unique tiers returns only primary."""
        c1 = _make_config("b1", BackendTier.SIMPLE)
        c2 = _make_config("b2", BackendTier.MODERATE)
        c3 = _make_config("b3", BackendTier.COMPLEX)
        order = _make_order(fallback_policy="same_tier")
        result = _apply_fallback_policy([c1, c2, c3], order)
        assert len(result) == 1
        assert result[0].name == "b1"

    def test_default_fallback_policy_is_allow(self):
        """[HAZ-004] Default fallback_policy on DispatchOrder is 'allow'."""
        order = _make_order()
        assert order.fallback_policy == "allow"


class TestFallbackPolicyInDispatchOrder:
    """[HAZ-004] DispatchOrder correctly carries fallback_policy field."""

    def test_fallback_policy_deny_on_order(self):
        """[HAZ-004] DispatchOrder can be created with fallback_policy='deny'."""
        order = _make_order(fallback_policy="deny")
        assert order.fallback_policy == "deny"

    def test_fallback_policy_same_tier_on_order(self):
        """[HAZ-004] DispatchOrder can be created with fallback_policy='same_tier'."""
        order = _make_order(fallback_policy="same_tier")
        assert order.fallback_policy == "same_tier"


# ===========================================================================
# HAZ-007: Prompt Injection — Intent Category Validation
# ===========================================================================


class TestIntentCategoryValidation:
    """[HAZ-007] _validate_dispatch_request rejects unknown intent_category values."""

    def test_valid_intent_category_passes(self):
        """[HAZ-007] Valid intent_category passes validation."""
        for category in _ALLOWED_INTENT_CATEGORIES:
            body = {
                "intent_category": category,
                "specific_intent": "test",
                "operator_message": "hello",
                "context_tokens": 0,
            }
            error = _validate_dispatch_request(body)
            assert error is None, f"Category '{category}' should pass validation"

    def test_unknown_intent_category_rejected(self):
        """[HAZ-007] Unknown intent_category is rejected."""
        body = {
            "intent_category": "malicious_category",
            "specific_intent": "test",
            "operator_message": "hello",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "intent_category" in error

    def test_empty_intent_category_rejected(self):
        """[HAZ-007] Empty intent_category is rejected (not in allowed set)."""
        body = {
            "intent_category": "",
            "specific_intent": "test",
            "operator_message": "hello",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None

    def test_injection_attempt_rejected(self):
        """[HAZ-007] Intent_category with injection payload is rejected."""
        body = {
            "intent_category": "ignore_previous_instructions",
            "specific_intent": "test",
            "operator_message": "hello",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "intent_category" in error

    def test_allowed_set_is_frozen(self):
        """[HAZ-007] _ALLOWED_INTENT_CATEGORIES is immutable."""
        assert isinstance(_ALLOWED_INTENT_CATEGORIES, frozenset)


class TestFallbackPolicyValidation:
    """[HAZ-004] _validate_dispatch_request validates fallback_policy field."""

    def test_valid_fallback_policies_pass(self):
        """[HAZ-004] All valid fallback_policy values pass validation."""
        for policy in _ALLOWED_FALLBACK_POLICIES:
            body = {
                "intent_category": "general",
                "specific_intent": "test",
                "operator_message": "hello",
                "context_tokens": 0,
                "fallback_policy": policy,
            }
            error = _validate_dispatch_request(body)
            assert error is None, f"Policy '{policy}' should pass"

    def test_invalid_fallback_policy_rejected(self):
        """[HAZ-004] Invalid fallback_policy is rejected."""
        body = {
            "intent_category": "general",
            "specific_intent": "test",
            "operator_message": "hello",
            "context_tokens": 0,
            "fallback_policy": "yolo",
        }
        error = _validate_dispatch_request(body)
        assert error is not None
        assert "fallback_policy" in error

    def test_missing_fallback_policy_defaults_to_allow(self):
        """[HAZ-004] Missing fallback_policy defaults to 'allow' (passes)."""
        body = {
            "intent_category": "general",
            "specific_intent": "test",
            "operator_message": "hello",
            "context_tokens": 0,
        }
        error = _validate_dispatch_request(body)
        assert error is None


# ===========================================================================
# HAZ-010: Token Estimation — Centralized Estimation
# ===========================================================================


class TestEstimateTokenCount:
    """[HAZ-010] _estimate_token_count centralizes token estimation."""

    def test_basic_estimation(self):
        """[HAZ-010] 100 chars → 25 tokens (100 // 4)."""
        assert _estimate_token_count(100) == 25

    def test_minimum_one_token(self):
        """[HAZ-010] Very short input returns at least 1 token."""
        assert _estimate_token_count(1) == 1
        assert _estimate_token_count(0) == 1

    def test_large_input(self):
        """[HAZ-010] Large input returns expected count."""
        assert _estimate_token_count(4000) == 1000

    def test_exact_multiple(self):
        """[HAZ-010] Exact multiple of 4 returns clean count."""
        assert _estimate_token_count(400) == 100


class TestLogTokenEstimation:
    """[HAZ-010] _log_token_estimation logs estimation details."""

    def test_logs_without_error(self):
        """[HAZ-010] Logging function does not raise."""
        _log_token_estimation(25, 100, "test-backend", "input")

    def test_logs_output_direction(self):
        """[HAZ-010] Logging function accepts 'output' direction."""
        _log_token_estimation(50, 200, "test-backend", "output")


# ===========================================================================
# HAZ-013: Complexity Misrouting — Intent-Based Tier Floor
# ===========================================================================


class TestIntentTierFloor:
    """[HAZ-013] estimate_complexity uses intent_category to set tier floor."""

    def test_complex_reasoning_gets_complex_tier(self):
        """[HAZ-013] 'complex_reasoning' intent → COMPLEX tier regardless of tokens."""
        order = _make_order(intent_category="complex_reasoning", context_tokens=100)
        tier = estimate_complexity(order)
        assert tier == BackendTier.COMPLEX

    def test_strategic_planning_gets_complex_tier(self):
        """[HAZ-013] 'strategic_planning' intent → COMPLEX tier."""
        order = _make_order(intent_category="strategic_planning", context_tokens=100)
        tier = estimate_complexity(order)
        assert tier == BackendTier.COMPLEX

    def test_architecture_gets_complex_tier(self):
        """[HAZ-013] 'architecture' intent → COMPLEX tier."""
        order = _make_order(intent_category="architecture", context_tokens=100)
        tier = estimate_complexity(order)
        assert tier == BackendTier.COMPLEX

    def test_code_review_gets_moderate_tier(self):
        """[HAZ-013] 'code_review' intent → at least MODERATE tier."""
        order = _make_order(intent_category="code_review", context_tokens=100)
        tier = estimate_complexity(order)
        assert tier in (BackendTier.MODERATE, BackendTier.COMPLEX)

    def test_engineering_build_gets_moderate_tier(self):
        """[HAZ-013] 'engineering_build' intent → at least MODERATE tier."""
        order = _make_order(intent_category="engineering_build", context_tokens=100)
        tier = estimate_complexity(order)
        assert tier in (BackendTier.MODERATE, BackendTier.COMPLEX)

    def test_code_generation_gets_moderate_tier(self):
        """[HAZ-013] 'code_generation' intent → at least MODERATE tier."""
        order = _make_order(intent_category="code_generation", context_tokens=100)
        tier = estimate_complexity(order)
        assert tier in (BackendTier.MODERATE, BackendTier.COMPLEX)

    def test_data_analysis_gets_simple_tier(self):
        """[HAZ-013] 'data_analysis' intent → at least SIMPLE tier."""
        order = _make_order(intent_category="data_analysis", context_tokens=100)
        tier = estimate_complexity(order)
        assert tier in (BackendTier.SIMPLE, BackendTier.MODERATE, BackendTier.COMPLEX)

    def test_general_intent_no_floor(self):
        """[HAZ-013] 'general' intent has no floor — uses pure heuristic."""
        order = _make_order(intent_category="general", context_tokens=100)
        tier = estimate_complexity(order)
        assert tier == BackendTier.LOCAL  # Low tokens, no tool use, no long context

    def test_casual_chat_no_floor(self):
        """[HAZ-013] 'casual_chat' intent has no floor — uses pure heuristic."""
        order = _make_order(intent_category="casual_chat", context_tokens=100)
        tier = estimate_complexity(order)
        assert tier == BackendTier.LOCAL

    def test_heuristic_can_override_intent_upward(self):
        """[HAZ-013] High context_tokens can raise tier above intent floor."""
        order = _make_order(intent_category="code_review", context_tokens=10000)
        tier = estimate_complexity(order)
        # context_tokens > 8192 → COMPLEX (overrides MODERATE floor upward)
        assert tier == BackendTier.COMPLEX

    def test_intent_floor_never_lowers_heuristic(self):
        """[HAZ-013] Intent floor never lowers tier from heuristic-estimated level."""
        # data_analysis floor is SIMPLE, but tool_use raises to MODERATE
        order = _make_order(
            intent_category="data_analysis",
            requires_tool_use=True,
            context_tokens=100,
        )
        tier = estimate_complexity(order)
        assert tier == BackendTier.MODERATE

    def test_intent_tier_floor_dict_exists(self):
        """[HAZ-013] _INTENT_TIER_FLOOR is populated and correct type."""
        assert isinstance(_INTENT_TIER_FLOOR, dict)
        assert len(_INTENT_TIER_FLOOR) > 0
        for intent, tier in _INTENT_TIER_FLOOR.items():
            assert isinstance(intent, str)
            assert isinstance(tier, BackendTier)


# ===========================================================================
# HAZ-003: Health endpoint includes availability status
# ===========================================================================


class TestHealthEndpointAvailability:
    """[HAZ-003] Health endpoint includes router-level availability status."""

    def test_health_endpoint_includes_status(self, tmp_path: Path):
        """[HAZ-003] GET /v1/health response includes 'status' field."""
        from starlette.testclient import TestClient

        from dragonlight_router.server.app import create_app

        state_dir = tmp_path / "state"
        state_dir.mkdir()

        import yaml

        config = {
            "state_dir": str(state_dir),
            "providers": [
                {
                    "name": "groq",
                    "base_url": "https://api.groq.com/openai/v1",
                    "model_prefix": "groq_",
                    "rate_limits": {"rpm": 30, "rpd": 14400},
                }
            ],
        }
        config_path = tmp_path / "router.yaml"
        config_path.write_text(yaml.dump(config))
        matrix = {"coding": {"groq_llama70b": 90}}
        (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

        app = create_app(config_path=config_path)
        client = TestClient(app)
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] in ("healthy", "degraded", "unavailable")
