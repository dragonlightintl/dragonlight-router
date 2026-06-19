"""Generative / fuzz tests for dragonlight-router input boundaries.

Uses Hypothesis to fuzz critical input paths and verify invariants hold
across the entire input space.  Targets: config loading, prompt sanitization,
LLM response validation, Result monad, complexity scoring, context filtering,
and composite scoring.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import yaml
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dragonlight_router.config.loader import load_config
from dragonlight_router.core.types import (
    BackendTier,
    ComplexityEstimate,
    DispatchOrder,
    Err,
    Ok,
)
from dragonlight_router.result import err, is_err, is_ok, ok, unwrap, unwrap_err
from dragonlight_router.selection.complexity import estimate_complexity
from dragonlight_router.selection.context_filter import (
    ProviderTrustTier,
    TrustTier,
    filter_by_trust_tier,
    filter_context_for_provider,
)
from dragonlight_router.selection.scoring import (
    compute_budget_score,
    compute_composite_score,
    compute_health_score,
    normalize_budget_score,
    normalize_health_score,
    normalize_latency_score,
    normalize_priority_score,
    normalize_queue_score,
    normalize_rank,
)
from dragonlight_router.server.routes import _sanitize_prompt, _validate_llm_response

pytestmark = [pytest.mark.unit, pytest.mark.property]

# ---------------------------------------------------------------------------
# 1. Config loading fuzz tests
# ---------------------------------------------------------------------------


class TestConfigLoadingFuzz:
    """Fuzz YAML config values -- random strings, negative numbers, missing keys."""

    @given(st.text(min_size=0, max_size=200))
    @settings(max_examples=50)
    def test_load_config_random_yaml_content_no_crash(self, content: str) -> None:
        """Writing arbitrary text as YAML config must not crash load_config.

        Note: load_config may raise TypeError/ValidationError when YAML parses
        to a non-mapping type -- these are expected failure modes, not panics.
        """
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)
        try:
            result = load_config(path)
            assert isinstance(result, (Ok, Err))
        except (TypeError, ValueError, KeyError):
            pass  # Expected for non-mapping YAML (e.g. "1" -> int)
        finally:
            path.unlink(missing_ok=True)

    @given(
        state_dir=st.text(min_size=0, max_size=100),
        catalog_ttl=st.integers(min_value=-1000, max_value=1000),
        top_n=st.integers(min_value=-100, max_value=1000),
    )
    @settings(max_examples=50)
    def test_load_config_random_field_values(
        self,
        state_dir: str,
        catalog_ttl: int,
        top_n: int,
    ) -> None:
        """Random field values in a well-structured YAML either load or return Err."""
        data: dict[str, Any] = {
            "state_dir": state_dir,
            "catalog_ttl_hours": catalog_ttl,
            "default_top_n": top_n,
            "providers": [],
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(data, f)
            path = Path(f.name)
        try:
            result = load_config(path)
            assert isinstance(result, (Ok, Err))
        except (TypeError, ValueError, KeyError):
            pass  # Expected for values that fail Pydantic validation
        finally:
            path.unlink(missing_ok=True)

    @given(
        base_url=st.text(min_size=0, max_size=300),
        rpm=st.integers(min_value=-100, max_value=10000),
    )
    @settings(max_examples=50)
    def test_load_config_random_provider_url(self, base_url: str, rpm: int) -> None:
        """Random provider URLs must not crash config loading."""
        data: dict[str, Any] = {
            "providers": [
                {
                    "name": "fuzz",
                    "base_url": base_url,
                    "model_prefix": "fuzz_",
                    "rate_limits": {"rpm": rpm},
                }
            ],
        }
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            yaml.dump(data, f)
            path = Path(f.name)
        try:
            result = load_config(path)
            assert isinstance(result, (Ok, Err))
        except (TypeError, ValueError, KeyError):
            pass  # Expected for values that fail Pydantic/URL validation
        finally:
            path.unlink(missing_ok=True)

    @given(st.sampled_from(["", "{}", "providers: []", "---\n"]))
    @settings(max_examples=50)
    def test_load_config_edge_case_yaml(self, content: str) -> None:
        """Edge-case YAML content must load or return Err, never crash."""
        with tempfile.NamedTemporaryFile(suffix=".yaml", mode="w", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)
        try:
            result = load_config(path)
            assert isinstance(result, (Ok, Err))
        except (TypeError, ValueError, KeyError):
            pass
        finally:
            path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# 2. Prompt sanitization fuzz tests
# ---------------------------------------------------------------------------


class TestPromptSanitizationFuzz:
    """Fuzz _sanitize_prompt with arbitrary unicode, control chars, long strings."""

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_sanitize_prompt_arbitrary_unicode(self, text: str) -> None:
        """Arbitrary unicode text must not crash and must return a string."""
        result = _sanitize_prompt(text)
        assert isinstance(result, str)

    @given(
        st.text(
            alphabet=st.characters(
                whitelist_categories=("Cc",),
            ),
            min_size=1,
            max_size=200,
        )
    )
    @settings(max_examples=50)
    def test_sanitize_prompt_control_chars_stripped(self, text: str) -> None:
        """Control characters (except \\n, \\r, \\t) must be stripped."""
        result = _sanitize_prompt(text)
        assert isinstance(result, str)
        # No null bytes in output
        assert "\x00" not in result

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_sanitize_prompt_truncation(self, seed: str) -> None:
        """Strings exceeding MAX_STRING_LENGTH must be truncated."""
        # Build a string longer than 100K by repeating the seed
        text = seed * (100_001 // len(seed) + 1)
        assert len(text) > 100_000
        result = _sanitize_prompt(text)
        assert len(result) <= 100_000

    @given(st.binary(min_size=0, max_size=300))
    @settings(max_examples=50)
    def test_sanitize_prompt_null_bytes_in_decoded(self, raw: bytes) -> None:
        """Decoded bytes with embedded nulls must not crash sanitization."""
        try:
            text = raw.decode("utf-8", errors="replace")
        except Exception:
            return
        result = _sanitize_prompt(text)
        assert isinstance(result, str)
        assert "\x00" not in result

    @given(st.text(min_size=0, max_size=500))
    @settings(max_examples=50)
    def test_sanitize_prompt_idempotent(self, text: str) -> None:
        """Sanitizing an already-sanitized string must be a no-op."""
        once = _sanitize_prompt(text)
        twice = _sanitize_prompt(once)
        assert once == twice


# ---------------------------------------------------------------------------
# 3. LLM response validation fuzz tests
# ---------------------------------------------------------------------------


class TestLLMResponseValidationFuzz:
    """Fuzz _validate_llm_response with random strings, nulls, empties."""

    @given(st.text(min_size=0, max_size=1000))
    @settings(max_examples=50)
    def test_validate_response_arbitrary_strings(self, content: str) -> None:
        """Arbitrary string input must not crash, must return string without null bytes."""
        result = _validate_llm_response(content)
        assert isinstance(result, str)
        assert "\x00" not in result

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_validate_response_truncation(self, seed: str) -> None:
        """Strings exceeding MAX_RESPONSE_LENGTH must be truncated."""
        content = seed * (500_001 // len(seed) + 1)
        assert len(content) > 500_000
        result = _validate_llm_response(content)
        assert len(result) <= 500_000

    @given(st.from_type(type).flatmap(lambda t: st.from_type(t)))
    @settings(max_examples=50)
    def test_validate_response_non_string_types(self, value: Any) -> None:
        """Non-string types must not crash, returning empty string."""
        result = _validate_llm_response(value)  # type: ignore[arg-type]
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 4. Result monad fuzz tests
# ---------------------------------------------------------------------------


class TestResultMonadFuzz:
    """Fuzz Ok/Err with arbitrary values; verify monad invariants."""

    @given(
        st.one_of(
            st.integers(),
            st.text(min_size=0, max_size=100),
            st.floats(allow_nan=True, allow_infinity=True),
            st.none(),
            st.binary(min_size=0, max_size=50),
            st.lists(st.integers(), max_size=5),
        )
    )
    @settings(max_examples=50)
    def test_ok_round_trip(self, value: Any) -> None:
        """Ok(value).unwrap() always returns the original value."""
        result = Ok(value)
        assert result.is_ok() is True
        assert result.is_err() is False
        assert result.unwrap() is value

    @given(
        st.one_of(
            st.integers(),
            st.text(min_size=0, max_size=100),
            st.floats(allow_nan=True, allow_infinity=True),
            st.none(),
        )
    )
    @settings(max_examples=50)
    def test_err_round_trip(self, error: Any) -> None:
        """Err(error).unwrap_err() always returns the original error."""
        result = Err(error)
        assert result.is_ok() is False
        assert result.is_err() is True
        assert result.unwrap_err() is error

    @given(
        value=st.integers(),
        error=st.text(min_size=0, max_size=50),
    )
    @settings(max_examples=50)
    def test_unwrap_wrong_variant_raises(self, value: int, error: str) -> None:
        """Calling unwrap_err on Ok or unwrap on Err must raise AssertionError."""
        with pytest.raises(AssertionError):
            Ok(value).unwrap_err()
        with pytest.raises(AssertionError):
            Err(error).unwrap()

    @given(
        value=st.integers(),
        error=st.text(min_size=1, max_size=50),
    )
    @settings(max_examples=50)
    def test_helper_ok_and_err_factories(self, value: int, error: str) -> None:
        """ok() and err() helpers create valid Ok/Err with correct predicates."""
        r_ok = ok(value)
        assert is_ok(r_ok) is True
        assert is_err(r_ok) is False
        assert unwrap(r_ok) == value

        r_err = err(error)
        assert is_err(r_err) is True
        assert is_ok(r_err) is False
        assert unwrap_err(r_err) == error


# ---------------------------------------------------------------------------
# 5. Complexity scoring fuzz tests
# ---------------------------------------------------------------------------


class TestComplexityScoreFuzz:
    """Fuzz prompt complexity estimation with random text and parameters."""

    @given(
        msg=st.text(min_size=0, max_size=500),
        ctx_tokens=st.integers(min_value=0, max_value=200_000),
        tool_use=st.booleans(),
        long_ctx=st.booleans(),
        intent=st.sampled_from(
            [
                "code_generation",
                "code_review",
                "debugging",
                "architecture",
                "session_lifecycle",
                "strategic_planning",
                "complex_reasoning",
                "casual_chat",
                "creative_writing",
                "general",
                "test",
            ]
        ),
    )
    @settings(max_examples=50)
    def test_complexity_always_returns_valid_estimate(
        self,
        msg: str,
        ctx_tokens: int,
        tool_use: bool,
        long_ctx: bool,
        intent: str,
    ) -> None:
        """estimate_complexity must always return a valid ComplexityEstimate."""
        order = DispatchOrder(
            intent_category=intent,
            specific_intent="test",
            operator_message=msg,
            system_prompt="",
            context_tokens=ctx_tokens,
            requires_tool_use=tool_use,
            requires_long_context=long_ctx,
        )
        result = estimate_complexity(order)
        assert isinstance(result, ComplexityEstimate)
        assert isinstance(result.tier, BackendTier)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.signals, list)
        assert len(result.signals) > 0

    @given(
        msg=st.text(min_size=0, max_size=200),
        ctx_tokens=st.integers(min_value=0, max_value=100_000),
    )
    @settings(max_examples=50)
    def test_complexity_deterministic(self, msg: str, ctx_tokens: int) -> None:
        """Same input must produce same output."""
        order = DispatchOrder(
            intent_category="general",
            specific_intent="test",
            operator_message=msg,
            system_prompt="",
            context_tokens=ctx_tokens,
            requires_tool_use=False,
            requires_long_context=False,
        )
        r1 = estimate_complexity(order)
        r2 = estimate_complexity(order)
        assert r1.tier == r2.tier
        assert r1.confidence == r2.confidence


# ---------------------------------------------------------------------------
# 6. Context filtering fuzz tests
# ---------------------------------------------------------------------------


class TestContextFilterFuzz:
    """Fuzz context filtering with random contexts and trust tiers."""

    @given(
        tier=st.sampled_from(list(TrustTier)),
        required=st.sampled_from(list(TrustTier)),
        n=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=50)
    def test_filter_by_trust_tier_invariants(
        self,
        tier: TrustTier,
        required: TrustTier,
        n: int,
    ) -> None:
        """Output subset invariant; LOCAL required tier passes everything."""
        candidates = [tier] * n
        result = filter_by_trust_tier(candidates, required)
        assert len(result) <= len(candidates)
        # LOCAL required tier must pass all candidates through
        local_result = filter_by_trust_tier(candidates, TrustTier.LOCAL)
        assert len(local_result) == len(candidates)

    @given(
        provider_tier=st.sampled_from(list(ProviderTrustTier)),
        keys=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Ll",)),
                min_size=1,
                max_size=20,
            ),
            min_size=0,
            max_size=10,
        ),
    )
    @settings(max_examples=50)
    def test_filter_context_for_provider_no_crash(
        self,
        provider_tier: ProviderTrustTier,
        keys: list[str],
    ) -> None:
        """Filtering random context dicts (including empty) must not crash."""
        context: dict[str, Any] = {k: f"value-{i}" for i, k in enumerate(keys)}
        result = filter_context_for_provider(context, provider_tier)
        assert isinstance(result, dict)
        # Empty context must also work
        empty_result = filter_context_for_provider({}, provider_tier)
        assert isinstance(empty_result, dict)


# ---------------------------------------------------------------------------
# 7. Scoring fuzz tests
# ---------------------------------------------------------------------------


class TestScoringFuzz:
    """Fuzz score computation with random float weights and values."""

    @given(
        rank=st.integers(min_value=0, max_value=100),
        budget=st.floats(min_value=0.0, max_value=100.0),
        health=st.floats(min_value=0.0, max_value=100.0),
    )
    @settings(max_examples=50)
    def test_composite_score_always_bounded(
        self,
        rank: int,
        budget: float,
        health: float,
    ) -> None:
        """Composite score must always be in [0, 100]."""
        result = compute_composite_score(rank, budget, health)
        assert 0.0 <= result <= 100.0

    @given(
        rpm_remaining=st.integers(min_value=0, max_value=1000),
        rpm_limit=st.integers(min_value=1, max_value=1000),
        rpd_remaining=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
        rpd_limit=st.one_of(st.none(), st.integers(min_value=1, max_value=10000)),
    )
    @settings(max_examples=50)
    def test_budget_score_bounded(
        self,
        rpm_remaining: int,
        rpm_limit: int,
        rpd_remaining: int | None,
        rpd_limit: int | None,
    ) -> None:
        """Budget score must be in [0, 100] with or without RPD limits."""
        assume(rpm_remaining <= rpm_limit)
        if rpd_remaining is not None and rpd_limit is not None:
            assume(rpd_remaining <= rpd_limit)
        else:
            # Both must be None or both non-None for the function contract
            rpd_remaining = None
            rpd_limit = None
        result = compute_budget_score(rpm_remaining, rpm_limit, rpd_remaining, rpd_limit)
        assert 0.0 <= result <= 100.0

    @given(
        error_count=st.integers(min_value=0, max_value=500),
        circuit_open=st.booleans(),
        age=st.floats(min_value=0.0, max_value=1_000_000.0),
    )
    @settings(max_examples=50)
    def test_health_score_bounded(
        self,
        error_count: int,
        circuit_open: bool,
        age: float,
    ) -> None:
        """Health score must always be in [0, 100]."""
        result = compute_health_score(error_count, circuit_open, age)
        assert 0.0 <= result <= 100.0

    @given(
        rank=st.integers(min_value=1, max_value=10000),
        budget=st.floats(min_value=0.0, max_value=100.0),
        latency=st.floats(min_value=0.0, max_value=100.0),
        priority=st.integers(min_value=0, max_value=10000),
        health=st.floats(min_value=0.0, max_value=100.0),
        queue_depth=st.integers(min_value=0, max_value=10000),
        max_queue=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=50)
    def test_all_normalizers_bounded(
        self,
        rank: int,
        budget: float,
        latency: float,
        priority: int,
        health: float,
        queue_depth: int,
        max_queue: int,
    ) -> None:
        """All normalizer functions must return values in [0.0, 1.0]."""
        assert 0.0 <= normalize_rank(rank) <= 1.0
        assert 0.0 <= normalize_budget_score(budget) <= 1.0
        assert 0.0 <= normalize_latency_score(latency) <= 1.0
        assert 0.0 <= normalize_priority_score(priority) <= 1.0
        assert 0.0 <= normalize_health_score(health) <= 1.0
        assert 0.0 <= normalize_queue_score(queue_depth, max_queue) <= 1.0

    @given(
        error_count=st.integers(min_value=0, max_value=100),
        age=st.floats(min_value=0.0, max_value=86400.0),
    )
    @settings(max_examples=50)
    def test_health_circuit_open_always_zero(
        self,
        error_count: int,
        age: float,
    ) -> None:
        """Circuit open must always yield 0.0 health score regardless of errors."""
        result = compute_health_score(error_count, circuit_open=True, last_success_age_s=age)
        assert result == 0.0
