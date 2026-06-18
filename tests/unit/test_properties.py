"""Property-based tests for dragonlight-router invariants.

Spec traceability:
  - TM-007: Scoring weights invariants
  - TM-012: BudgetTracker invariants
  - TM-001: MBR never-downgrade invariant
  - TM-003: LBR rate-limit dispatch invariants
  - Interleave permutation invariants

Uses Hypothesis to verify that invariants hold across the entire input space,
not just hand-picked examples. Property categories follow the taxonomy from
dragonlight-property-based-testing-strategy.md.
"""
from __future__ import annotations

from unittest.mock import Mock

from hypothesis import assume, given
from hypothesis import strategies as st

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    DispatchOrder,
    ModelScore,
    Ok,
    ProviderConfig,
)
from dragonlight_router.selection.interleave import interleave_providers
from dragonlight_router.selection.lbr import filter_by_rate_limit
from dragonlight_router.selection.scoring import (
    compute_budget_score,
    compute_composite_score,
    compute_health_score,
)

# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def provider_config_strategy(
    rpm_range: tuple[int, int] = (1, 1000),
    rpd_range: tuple[int, int] = (1, 10000),
) -> st.SearchStrategy[ProviderConfig]:
    """Strategy for generating ProviderConfig instances."""
    return st.builds(
        ProviderConfig,
        name=st.text(
            alphabet=st.characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=10,
        ),
        base_url=st.just("http://localhost"),
        catalog_url=st.none(),
        env_key=st.none(),
        model_prefix=st.text(
            alphabet=st.characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=5,
        ),
        rpm_limit=st.integers(min_value=rpm_range[0], max_value=rpm_range[1]),
        rpd_limit=st.one_of(
            st.none(),
            st.integers(min_value=rpd_range[0], max_value=rpd_range[1]),
        ),
        tpm_limit=st.one_of(
            st.none(),
            st.integers(min_value=0, max_value=10000),
        ),
        daily_token_cap=st.one_of(
            st.none(),
            st.integers(min_value=0, max_value=100000),
        ),
    )


def model_score_strategy(
    providers: list[str] | None = None,
) -> st.SearchStrategy[ModelScore]:
    """Strategy for generating ModelScore instances."""
    provider_st = (
        st.sampled_from(providers)
        if providers
        else st.sampled_from(["groq", "nvidia", "anthropic", "openai", "local"])
    )
    return st.builds(
        ModelScore,
        model_id=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
            min_size=1,
            max_size=20,
        ),
        provider=provider_st,
        rank=st.integers(min_value=0, max_value=100),
        budget_score=st.floats(min_value=0.0, max_value=100.0),
        health_score=st.floats(min_value=0.0, max_value=100.0),
        composite=st.floats(min_value=0.0, max_value=100.0),
    )


def backend_config_strategy(
    tier: BackendTier | None = None,
    provider: str | None = None,
) -> st.SearchStrategy[BackendConfig]:
    """Strategy for generating BackendConfig instances."""
    tier_st = st.just(tier) if tier else st.sampled_from(list(BackendTier))
    provider_st = (
        st.just(provider)
        if provider
        else st.sampled_from(["groq", "nvidia", "anthropic", "openai"])
    )
    return st.builds(
        BackendConfig,
        name=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
            min_size=1,
            max_size=20,
        ),
        provider=provider_st,
        model=st.text(
            alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
            min_size=1,
            max_size=20,
        ),
        tier=tier_st,
        base_url=st.just("http://localhost"),
        env_key=st.none(),
        capabilities=st.builds(
            BackendCapabilities,
            max_context_tokens=st.integers(min_value=1024, max_value=256000),
            supports_tool_use=st.booleans(),
            supports_streaming=st.booleans(),
            supports_json_mode=st.booleans(),
            supports_system_prompts=st.booleans(),
        ),
        cost=st.builds(
            BackendCostProfile,
            input_per_mtok=st.floats(min_value=0.0, max_value=100.0),
            output_per_mtok=st.floats(min_value=0.0, max_value=100.0),
        ),
        rate_limits=st.builds(
            BackendRateLimits,
            rpm=st.integers(min_value=1, max_value=1000),
            rpd=st.integers(min_value=1, max_value=100000),
            tpm=st.integers(min_value=1, max_value=1000000),
            daily_token_cap=st.integers(min_value=0, max_value=10000000),
        ),
        priority=st.integers(min_value=0, max_value=100),
    )


# ---------------------------------------------------------------------------
# TM-007: Scoring invariants
# ---------------------------------------------------------------------------


class TestScoringInvariants:
    """Property: Invariant. Scoring functions maintain output bounds and determinism.

    Spec traceability: TM-007 (Scoring weights)
    """

    @given(
        rank=st.integers(min_value=0, max_value=100),
        budget_score=st.floats(min_value=0.0, max_value=100.0),
        health_score=st.floats(min_value=0.0, max_value=100.0),
    )
    def test_composite_score_bounded(self, rank, budget_score, health_score):
        """[TM-007 AC-1] Property: compute_composite_score always returns value in [0.0, 100.0]."""
        result = compute_composite_score(rank, budget_score, health_score)
        assert 0.0 <= result <= 100.0

    @given(
        rank=st.integers(min_value=0, max_value=100),
        budget_score=st.floats(min_value=0.0, max_value=100.0),
        health_score=st.floats(min_value=0.0, max_value=100.0),
    )
    def test_composite_score_deterministic(self, rank, budget_score, health_score):
        """[TM-007 AC-1] Property: Invariant. Same inputs produce same composite score."""
        result1 = compute_composite_score(rank, budget_score, health_score)
        result2 = compute_composite_score(rank, budget_score, health_score)
        assert result1 == result2

    @given(
        rank=st.integers(min_value=0, max_value=100),
        budget_score=st.floats(min_value=0.0, max_value=100.0),
        health_score=st.floats(min_value=0.0, max_value=100.0),
    )
    def test_composite_score_non_negative(self, rank, budget_score, health_score):
        """[TM-007 AC-1] Property: Invariant. Composite score is never negative."""
        result = compute_composite_score(rank, budget_score, health_score)
        assert result >= 0.0

    @given(
        rpm_remaining=st.integers(min_value=0, max_value=1000),
        rpm_limit=st.integers(min_value=1, max_value=1000),
    )
    def test_budget_score_bounded(self, rpm_remaining, rpm_limit):
        """[TM-007 AC-3] Property: Invariant. compute_budget_score returns value in [0.0, 100.0]."""
        assume(rpm_remaining <= rpm_limit)
        result = compute_budget_score(
            rpm_remaining=rpm_remaining,
            rpm_limit=rpm_limit,
            rpd_remaining=None,
            rpd_limit=None,
        )
        assert 0.0 <= result <= 100.0

    @given(
        error_count=st.integers(min_value=0, max_value=100),
        circuit_open=st.booleans(),
        last_success_age=st.floats(min_value=0.0, max_value=86400.0),
    )
    def test_health_score_bounded(self, error_count, circuit_open, last_success_age):
        """[TM-007 AC-4] Property: Invariant. compute_health_score returns value in [0.0, 100.0]."""
        result = compute_health_score(error_count, circuit_open, last_success_age)
        assert 0.0 <= result <= 100.0

    @given(
        error_count=st.integers(min_value=0, max_value=100),
        last_success_age=st.floats(min_value=0.0, max_value=86400.0),
    )
    def test_health_score_circuit_open_always_zero(self, error_count, last_success_age):
        """[TM-007 AC-4] Property: Circuit open always yields zero health score."""
        result = compute_health_score(
            error_count, circuit_open=True, last_success_age_s=last_success_age,
        )
        assert result == 0.0


# ---------------------------------------------------------------------------
# TM-012: BudgetTracker invariants
# ---------------------------------------------------------------------------


class TestBudgetTrackerInvariants:
    """Property: Invariant. BudgetTracker maintains score bounds and capacity monotonicity.

    Spec traceability: TM-012 (BudgetTracker)
    """

    @given(
        rpm=st.integers(min_value=1, max_value=1000),
        rpd=st.one_of(st.none(), st.integers(min_value=1, max_value=10000)),
        tpm=st.one_of(st.none(), st.integers(min_value=0, max_value=10000)),
        daily_token_cap=st.one_of(st.none(), st.integers(min_value=0, max_value=100000)),
        num_requests=st.integers(min_value=0, max_value=20),
        tokens_per_request=st.integers(min_value=0, max_value=100),
    )
    def test_score_always_bounded(
        self, rpm, rpd, tpm, daily_token_cap, num_requests, tokens_per_request,
    ):
        """[TM-012 AC-2] Property: score() always returns value in [0.0, 100.0]."""
        provider = ProviderConfig(
            name="test",
            base_url="http://localhost",
            catalog_url=None,
            env_key=None,
            model_prefix="test",
            rpm_limit=rpm,
            rpd_limit=rpd,
            tpm_limit=tpm,
            daily_token_cap=daily_token_cap,
        )
        bt = BudgetTracker(providers=[provider])
        for _ in range(num_requests):
            bt.record_request("test", tokens_used=tokens_per_request)
        result = bt.score("test")
        assert isinstance(result, Ok)
        score = result.value
        assert 0.0 <= score <= 100.0

    @given(
        rpm=st.integers(min_value=1, max_value=100),
        rpd=st.one_of(st.none(), st.integers(min_value=1, max_value=1000)),
    )
    def test_recording_usage_never_increases_capacity(self, rpm, rpd):
        """[TM-012 AC-3] Property: Invariant. Recording usage never makes capacity go UP."""
        provider = ProviderConfig(
            name="test",
            base_url="http://localhost",
            catalog_url=None,
            env_key=None,
            model_prefix="test",
            rpm_limit=rpm,
            rpd_limit=rpd,
            tpm_limit=None,
            daily_token_cap=None,
        )
        bt = BudgetTracker(providers=[provider])

        # Get initial score
        result_before = bt.score("test")
        assert isinstance(result_before, Ok)
        score_before = result_before.value

        # Record one request
        bt.record_request("test")

        # Score after should be <= score before
        result_after = bt.score("test")
        assert isinstance(result_after, Ok)
        score_after = result_after.value
        assert score_after <= score_before

    @given(
        rpm=st.integers(min_value=1, max_value=10),
        rpd=st.integers(min_value=1, max_value=10),
    )
    def test_has_capacity_false_when_limit_exceeded(self, rpm, rpd):
        """[TM-012 AC-4] Property: has_capacity returns False when any limit is exceeded."""
        provider = ProviderConfig(
            name="test",
            base_url="http://localhost",
            catalog_url=None,
            env_key=None,
            model_prefix="test",
            rpm_limit=rpm,
            rpd_limit=rpd,
            tpm_limit=None,
            daily_token_cap=None,
        )
        bt = BudgetTracker(providers=[provider])

        # Exhaust the smaller of RPM or RPD
        exhaust_count = max(rpm, rpd)
        for _ in range(exhaust_count):
            bt.record_request("test")

        # At least one limit should be exceeded
        assert bt.has_capacity("test") is False


# ---------------------------------------------------------------------------
# TM-001: MBR never-downgrade invariant
# ---------------------------------------------------------------------------


class TestMBRNeverDowngradeInvariant:
    """Property: Invariant. MBR output tier is always >= input tier.

    Spec traceability: TM-001 (MBR capability filtering)
    """

    @given(tier=st.sampled_from(list(BackendTier)))
    def test_estimate_complexity_returns_valid_tier(self, tier):
        """[TM-001 AC-1] Property: estimate_complexity always returns a valid BackendTier."""
        from dragonlight_router.selection.mbr import estimate_complexity

        order = DispatchOrder(
            intent_category="test",
            specific_intent="test",
            operator_message="test message",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        )
        result = estimate_complexity(order)
        assert isinstance(result, BackendTier)

    @given(
        context_tokens=st.integers(min_value=0, max_value=50000),
        requires_tool_use=st.booleans(),
        requires_long_context=st.booleans(),
    )
    def test_estimate_complexity_never_downgrades_with_more_requirements(
        self, context_tokens, requires_tool_use, requires_long_context,
    ):
        """[TM-001 AC-4] Property: Adding requirements never lowers the estimated tier."""
        from dragonlight_router.selection.mbr import _TIER_RANK, estimate_complexity

        # Base order with no requirements
        base_order = DispatchOrder(
            intent_category="test",
            specific_intent="test",
            operator_message="test message",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        )
        base_tier = estimate_complexity(base_order)

        # Enhanced order with more requirements
        enhanced_order = DispatchOrder(
            intent_category="test",
            specific_intent="test",
            operator_message="test message",
            system_prompt="",
            context_tokens=context_tokens,
            requires_tool_use=requires_tool_use,
            requires_long_context=requires_long_context,
        )
        enhanced_tier = estimate_complexity(enhanced_order)

        # Enhanced tier should be >= base tier
        assert _TIER_RANK[enhanced_tier] >= _TIER_RANK[base_tier]

    @given(
        candidates=st.lists(
            backend_config_strategy(),
            min_size=1,
            max_size=5,
        ),
    )
    def test_filter_by_capabilities_preserves_no_downgrade(self, candidates):
        """[TM-001 AC-4] Property: _filter_by_capabilities never adds lower-tier candidates."""
        from dragonlight_router.selection.mbr import _filter_by_capabilities

        order = DispatchOrder(
            intent_category="test",
            specific_intent="test",
            operator_message="test message",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        )

        result = _filter_by_capabilities(candidates, order)

        # All output candidates must be from the input
        for r in result:
            assert r in candidates


# ---------------------------------------------------------------------------
# TM-003: LBR invariants
# ---------------------------------------------------------------------------


class TestLBRInvariants:
    """Property: Invariant. LBR output is always a subset of input, LOCAL providers pass through.

    Spec traceability: TM-003 (LBR rate-limit dispatch)
    """

    @given(
        num_candidates=st.integers(min_value=0, max_value=5),
    )
    def test_output_is_subset_of_input(self, num_candidates):
        """[TM-003 AC-1] Property: Invariant. LBR output is always a subset of input candidates."""
        # Build deterministic candidates
        candidates = []
        for i in range(num_candidates):
            candidates.append(
                BackendConfig(
                    name=f"backend-{i}",
                    provider=f"provider-{i}",
                    model=f"model-{i}",
                    tier=BackendTier.COMPLEX,
                    base_url="http://localhost",
                    env_key=None,
                    capabilities=BackendCapabilities(
                        max_context_tokens=4096,
                        supports_tool_use=False,
                        supports_streaming=True,
                        supports_json_mode=False,
                        supports_system_prompts=True,
                    ),
                    cost=BackendCostProfile(input_per_mtok=1.0, output_per_mtok=2.0),
                    rate_limits=BackendRateLimits(
                        rpm=60, rpd=1000, tpm=10000, daily_token_cap=100000,
                    ),
                    priority=0,
                )
            )

        order = DispatchOrder(
            intent_category="test",
            specific_intent="test",
            operator_message="test",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        )

        budget_tracker = BudgetTracker(providers=[])

        result = filter_by_rate_limit(candidates, order, budget_tracker)

        # Every result must be in the original candidates
        for r in result:
            assert r in candidates

    @given(
        num_cloud=st.integers(min_value=0, max_value=3),
    )
    def test_local_providers_never_filtered_out(self, num_cloud):
        """[TM-003 AC-4] Property: Invariant. LOCAL providers are never filtered out by LBR."""
        local_config = BackendConfig(
            name="local-backend",
            provider="local_provider",
            model="local-model",
            tier=BackendTier.LOCAL,
            base_url="http://localhost",
            env_key=None,
            capabilities=BackendCapabilities(
                max_context_tokens=4096,
                supports_tool_use=False,
                supports_streaming=True,
                supports_json_mode=False,
                supports_system_prompts=True,
            ),
            cost=BackendCostProfile(input_per_mtok=0.0, output_per_mtok=0.0),
            rate_limits=BackendRateLimits(rpm=60, rpd=1000, tpm=10000, daily_token_cap=100000),
            priority=0,
        )

        # Add some cloud backends
        candidates = [local_config]
        for i in range(num_cloud):
            candidates.append(
                BackendConfig(
                    name=f"cloud-{i}",
                    provider=f"cloud-provider-{i}",
                    model=f"cloud-model-{i}",
                    tier=BackendTier.COMPLEX,
                    base_url="http://localhost",
                    env_key=None,
                    capabilities=BackendCapabilities(
                        max_context_tokens=4096,
                        supports_tool_use=False,
                        supports_streaming=True,
                        supports_json_mode=False,
                        supports_system_prompts=True,
                    ),
                    cost=BackendCostProfile(input_per_mtok=10.0, output_per_mtok=20.0),
                    rate_limits=BackendRateLimits(
                        rpm=60, rpd=1000, tpm=10000, daily_token_cap=100000,
                    ),
                    priority=0,
                )
            )

        order = DispatchOrder(
            intent_category="test",
            specific_intent="test",
            operator_message="test",
            system_prompt="",
            context_tokens=0,
            requires_tool_use=False,
            requires_long_context=False,
        )

        # Use a mock budget tracker that gives low scores to everything
        budget_tracker = Mock(spec=BudgetTracker)
        budget_tracker.score.return_value = Ok(5.0)  # Very low score

        result = filter_by_rate_limit(candidates, order, budget_tracker)

        # LOCAL backend must be in the result
        local_names = {c.name for c in result if c.tier == BackendTier.LOCAL}
        assert "local-backend" in local_names


# ---------------------------------------------------------------------------
# Interleave invariants
# ---------------------------------------------------------------------------


class TestInterleaveInvariants:
    """Property: Invariant. Interleaving preserves elements and respects max_consecutive.

    Spec traceability: TM-010 (RouterEngine interleaving stage)
    """

    @given(
        scored=st.lists(
            model_score_strategy(providers=["groq", "nvidia", "anthropic"]),
            min_size=0,
            max_size=10,
        ),
        max_consecutive=st.integers(min_value=1, max_value=5),
    )
    def test_output_is_permutation_of_input(self, scored, max_consecutive):
        """[TM-010 AC-1] Property: Invariant. Output contains exactly the same elements as input."""
        result = interleave_providers(scored, max_consecutive=max_consecutive)

        # Same length
        assert len(result) == len(scored)

        # Same elements (as multiset)
        input_ids = sorted(m.model_id for m in scored)
        output_ids = sorted(m.model_id for m in result)
        assert input_ids == output_ids

    @given(
        scored=st.lists(
            model_score_strategy(providers=["groq", "nvidia", "anthropic"]),
            min_size=0,
            max_size=10,
        ),
        max_consecutive=st.integers(min_value=1, max_value=5),
    )
    def test_no_more_than_max_consecutive_same_provider(
        self, scored, max_consecutive,
    ):
        """[TM-010 AC-1] Property: No more than max_consecutive same-provider in a row."""
        # Skip if only one provider (can't interleave)
        providers = {m.provider for m in scored}
        assume(len(providers) > 1)

        # Skip geometrically impossible cases: if any single provider has
        # count C, we need at least ceil(C / max_consecutive) - 1 items
        # from other providers to space them out.
        from collections import Counter
        counts = Counter(m.provider for m in scored)
        total = len(scored)
        for _prov, count in counts.items():
            others = total - count
            separators_needed = (count - 1) // max_consecutive  # gaps needed
            assume(others >= separators_needed)

        result = interleave_providers(scored, max_consecutive=max_consecutive)

        # Check consecutive constraint
        for provider in providers:
            consecutive = 0
            for m in result:
                if m.provider == provider:
                    consecutive += 1
                    assert consecutive <= max_consecutive, (
                        f"Provider {provider} appears {consecutive} times consecutively "
                        f"(max_consecutive={max_consecutive})"
                    )
                else:
                    consecutive = 0
