"""Property-based tests for dragonlight-router invariants.

Spec traceability:
  - TM-007: Scoring weights invariants
  - TM-012: BudgetTracker invariants
  - TM-001: MBR never-downgrade invariant
  - TM-003: LBR rate-limit dispatch invariants
  - Interleave permutation invariants
  - TS-001: Cascade dispatch, health scoring, model selection, budget monotonicity
  - IBR-TEST-02: IBR scoring, confidence gating, and degradation invariants

Uses Hypothesis to verify that invariants hold across the entire input space,
not just hand-picked examples. Property categories follow the taxonomy from
dragonlight-property-based-testing-strategy.md.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_NEUTRAL_FLAVOR,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    ClassifiedIntent,
    DispatchOrder,
    FlavorScore,
    ModelFlavorProfile,
    ModelScore,
    Ok,
    ProviderConfig,
)
from dragonlight_router.health.tracker import HealthTracker
from dragonlight_router.selection.flavor import (
    compute_flavor_match,
    compute_flavor_scores,
    should_apply_flavor_match,
)
from dragonlight_router.selection.interleave import interleave_providers
from dragonlight_router.selection.lbr import filter_by_rate_limit
from dragonlight_router.selection.scoring import (
    ScoringWeightsConfig,
    compute_budget_score,
    compute_composite_score,
    compute_health_score,
)

pytestmark = [pytest.mark.unit, pytest.mark.property]


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
            error_count,
            circuit_open=True,
            last_success_age_s=last_success_age,
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
        self,
        rpm,
        rpd,
        tpm,
        daily_token_cap,
        num_requests,
        tokens_per_request,
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
        self,
        context_tokens,
        requires_tool_use,
        requires_long_context,
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
                        rpm=60,
                        rpd=1000,
                        tpm=10000,
                        daily_token_cap=100000,
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
                        rpm=60,
                        rpd=1000,
                        tpm=10000,
                        daily_token_cap=100000,
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
        self,
        scored,
        max_consecutive,
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


# ---------------------------------------------------------------------------
# TS-001: Health scoring invariant
# ---------------------------------------------------------------------------


class TestHealthScoringInvariant:
    """Property: Invariant. HealthTracker.score() is always in [0, 100] for any
    sequence of record_success/record_error calls.

    Spec traceability: TS-001 (Health scoring invariant)
    """

    @given(error_count=st.integers(min_value=0, max_value=100))
    def test_health_score_always_in_range(self, error_count: int) -> None:
        """[TS-001 AC-1] For any number of errors, health score is in [0, 100]."""
        tracker = HealthTracker()
        model_id = "test/model-a"
        for _ in range(error_count):
            tracker.record_error(model_id)
        result = tracker.score(model_id)
        assert isinstance(result, Ok)
        assert 0.0 <= result.value <= 100.0

    @given(
        successes=st.integers(min_value=0, max_value=20),
        errors=st.integers(min_value=0, max_value=20),
    )
    def test_health_score_bounded_after_mixed_operations(
        self,
        successes: int,
        errors: int,
    ) -> None:
        """[TS-001 AC-2] Any interleaving of success/error calls yields score in [0, 100]."""
        tracker = HealthTracker()
        model_id = "test/model-mixed"
        for _ in range(successes):
            tracker.record_success(model_id, latency_ms=50.0)
        for _ in range(errors):
            tracker.record_error(model_id)
        result = tracker.score(model_id)
        assert isinstance(result, Ok)
        assert 0.0 <= result.value <= 100.0


# ---------------------------------------------------------------------------
# TS-001: Model selection invariant
# ---------------------------------------------------------------------------


class TestModelSelectionInvariant:
    """Property: Invariant. select_models() always returns model IDs that exist
    in the role matrix for the requested role.

    Spec traceability: TS-001 (Model selection idempotence)
    """

    @given(
        call_count=st.integers(min_value=1, max_value=5),
    )
    @settings(deadline=5000)
    def test_select_models_returns_from_role_matrix(self, call_count: int) -> None:
        """[TS-001 AC-3] Every model_id from select_models exists in the role matrix."""
        import json
        import tempfile
        from pathlib import Path

        import yaml

        from dragonlight_router.catalog.cache import CatalogCache
        from dragonlight_router.core.types import CatalogEntry
        from dragonlight_router.router import RouterEngine

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            state_dir = tmp_path / "state"
            state_dir.mkdir()

            config = {
                "state_dir": str(state_dir),
                "catalog_ttl_hours": 24,
                "default_top_n": 12,
                "max_consecutive_same_provider": 2,
                "providers": [
                    {
                        "name": "groq",
                        "base_url": "https://api.groq.com/openai/v1",
                        "model_prefix": "groq_",
                        "rate_limits": {"rpm": 30, "rpd": 14400},
                    },
                    {
                        "name": "nvidia",
                        "base_url": "https://integrate.api.nvidia.com/v1",
                        "model_prefix": "nvidia_",
                        "rate_limits": {"rpm": 60, "rpd": 5000},
                    },
                ],
            }
            config_path = tmp_path / "router.yaml"
            config_path.write_text(yaml.dump(config))

            matrix = {
                "coding": {
                    "groq_llama70b": 90,
                    "nvidia_nemotron": 85,
                    "groq_mixtral": 75,
                },
            }
            (state_dir / "model_role_matrix.json").write_text(json.dumps(matrix))

            catalog = {
                "groq": [
                    CatalogEntry(model_id="groq_llama70b", provider="groq"),
                    CatalogEntry(model_id="groq_mixtral", provider="groq"),
                ],
                "nvidia": [
                    CatalogEntry(model_id="nvidia_nemotron", provider="nvidia"),
                ],
            }
            cache = CatalogCache(
                cache_path=state_dir / "provider_catalog.json",
                ttl_hours=24,
            )
            cache.set(catalog)

            engine = RouterEngine(config_path=config_path)
            valid_model_ids = set(matrix["coding"].keys())

            for _ in range(call_count):
                result = engine.select_models("coding")
                for model_id in result:
                    assert model_id in valid_model_ids, (
                        f"{model_id} not in role matrix: {valid_model_ids}"
                    )


# ---------------------------------------------------------------------------
# TS-001: Budget scoring monotonicity
# ---------------------------------------------------------------------------


class TestBudgetScoringMonotonicity:
    """Property: Invariant. As requests are recorded against a provider, the
    budget score monotonically decreases (or stays the same).

    Spec traceability: TS-001 (Budget scoring monotonicity)
    """

    @given(
        rpm=st.integers(min_value=1, max_value=100),
        rpd=st.one_of(st.none(), st.integers(min_value=1, max_value=1000)),
        num_requests=st.integers(min_value=1, max_value=20),
        tokens_per_request=st.integers(min_value=0, max_value=100),
    )
    def test_budget_score_monotonically_decreases(
        self,
        rpm: int,
        rpd: int | None,
        num_requests: int,
        tokens_per_request: int,
    ) -> None:
        """[TS-001 AC-4] Budget score never increases as more requests are recorded."""
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

        previous_score_result = bt.score("test")
        assert isinstance(previous_score_result, Ok)
        previous_score = previous_score_result.value

        for _ in range(num_requests):
            bt.record_request("test", tokens_used=tokens_per_request)
            current_result = bt.score("test")
            assert isinstance(current_result, Ok)
            current_score = current_result.value
            assert current_score <= previous_score, (
                f"Budget score increased from {previous_score} to {current_score}"
            )
            previous_score = current_score


# ---------------------------------------------------------------------------
# TS-001: Cascade dispatch invariant
# ---------------------------------------------------------------------------


class TestCascadeDispatchInvariant:
    """Property: Invariant. For any valid backend list, the cascade dispatch's
    candidate selection always returns models from the input list or an error.

    Since full cascade dispatch requires async I/O and real adapters, this
    property is tested at the MBR filter + LBR filter level, which are the
    synchronous core of the dispatch pipeline.

    Spec traceability: TS-001 (Cascade dispatch invariant)
    """

    @given(
        num_backends=st.integers(min_value=1, max_value=5),
        tier=st.sampled_from(list(BackendTier)),
    )
    def test_cascade_candidates_always_subset_of_input(
        self,
        num_backends: int,
        tier: BackendTier,
    ) -> None:
        """[TS-001 AC-5] Filtered candidates are always a subset of the input backends."""
        from dragonlight_router.selection.mbr import _filter_by_capabilities

        candidates = []
        for i in range(num_backends):
            candidates.append(
                BackendConfig(
                    name=f"backend-{i}",
                    provider=f"provider-{i}",
                    model=f"model-{i}",
                    tier=tier,
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
                        rpm=60,
                        rpd=1000,
                        tpm=10000,
                        daily_token_cap=100000,
                    ),
                    priority=0,
                ),
            )

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

        # Every result must come from the input list
        input_names = {c.name for c in candidates}
        for r in result:
            assert r.name in input_names, f"Result {r.name} not in input candidates: {input_names}"

    @given(
        num_backends=st.integers(min_value=1, max_value=5),
    )
    def test_lbr_filter_returns_subset_or_empty(
        self,
        num_backends: int,
    ) -> None:
        """[TS-001 AC-6] LBR filter output is always a subset of input, never None."""
        candidates = []
        for i in range(num_backends):
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
                        rpm=60,
                        rpd=1000,
                        tpm=10000,
                        daily_token_cap=100000,
                    ),
                    priority=0,
                ),
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

        # Result is never None
        assert result is not None
        # Result is a list (could be empty)
        assert isinstance(result, list)
        # Every item in result is from the input
        input_names = {c.name for c in candidates}
        for r in result:
            assert r.name in input_names


# ---------------------------------------------------------------------------
# IBR-TEST-02: Intent Based Router property tests
# ---------------------------------------------------------------------------

# --- IBR strategies ---


def _ibr_flavor_score_strategy() -> st.SearchStrategy[FlavorScore]:
    """Strategy for generating FlavorScore instances."""
    return st.builds(
        FlavorScore,
        score=st.floats(min_value=0.0, max_value=1.0),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        sample_count=st.integers(min_value=0, max_value=1000),
    )


def _ibr_classified_intent_strategy() -> st.SearchStrategy[ClassifiedIntent]:
    """Strategy for generating valid ClassifiedIntent instances."""
    return st.builds(
        ClassifiedIntent,
        task_type=st.sampled_from(sorted(IBR_TASK_TYPES)),
        domain=st.sampled_from(sorted(IBR_DOMAINS)),
        quality_speed=st.sampled_from(sorted(IBR_QUALITY_SPEED)),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        latency_ms=st.floats(min_value=0.0, max_value=200.0),
        from_cache=st.booleans(),
    )


def _ibr_model_flavor_profile_strategy(
    model_id: str = "test-model",
) -> st.SearchStrategy[ModelFlavorProfile]:
    """Strategy for generating ModelFlavorProfile with scores for all dimensions."""
    task_scores_st = st.fixed_dictionaries(
        {tt: _ibr_flavor_score_strategy() for tt in sorted(IBR_TASK_TYPES)},
    )
    domain_scores_st = st.fixed_dictionaries(
        {d: _ibr_flavor_score_strategy() for d in sorted(IBR_DOMAINS)},
    )
    qs_scores_st = st.fixed_dictionaries(
        {qs: _ibr_flavor_score_strategy() for qs in sorted(IBR_QUALITY_SPEED)},
    )
    return st.builds(
        ModelFlavorProfile,
        model_id=st.just(model_id),
        version=st.just(1),
        updated_at=st.just("2026-01-01T00:00:00Z"),
        task_scores=task_scores_st,
        domain_scores=domain_scores_st,
        qs_scores=qs_scores_st,
    )


def valid_scoring_weights_with_flavor() -> st.SearchStrategy[ScoringWeightsConfig]:
    """Strategy for generating valid ScoringWeightsConfig with flavor_match.

    Generates 6 non-negative weights that sum to 1.0.
    """

    @st.composite
    def _build(draw: st.DrawFn) -> ScoringWeightsConfig:
        flavor_match = draw(st.floats(min_value=0.0, max_value=0.30))
        remainder = 1.0 - flavor_match
        # Distribute remainder across 5 weights proportionally
        raw = [draw(st.floats(min_value=0.01, max_value=1.0)) for _ in range(5)]
        total_raw = sum(raw)
        scaled = [r / total_raw * remainder for r in raw]
        return ScoringWeightsConfig(
            cost=scaled[0],
            latency=scaled[1],
            priority=scaled[2],
            queue=scaled[3],
            health=scaled[4],
            flavor_match=flavor_match,
        )

    return _build()


class TestIBRFlavorMatchInvariant:
    """Property: Invariant. Flavor match score is always in [0.0, 1.0].

    Spec traceability: IBR-TEST-02 (Scoring invariant)
    """

    @given(
        task_score=st.floats(min_value=0.0, max_value=1.0),
        domain_score=st.floats(min_value=0.0, max_value=1.0),
        qs_score=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_flavor_match_always_in_unit_interval(
        self,
        task_score: float,
        domain_score: float,
        qs_score: float,
    ) -> None:
        """[IBR-TEST-02 AC-1] compute_flavor_match always returns [0.0, 1.0]."""
        intent = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=0.9,
            latency_ms=10.0,
            from_cache=False,
        )
        profile = ModelFlavorProfile(
            model_id="test",
            version=1,
            updated_at="2026-01-01T00:00:00Z",
            task_scores=dict.fromkeys(IBR_TASK_TYPES, IBR_NEUTRAL_FLAVOR)
            | {"analysis": FlavorScore(score=task_score, confidence=1.0, sample_count=0)},
            domain_scores=dict.fromkeys(IBR_DOMAINS, IBR_NEUTRAL_FLAVOR)
            | {"code": FlavorScore(score=domain_score, confidence=1.0, sample_count=0)},
            qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, IBR_NEUTRAL_FLAVOR)
            | {"balanced": FlavorScore(score=qs_score, confidence=1.0, sample_count=0)},
        )
        result = compute_flavor_match(intent, profile)
        assert 0.0 <= result <= 1.0

    @given(
        intent=_ibr_classified_intent_strategy(),
        profile=_ibr_model_flavor_profile_strategy(),
    )
    def test_flavor_match_bounded_for_arbitrary_profiles(
        self,
        intent: ClassifiedIntent,
        profile: ModelFlavorProfile,
    ) -> None:
        """[IBR-TEST-02 AC-1] Flavor match is in [0.0, 1.0] for any valid intent+profile."""
        result = compute_flavor_match(intent, profile)
        assert 0.0 <= result <= 1.0


class TestIBRScoringWeightsInvariant:
    """Property: Invariant. ScoringWeightsConfig including flavor_match sums to 1.0.

    Spec traceability: IBR-TEST-02 (Weight sum invariant)
    """

    @given(weights=valid_scoring_weights_with_flavor())
    def test_scoring_weights_with_flavor_sum_to_one(
        self,
        weights: ScoringWeightsConfig,
    ) -> None:
        """[IBR-TEST-02 AC-2] ScoringWeightsConfig always sums to 1.0 including flavor_match."""
        total = (
            weights.cost
            + weights.latency
            + weights.priority
            + weights.queue
            + weights.health
            + weights.flavor_match
        )
        assert abs(total - 1.0) < 1e-6


class TestIBRConfidenceGatingInvariant:
    """Property: Invariant. Low confidence always prevents flavor match application.

    Spec traceability: IBR-TEST-02 (Confidence gating invariant)
    """

    @given(confidence=st.floats(min_value=0.0, max_value=1.0))
    def test_confidence_gating_monotonic(self, confidence: float) -> None:
        """[IBR-TEST-02 AC-3] Higher confidence is more likely to pass gating."""
        intent_low = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=0.0,
            latency_ms=10.0,
            from_cache=False,
        )
        intent_high = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=1.0,
            latency_ms=10.0,
            from_cache=False,
        )
        # Full-confidence profile so profile threshold doesn't gate
        full_fs = FlavorScore(score=0.8, confidence=1.0, sample_count=10)
        profile = ModelFlavorProfile(
            model_id="test",
            version=1,
            updated_at="2026-01-01T00:00:00Z",
            task_scores=dict.fromkeys(IBR_TASK_TYPES, full_fs),
            domain_scores=dict.fromkeys(IBR_DOMAINS, full_fs),
            qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, full_fs),
        )

        result_low = should_apply_flavor_match(
            intent_low,
            profile,
            confidence_threshold=confidence,
        )
        result_high = should_apply_flavor_match(
            intent_high,
            profile,
            confidence_threshold=confidence,
        )

        # If low confidence passes, high must also pass (monotonic)
        if result_low:
            assert result_high

    @given(confidence=st.floats(min_value=0.0, max_value=0.59))
    def test_below_default_threshold_always_gated(self, confidence: float) -> None:
        """[IBR-TEST-02 AC-3] Confidence below 0.6 default is always gated."""
        intent = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=confidence,
            latency_ms=10.0,
            from_cache=False,
        )
        full_fs = FlavorScore(score=0.8, confidence=1.0, sample_count=10)
        profile = ModelFlavorProfile(
            model_id="test",
            version=1,
            updated_at="2026-01-01T00:00:00Z",
            task_scores=dict.fromkeys(IBR_TASK_TYPES, full_fs),
            domain_scores=dict.fromkeys(IBR_DOMAINS, full_fs),
            qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, full_fs),
        )
        assert should_apply_flavor_match(intent, profile) is False


class TestIBRDegradationInvariant:
    """Property: Invariant. When classification returns None, behavior is identical to IBR-disabled.

    Spec traceability: IBR-TEST-02 (Degradation invariant)
    """

    @given(
        model_ids=st.lists(
            st.text(
                alphabet=st.characters(whitelist_categories=("Ll", "Nd")),
                min_size=1,
                max_size=10,
            ),
            min_size=1,
            max_size=5,
        ),
    )
    def test_ibr_degradation_invariant(self, model_ids: list[str]) -> None:
        """[IBR-TEST-02 AC-4] None intent produces empty flavor scores (IBR inactive)."""
        profiles = {
            mid: ModelFlavorProfile(
                model_id=mid,
                version=1,
                updated_at="2026-01-01T00:00:00Z",
                task_scores=dict.fromkeys(IBR_TASK_TYPES, IBR_NEUTRAL_FLAVOR),
                domain_scores=dict.fromkeys(IBR_DOMAINS, IBR_NEUTRAL_FLAVOR),
                qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, IBR_NEUTRAL_FLAVOR),
            )
            for mid in model_ids
        }
        # None intent → empty scores → IBR has no effect
        scores = compute_flavor_scores(None, profiles, model_ids)
        assert scores == {}

    def test_none_intent_gating_always_false(self) -> None:
        """[IBR-TEST-02 AC-4] should_apply_flavor_match always False for None intent."""
        full_fs = FlavorScore(score=0.8, confidence=1.0, sample_count=10)
        profile = ModelFlavorProfile(
            model_id="test",
            version=1,
            updated_at="2026-01-01T00:00:00Z",
            task_scores=dict.fromkeys(IBR_TASK_TYPES, full_fs),
            domain_scores=dict.fromkeys(IBR_DOMAINS, full_fs),
            qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, full_fs),
        )
        assert should_apply_flavor_match(None, profile) is False


class TestIBRFlavorMatchMonotonicity:
    """Property: Invariant. Higher scores produce higher flavor match, all else equal.

    Spec traceability: IBR-TEST-02 (Monotonicity)
    """

    @given(
        low_score=st.floats(min_value=0.0, max_value=0.49),
        high_score=st.floats(min_value=0.51, max_value=1.0),
    )
    def test_higher_task_score_higher_match(
        self,
        low_score: float,
        high_score: float,
    ) -> None:
        """[IBR-TEST-02 AC-5] Higher task_score produces higher flavor_match, all else equal."""
        intent = ClassifiedIntent(
            task_type="analysis",
            domain="code",
            quality_speed="balanced",
            confidence=0.9,
            latency_ms=10.0,
            from_cache=False,
        )

        def _profile(task_score: float) -> ModelFlavorProfile:
            return ModelFlavorProfile(
                model_id="test",
                version=1,
                updated_at="2026-01-01T00:00:00Z",
                task_scores=dict.fromkeys(IBR_TASK_TYPES, IBR_NEUTRAL_FLAVOR)
                | {"analysis": FlavorScore(score=task_score, confidence=1.0, sample_count=0)},
                domain_scores=dict.fromkeys(IBR_DOMAINS, IBR_NEUTRAL_FLAVOR),
                qs_scores=dict.fromkeys(IBR_QUALITY_SPEED, IBR_NEUTRAL_FLAVOR),
            )

        low_match = compute_flavor_match(intent, _profile(low_score))
        high_match = compute_flavor_match(intent, _profile(high_score))
        assert high_match > low_match
