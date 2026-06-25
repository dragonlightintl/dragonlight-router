"""Domain-specific Hypothesis strategies for core dragonlight-router types.

Provides reusable strategies for generating valid instances of:
  - BackendCapabilities
  - BackendCostProfile
  - BackendRateLimits
  - BackendConfig
  - DispatchOrder
  - ScoredCandidate
  - ProviderConfig
  - ModelScore
  - SpectrographScore
  - ModelSpectrographProfile

All strategies respect the field constraints and value ranges defined in
src/dragonlight_router/core/types.py.
"""

from __future__ import annotations

from hypothesis import strategies as st

from dragonlight_router.core.types import (
    BackendCapabilities,
    BackendConfig,
    BackendCostProfile,
    BackendRateLimits,
    BackendTier,
    DispatchOrder,
    ModelScore,
    ModelSpectrographProfile,
    ProviderConfig,
    ScoredCandidate,
    SpectrographScore,
)

# ---------------------------------------------------------------------------
# Leaf strategies (sub-components)
# ---------------------------------------------------------------------------

_IDENTIFIER_ALPHABET = st.characters(whitelist_categories=("Ll", "Nd"))


def backend_capabilities_strategy() -> st.SearchStrategy[BackendCapabilities]:
    """Strategy for generating BackendCapabilities instances."""
    return st.builds(
        BackendCapabilities,
        max_context_tokens=st.integers(min_value=1024, max_value=256000),
        supports_tool_use=st.booleans(),
        supports_streaming=st.booleans(),
        supports_json_mode=st.booleans(),
        supports_system_prompts=st.booleans(),
    )


def backend_cost_profile_strategy() -> st.SearchStrategy[BackendCostProfile]:
    """Strategy for generating BackendCostProfile instances."""
    return st.builds(
        BackendCostProfile,
        input_per_mtok=st.floats(min_value=0.0, max_value=100.0),
        output_per_mtok=st.floats(min_value=0.0, max_value=100.0),
        cache_read_per_mtok=st.floats(min_value=0.0, max_value=100.0),
        cache_write_per_mtok=st.floats(min_value=0.0, max_value=100.0),
    )


def backend_rate_limits_strategy() -> st.SearchStrategy[BackendRateLimits]:
    """Strategy for generating BackendRateLimits instances."""
    return st.builds(
        BackendRateLimits,
        rpm=st.integers(min_value=1, max_value=1000),
        rpd=st.integers(min_value=1, max_value=100000),
        tpm=st.integers(min_value=1, max_value=1000000),
        daily_token_cap=st.integers(min_value=0, max_value=10000000),
    )


# ---------------------------------------------------------------------------
# Composite strategies
# ---------------------------------------------------------------------------


def backend_config_strategy(
    tier: BackendTier | None = None,
    provider: str | None = None,
) -> st.SearchStrategy[BackendConfig]:
    """Strategy for generating BackendConfig instances.

    Args:
        tier: Fix the tier to a specific value, or None for random.
        provider: Fix the provider to a specific value, or None for random.
    """
    tier_st = st.just(tier) if tier else st.sampled_from(list(BackendTier))
    provider_st = (
        st.just(provider)
        if provider
        else st.sampled_from(["groq", "nvidia", "anthropic", "openai"])
    )
    return st.builds(
        BackendConfig,
        name=st.text(alphabet=_IDENTIFIER_ALPHABET, min_size=1, max_size=20),
        provider=provider_st,
        model=st.text(alphabet=_IDENTIFIER_ALPHABET, min_size=1, max_size=20),
        tier=tier_st,
        base_url=st.just("http://localhost"),
        env_key=st.none(),
        capabilities=backend_capabilities_strategy(),
        cost=backend_cost_profile_strategy(),
        rate_limits=backend_rate_limits_strategy(),
        priority=st.integers(min_value=0, max_value=100),
    )


def dispatch_order_strategy() -> st.SearchStrategy[DispatchOrder]:
    """Strategy for generating DispatchOrder instances.

    Covers valid intent categories, arbitrary operator messages, and
    optional capability requirements.
    """
    return st.builds(
        DispatchOrder,
        intent_category=st.sampled_from(
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
        specific_intent=st.text(alphabet=_IDENTIFIER_ALPHABET, min_size=1, max_size=20),
        operator_message=st.text(min_size=0, max_size=200),
        system_prompt=st.text(min_size=0, max_size=200),
        context_tokens=st.integers(min_value=0, max_value=200000),
        requires_tool_use=st.booleans(),
        requires_long_context=st.booleans(),
        persona=st.one_of(st.none(), st.text(min_size=1, max_size=20)),
        fallback_policy=st.sampled_from(["allow", "deny", "same_tier"]),
        model=st.none(),
    )


def scored_candidate_strategy(
    tier: BackendTier | None = None,
) -> st.SearchStrategy[ScoredCandidate]:
    """Strategy for generating ScoredCandidate instances.

    Args:
        tier: Fix the tier of the underlying BackendConfig, or None for random.
    """
    return st.builds(
        ScoredCandidate,
        config=backend_config_strategy(tier=tier),
        score=st.floats(min_value=0.0, max_value=100.0),
    )


def provider_config_strategy(
    rpm_range: tuple[int, int] = (1, 1000),
    rpd_range: tuple[int, int] = (1, 10000),
) -> st.SearchStrategy[ProviderConfig]:
    """Strategy for generating ProviderConfig instances."""
    return st.builds(
        ProviderConfig,
        name=st.text(alphabet=_IDENTIFIER_ALPHABET, min_size=1, max_size=10),
        base_url=st.just("http://localhost"),
        catalog_url=st.none(),
        env_key=st.none(),
        model_prefix=st.text(alphabet=_IDENTIFIER_ALPHABET, min_size=1, max_size=5),
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
        model_id=st.text(alphabet=_IDENTIFIER_ALPHABET, min_size=1, max_size=20),
        provider=provider_st,
        rank=st.integers(min_value=0, max_value=100),
        budget_score=st.floats(min_value=0.0, max_value=100.0),
        health_score=st.floats(min_value=0.0, max_value=100.0),
        composite=st.floats(min_value=0.0, max_value=100.0),
    )


def spectrograph_score_strategy() -> st.SearchStrategy[SpectrographScore]:
    """Strategy for generating SpectrographScore instances.

    score and confidence are in [0.0, 1.0]; sample_count is non-negative.
    """
    return st.builds(
        SpectrographScore,
        score=st.floats(min_value=0.0, max_value=1.0),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        sample_count=st.integers(min_value=0, max_value=1000),
    )


def model_spectrograph_profile_strategy(
    model_id: str = "test-model",
) -> st.SearchStrategy[ModelSpectrographProfile]:
    """Strategy for generating ModelSpectrographProfile with scores for all IBR dimensions."""
    from dragonlight_router.core.types import (
        IBR_DOMAINS,
        IBR_QUALITY_SPEED,
        IBR_TASK_TYPES,
    )

    task_scores_st = st.fixed_dictionaries(
        {tt: spectrograph_score_strategy() for tt in sorted(IBR_TASK_TYPES)},
    )
    domain_scores_st = st.fixed_dictionaries(
        {d: spectrograph_score_strategy() for d in sorted(IBR_DOMAINS)},
    )
    qs_scores_st = st.fixed_dictionaries(
        {qs: spectrograph_score_strategy() for qs in sorted(IBR_QUALITY_SPEED)},
    )
    return st.builds(
        ModelSpectrographProfile,
        model_id=st.just(model_id),
        version=st.just(1),
        updated_at=st.just("2026-01-01T00:00:00Z"),
        task_scores=task_scores_st,
        domain_scores=domain_scores_st,
        qs_scores=qs_scores_st,
    )
