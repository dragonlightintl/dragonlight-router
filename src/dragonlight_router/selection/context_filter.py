"""Context trust tier filtering for DIAN CECHT."""

from __future__ import annotations

from enum import Enum

import structlog

logger = structlog.get_logger(__name__)


class TrustTier(Enum):
    """Trust tiers for context filtering."""

    LOCAL = 1
    SIMPLE = 2
    MODERATE = 3
    COMPLEX = 4


class ProviderTrustTier(Enum):
    """Trust tiers for provider context filtering (DIAN CECHT)."""

    TRUSTED = 1
    SEMI_TRUSTED = 2
    UNTRUSTED = 3
    LOCAL = 4  # TIER 3-LOCAL: full context, no network egress


def filter_by_trust_tier(candidates: list[TrustTier], required_tier: TrustTier) -> list[TrustTier]:
    """Return candidates allowed for the required trust tier.

    Trust hierarchy:
        LOCAL trusts all tiers (LOCAL, HAIKU, SONNET, OPUS)
        HAIKU trusts HAIKU and above (HAIKU, SONNET, OPUS)
        SONNET trusts SONNET and above (SONNET, OPUS)
        OPUS trusts only OPUS

    Args:
        candidates: List of trust tier candidates to filter.
        required_tier: Minimum required trust tier.

    Returns:
        List of candidates that meet or exceed the required trust tier.
    """
    # Guard clauses
    assert isinstance(candidates, list), "candidates must be a list"
    assert isinstance(required_tier, TrustTier), "required_tier must be TrustTier enum"

    logger.debug(
        "filtering candidates by trust tier",
        candidate_count=len(candidates),
        required_tier=required_tier.name,
    )

    # LOCAL tier trusts all
    if required_tier == TrustTier.LOCAL:
        return candidates

    # Define minimum allowed tier index (higher number = higher trust)
    min_index = required_tier.value

    # Filter candidates where candidate.value >= min_index
    filtered = [c for c in candidates if c.value >= min_index]

    logger.debug(
        "filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered),
    )

    return filtered


def filter_context_for_provider(context: dict, provider_trust_tier: ProviderTrustTier) -> dict:
    """Filter context based on provider trust tier.

    Args:
        context: The context dictionary containing system, history, task, etc.
        provider_trust_tier: The trust tier of the provider.

    Returns:
        Filtered context dictionary.
    """
    logger.debug(
        "filtering context for provider",
        provider_trust_tier=provider_trust_tier.name,
        context_keys=list(context.keys()) if context else [],
    )

    # Make a shallow copy to avoid mutating the original
    filtered_context = context.copy()

    if provider_trust_tier == ProviderTrustTier.TRUSTED:
        # TRUSTED providers receive full system-level context (minus PII, already absent)
        logger.debug("trusted provider: passing through full context")
        return filtered_context

    elif provider_trust_tier == ProviderTrustTier.SEMI_TRUSTED:
        # SEMI_TRUSTED providers receive context without behavioral rules or persona names
        # and limited history
        logger.debug("semi-trusted provider: removing behavioral rules, replacing persona names, limiting history")
        system = filtered_context.get("system", {})
        if isinstance(system, dict):
            # Remove behavioral rules
            system = {k: v for k, v in system.items() if k not in ("behavioral_rules", "behavioral\\s*rules")}
            # Replace persona names with placeholder
            if "persona" in system:
                system["persona"] = "[REDACTED PERSONA]"
            # Also replace any nested persona fields (defensive)
            for k in list(system.keys()):
                if "persona" in k.lower():
                    system[k] = "[REDACTED PERSONA]"
            filtered_context["system"] = system
        # Limit history to last 3 turns
        history = filtered_context.get("history", [])
        if isinstance(history, list) and len(history) > 3:
            filtered_context["history"] = history[-3:]
        logger.debug(
            "semi-trusted filtering applied",
            system_keys=list(system.keys()) if isinstance(system, dict) else [],
            history_length=len(filtered_context.get("history", [])),
        )
        return filtered_context

    elif provider_trust_tier == ProviderTrustTier.UNTRUSTED:
        # UNTRUSTED providers receive task-specific instruction only
        logger.debug("untrusted provider: returning task instruction only")
        task = filtered_context.get("task", "")
        return {"task": task} if task else {}

    elif provider_trust_tier == ProviderTrustTier.LOCAL:
        # LOCAL providers receive full context (no network egress risk)
        logger.debug("local provider: passing through full context (no network egress)")
        return filtered_context

    else:
        logger.warning("unknown provider trust tier, returning empty context", tier=provider_trust_tier)
        return {}