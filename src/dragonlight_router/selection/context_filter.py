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
    assert isinstance(candidates, list), "candidates must be a list"
    assert isinstance(required_tier, TrustTier), "required_tier must be TrustTier enum"

    logger.debug(
        "filtering candidates by trust tier",
        candidate_count=len(candidates),
        required_tier=required_tier.name,
    )

    if required_tier == TrustTier.LOCAL:
        return candidates

    min_index = required_tier.value
    filtered = [c for c in candidates if c.value >= min_index]

    logger.debug(
        "filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered),
    )

    assert len(filtered) <= len(candidates), "filtered count must not exceed original"
    return filtered


def filter_context_for_provider(context: dict, provider_trust_tier: ProviderTrustTier) -> dict:
    """Filter context based on provider trust tier.

    Args:
        context: The context dictionary containing system, history, task, etc.
        provider_trust_tier: The trust tier of the provider.

    Returns:
        Filtered context dictionary.
    """
    assert isinstance(context, dict), "context must be a dict"
    assert isinstance(provider_trust_tier, ProviderTrustTier), "provider_trust_tier must be ProviderTrustTier"

    logger.debug(
        "filtering context for provider",
        provider_trust_tier=provider_trust_tier.name,
        context_keys=list(context.keys()) if context else [],
    )

    filtered_context = context.copy()

    dispatch = {
        ProviderTrustTier.TRUSTED: _filter_trusted,
        ProviderTrustTier.SEMI_TRUSTED: _filter_semi_trusted,
        ProviderTrustTier.UNTRUSTED: _filter_untrusted,
        ProviderTrustTier.LOCAL: _filter_local,
    }

    handler = dispatch.get(provider_trust_tier)
    if handler is None:
        logger.warning("unknown provider trust tier, returning empty context", tier=provider_trust_tier)
        return {}

    result = handler(filtered_context)
    assert isinstance(result, dict), "filter handler must return a dict"
    return result


def _filter_trusted(context: dict) -> dict:
    """TRUSTED providers receive full system-level context (minus PII, already absent)."""
    logger.debug("trusted provider: passing through full context")
    return context


def _filter_semi_trusted(context: dict) -> dict:
    """SEMI_TRUSTED: remove behavioral rules, redact persona names, limit history."""
    assert isinstance(context, dict), "context must be a dict"
    assert "task" not in context or isinstance(context.get("task"), str), "task must be a string if present"

    logger.debug("semi-trusted provider: removing behavioral rules, replacing persona names, limiting history")

    context = _redact_system_fields(context)
    context = _limit_history(context, max_turns=3)

    system = context.get("system", {})
    logger.debug(
        "semi-trusted filtering applied",
        system_keys=list(system.keys()) if isinstance(system, dict) else [],
        history_length=len(context.get("history", [])),
    )
    return context


def _redact_system_fields(context: dict) -> dict:
    """Remove behavioral rules and redact persona names from system context."""
    assert isinstance(context, dict), "context must be a dict"
    assert "system" not in context or isinstance(context.get("system"), dict), "system must be a dict if present"

    system = context.get("system", {})
    system = {k: v for k, v in system.items() if k not in ("behavioral_rules", "behavioral\\s*rules")}
    system = _redact_persona_fields(system)
    context["system"] = system
    return context


def _redact_persona_fields(system: dict) -> dict:
    """Replace persona-related fields with redaction placeholder."""
    assert isinstance(system, dict), "system must be a dict"

    for k in list(system.keys()):
        if "persona" in k.lower():
            system[k] = "[REDACTED PERSONA]"

    return system


def _limit_history(context: dict, max_turns: int) -> dict:
    """Limit conversation history to the most recent N turns."""
    assert isinstance(context, dict), "context must be a dict"
    assert max_turns > 0, "max_turns must be positive"

    history = context.get("history", [])
    if isinstance(history, list) and len(history) > max_turns:
        context["history"] = history[-max_turns:]
    return context


def _filter_untrusted(context: dict) -> dict:
    """UNTRUSTED providers receive task-specific instruction only."""
    logger.debug("untrusted provider: returning task instruction only")
    task = context.get("task", "")
    return {"task": task} if task else {}


def _filter_local(context: dict) -> dict:
    """LOCAL providers receive full context (no network egress risk)."""
    logger.debug("local provider: passing through full context (no network egress)")
    return context
