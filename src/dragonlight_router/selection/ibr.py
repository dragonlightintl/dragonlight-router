"""IBR orchestration — intent classification + flavor matching pipeline stage.

Wires classify_intent() and FlavorProfileLoader into a single async stage
that sits between MBR and CBR in the dispatch cascade.  When IBR is disabled
or classification fails, returns an inactive result so the cascade degrades
transparently to v0.3.0 behavior (IBR-SYS-02, IBR-SYS-03).

Spec reference: intent-based-router-v0.1.0-spec.md sections 4–5.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

import structlog

from dragonlight_router.config.schema import IntentClassificationConfig
from dragonlight_router.core.types import (
    BackendConfig,
    ClassifiedIntent,
    DispatchOrder,
    GenerativeBackend,
    ModelFlavorProfile,
)
from dragonlight_router.selection.classifier import classify_intent
from dragonlight_router.selection.flavor import (
    FlavorProfileLoader,
    compute_flavor_scores,
    get_profile_for_model,
    should_apply_flavor_match,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# IBRResult — frozen dataclass carrying stage output (IBR-DATA-01)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class IBRResult:
    """Output of the IBR pipeline stage.

    classified_intent: the classification, or None if skipped/failed.
    flavor_scores: model_id -> flavor_match score (empty when inactive).
    ibr_active: True only when valid scores were produced and gating passed.
    """

    classified_intent: ClassifiedIntent | None
    flavor_scores: dict[str, float]
    ibr_active: bool


# Singleton inactive result — avoids re-creating on every disabled path.
_INACTIVE_RESULT = IBRResult(
    classified_intent=None,
    flavor_scores={},
    ibr_active=False,
)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# DEVIATION CS-PARAM-001: run_ibr_stage takes 5 params — dataclass grouping would break API.
async def run_ibr_stage(
    order: DispatchOrder,
    candidates: list[BackendConfig],
    ibr_config: IntentClassificationConfig,
    flavor_loader: FlavorProfileLoader,
    classification_adapter: GenerativeBackend | None,
) -> IBRResult:
    """Run the IBR pipeline stage: classify intent and compute flavor scores.

    Returns an inactive IBRResult when IBR is disabled, no adapter is
    available, or classification fails.  Never raises — all errors are
    logged and degraded to v0.3.0 behavior (IBR-SYS-03).
    """
    assert isinstance(order, DispatchOrder), "order must be a DispatchOrder"
    assert isinstance(candidates, list), "candidates must be a list"

    if not ibr_config.enabled or classification_adapter is None:
        return _INACTIVE_RESULT

    try:
        return await _execute_ibr(
            order, candidates, ibr_config, flavor_loader, classification_adapter,
        )
    except (KeyError, ValueError, TypeError, RuntimeError, OSError, TimeoutError):
        logger.warning("ibr_stage_unexpected_error", exc_info=True)
        return _INACTIVE_RESULT


# ---------------------------------------------------------------------------
# Internal orchestration
# ---------------------------------------------------------------------------


# DEVIATION CS-PARAM-001: _execute_ibr takes 5 params — dataclass grouping would break API.
async def _execute_ibr(
    order: DispatchOrder,
    candidates: list[BackendConfig],
    ibr_config: IntentClassificationConfig,
    flavor_loader: FlavorProfileLoader,
    adapter: GenerativeBackend,
) -> IBRResult:
    """Core IBR execution: classify + load profiles concurrently, then score.

    Classification (async LLM call) and profile lookup (sync, from memory)
    are launched concurrently via asyncio.gather (IBR-PIPE-02).
    """
    assert isinstance(adapter, GenerativeBackend), "adapter must be GenerativeBackend"
    assert len(candidates) > 0, "candidates must not be empty"

    timeout_s = ibr_config.timeout_ms / 1000.0

    # Concurrent: classification (async) + profile reload check (sync, wrapped)
    intent, _profiles = await asyncio.gather(
        classify_intent(order.operator_message, adapter, timeout_s=timeout_s),
        _reload_profiles(flavor_loader),
    )

    if intent is None:
        logger.debug("ibr_classification_returned_none")
        return _INACTIVE_RESULT

    return _build_ibr_result(intent, candidates, ibr_config, flavor_loader)


async def _reload_profiles(loader: FlavorProfileLoader) -> None:
    """Trigger a hot-reload check on the flavor profile loader.

    Wrapped as a coroutine so it can participate in asyncio.gather
    alongside the classification call (IBR-PIPE-02).
    """
    assert isinstance(loader, FlavorProfileLoader), "loader must be FlavorProfileLoader"
    loader.reload_if_changed()


def _passes_confidence_gate(
    intent: ClassifiedIntent,
    candidate_ids: list[str],
    profiles: dict[str, ModelFlavorProfile],
    ibr_config: IntentClassificationConfig,
) -> bool:
    """Check whether the classification passes confidence gating (IBR-SCORE-04)."""
    assert len(candidate_ids) > 0, "candidate_ids must not be empty"
    gate_profile = get_profile_for_model(candidate_ids[0], profiles)
    return should_apply_flavor_match(
        intent,
        gate_profile,
        confidence_threshold=ibr_config.confidence_threshold,
        profile_confidence_threshold=ibr_config.profile_confidence_threshold,
    )


def _build_ibr_result(
    intent: ClassifiedIntent,
    candidates: list[BackendConfig],
    ibr_config: IntentClassificationConfig,
    flavor_loader: FlavorProfileLoader,
) -> IBRResult:
    """Apply confidence gating and compute flavor scores."""
    assert isinstance(intent, ClassifiedIntent), "intent must be ClassifiedIntent"
    assert len(candidates) > 0, "candidates must not be empty"

    profiles = flavor_loader.profiles
    candidate_ids = [c.name for c in candidates]

    if not _passes_confidence_gate(intent, candidate_ids, profiles, ibr_config):
        logger.info(
            "ibr_confidence_gated",
            classifier_confidence=intent.confidence,
            threshold=ibr_config.confidence_threshold,
        )
        return IBRResult(classified_intent=intent, flavor_scores={}, ibr_active=False)

    scores = compute_flavor_scores(intent, profiles, candidate_ids)
    _log_flavor_scores(scores)
    return IBRResult(classified_intent=intent, flavor_scores=scores, ibr_active=True)


def _log_flavor_scores(scores: dict[str, float]) -> None:
    """Emit structured log for flavor match scores (IBR-OBS-02)."""
    score_range = (
        (round(min(scores.values()), 4), round(max(scores.values()), 4))
        if scores else (0.0, 0.0)
    )
    logger.info("ibr_flavor_match", candidate_count=len(scores), score_range=score_range)
