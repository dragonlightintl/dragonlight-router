"""Domain-specific Hypothesis strategies for IBR intent types.

Provides reusable strategies for generating valid instances of:
  - ClassifiedIntent

All strategies respect the validation constants defined in
src/dragonlight_router/core/types.py (IBR_TASK_TYPES, IBR_DOMAINS, IBR_QUALITY_SPEED).
"""

from __future__ import annotations

from hypothesis import strategies as st

from dragonlight_router.core.types import (
    IBR_DOMAINS,
    IBR_QUALITY_SPEED,
    IBR_TASK_TYPES,
    ClassifiedIntent,
)


def classified_intent_strategy() -> st.SearchStrategy[ClassifiedIntent]:
    """Strategy for generating valid ClassifiedIntent instances.

    All fields are constrained to their canonical allowed values:
      - task_type: one of IBR_TASK_TYPES
      - domain: one of IBR_DOMAINS
      - quality_speed: one of IBR_QUALITY_SPEED
      - confidence: [0.0, 1.0]
      - latency_ms: [0.0, 200.0]
      - from_cache: boolean
    """
    return st.builds(
        ClassifiedIntent,
        task_type=st.sampled_from(sorted(IBR_TASK_TYPES)),
        domain=st.sampled_from(sorted(IBR_DOMAINS)),
        quality_speed=st.sampled_from(sorted(IBR_QUALITY_SPEED)),
        confidence=st.floats(min_value=0.0, max_value=1.0),
        latency_ms=st.floats(min_value=0.0, max_value=200.0),
        from_cache=st.booleans(),
    )
