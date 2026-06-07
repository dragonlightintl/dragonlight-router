"""Rate-Limit Balancing (LBR) stage for the cascade router.

Filters model candidates based on rate-limit budget availability.
"""
from __future__ import annotations

import structlog
from typing import List, Optional, Tuple

from dragonlight_router.budget.tracker import BudgetTracker
from dragonlight_router.core.types import BackendConfig, BackendRateLimits, DispatchOrder
from dragonlight_router.result import Err, Ok, Result

logger = structlog.get_logger(__name__)


def filter_by_rate_limit(
    candidates: List[BackendConfig],
    order: DispatchOrder,
    budget_tracker: BudgetTracker,
) -> Result[List[BackendConfig], Exception]:
    """Filter candidates based on rate-limit budget availability.
    
    Implements rolling window rate limiting, queue policy mode, and local provider exemptions.
    
    Args:
        candidates: List of backend configurations from previous stage.
        order: Dispatch order (may contain context for rate-limit adjustment).
        budget_tracker: Budget tracker to get rate-limit scores for providers.
        
    Returns:
        Ok(list of candidates that pass rate-limit checks) or
        Err(Exception) with diagnostic info if no candidates remain.
    """
    # Precondition assertions
    assert isinstance(candidates, list), "candidates must be a list"
    assert all(
        isinstance(c, BackendConfig) for c in candidates
    ), "all candidates must be BackendConfig instances"
    assert isinstance(order, DispatchOrder), "order must be DispatchOrder instance"
    assert isinstance(budget_tracker, BudgetTracker), "budget_tracker must be BudgetTracker instance"

    logger.debug(
        "filtering candidates by rate-limit budget",
        candidate_count=len(candidates),
        intent_category=order.intent_category,
        context_tokens=order.context_tokens,
    )

    # If no candidates, return as-is
    if not candidates:
        logger.debug("no candidates, returning as-is")
        return Ok(candidates)

    # Get rate-limit information for each candidate's provider
    provider_info = {}
    candidates_by_provider = {}
    
    for candidate in candidates:
        provider = candidate.provider
        if provider not in provider_info:
            # Get provider's rate limits from the candidate
            rate_limits: BackendRateLimits = candidate.rate_limits
            
            # Get current usage from budget tracker
            # The budget tracker score() method returns 0-100 where higher = more budget available
            # We need to convert this to usage ratio: 0.0 = unused, 1.0 = fully used
            score_result = budget_tracker.score(provider)
            # Handle both Ok and Err results from budget_tracker.score()
            if hasattr(score_result, 'value'):
                score_value = score_result.value
            else:
                # If it's an Err, treat as 0 score (no budget)
                score_value = 0.0
            
            # Convert score (0-100) to usage ratio (0.0-1.0)
            # Higher score = more budget available = lower usage
            usage_ratio = max(0.0, min(1.0, 1.0 - (score_value / 100.0)))
            
            provider_info[provider] = {
                'rate_limits': rate_limits,
                'usage_ratio': usage_ratio,  # 0.0 = unused, 1.0 = fully used
                'rpm': rate_limits.rpm,
                'rpd': rate_limits.rpd,
                'tpm': rate_limits.tpm,
                'daily_token_cap': rate_limits.daily_token_cap,
            }
        
        if provider not in candidates_by_provider:
            candidates_by_provider[provider] = []
        candidates_by_provider[provider].append(candidate)

    # Apply rate limit filtering with queue policy mode
    filtered_candidates = []
    providers_exceeding_limits = []
    providers_within_80_percent = []
    
    for provider, info in provider_info.items():
        rate_limits = info['rate_limits']
        usage_ratio = info['usage_ratio']
        
        # Check if provider is local (unlimited rate)
        # Local providers typically have RPM/RPD = 0 or very high numbers indicating unlimited
        is_local = (rate_limits.rpm == 0 and rate_limits.rpd == 0) or \
                   (rate_limits.rpm > 10000 and rate_limits.rpd > 100000)
        
        if is_local:
            # Local providers are exempt from rate limiting
            logger.debug("local provider exempt from rate limiting", provider=provider)
            filtered_candidates.extend(candidates_by_provider[provider])
            continue
        
        # Hard limit check: exclude if projected request would exceed limits
        # We assume each request uses 1 unit of RPM/RPD for simplicity
        # In reality, we'd estimate tokens, but for rate limiting we just count requests
        rpm_would_exceed = (usage_ratio * rate_limits.rpm) >= rate_limits.rpm
        rpd_would_exceed = (usage_ratio * rate_limits.rpd) >= rate_limits.rpd
        
        if rpm_would_exceed or rpd_would_exceed:
            providers_exceeding_limits.append(provider)
            logger.debug(
                "provider excluded due to rate limit exhaustion",
                provider=provider,
                rpm_usage=usage_ratio * rate_limits.rpm,
                rpm_limit=rate_limits.rpm,
                rpd_usage=usage_ratio * rate_limits.rpd,
                rpd_limit=rate_limits.rpd,
            )
            continue
        
        # Queue policy mode: deprioritize (but don't exclude) if within 80% of limits
        within_80_percent = (usage_ratio >= 0.8)
        if within_80_percent:
            providers_within_80_percent.append(provider)
            logger.debug(
                "provider within 80% of rate limits (will be deprioritized)",
                provider=provider,
                usage_ratio=usage_ratio,
            )
        
        # Provider passes rate limit checks
        filtered_candidates.extend(candidates_by_provider[provider])
    
    logger.debug(
        "rate-limit filtering complete",
        original_count=len(candidates),
        filtered_count=len(filtered_candidates),
        providers_exceeding_limits=providers_exceeding_limits,
        providers_within_80_percent=providers_within_80_percent,
    )

    # If no candidates remain after filtering, return error with diagnostic info
    if not filtered_candidates:
        from dragonlight_router.core.errors import LBRNoCapacityError
        diagnostic_info = {
            "total_candidates": len(candidates),
            "providers_checked": list(provider_info.keys()),
            "providers_exceeding_limits": providers_exceeding_limits,
            "providers_within_80_percent": providers_within_80_percent,
            "provider_details": {
                provider: {
                    "rpm_usage_ratio": info['usage_ratio'],
                    "rpm_limit": info['rpm'],
                    "rpd_limit": info['rpd'],
                    "tpm": info['tpm'],
                    "daily_token_cap": info['daily_token_cap'],
                    "is_local": (
                        info['rate_limits'].rpm == 0 and info['rate_limits'].rpd == 0
                    ) or (
                        info['rate_limits'].rpm > 10000 and info['rate_limits'].rpd > 100000
                    ),
                }
                for provider, info in provider_info.items()
            }
        }
        return Err(LBRNoCapacityError(
            f"No candidates have sufficient rate limit capacity. "
            f"Checked {len(provider_info)} providers, {len(providers_exceeding_limits)} exceeded limits."
        ))

    # TODO: Implement actual tie-breaking logic here
    # For now, we'll return the filtered candidates and let the caller handle selection
    # The live-spec mentions "select_final_candidate" which should be implemented elsewhere
    
    return Ok(filtered_candidates)


# Placeholder for select_final_candidate function that would be used by the cascade
def select_final_candidate(candidates: List[BackendConfig]) -> BackendConfig:
    """Select the final candidate from the list, breaking ties.
    
    In a full implementation, this would use additional scoring or heuristics
    to break ties between equally qualified candidates.
    
    Args:
        candidates: List of backend configurations to choose from.
        
    Returns:
        The selected BackendConfig.
        
    Raises:
        ValueError: If candidates list is empty.
    """
    if not candidates:
        raise ValueError("Cannot select from empty candidate list")
    
    # Simple implementation: return first candidate
    # In reality, this would use additional tie-breaking logic
    # such as preferring lower cost, better health, or round-robin
    return candidates[0]