# Implementation Plan: RT-014 - Create selection/cbr.py — CBR cost balancing stage

## Task Description
**RT-014**: Create selection/cbr.py — CBR cost balancing stage — critical, complex effort, parallelizable=False

- Depends on: RT-003, RT-013
- Targets: `src/dragonlight_router/selection/cbr.py`

## Changes
- Create CostGovernorConfig frozen dataclass
- Create cost_governor_active(daily_spend, daily_budget, config) -> bool
- Create cost_adjusted_weights(base_weights, daily_spend, daily_budget) -> ScoringWeights
- Create filter_by_budget(candidates, budget_tracker) -> Result[list[Candidate], CBRBudgetExhaustedError]
- Create score_by_cost(candidates, budget_tracker, cost_governor) -> list[ScoredCandidate]

## Acceptance Criteria
- [ ] All 4 TM-002 acceptance criteria met
- [ ] CostGovernor activates at 80% daily_budget threshold
- [ ] cost_adjusted_weights smoothly shifts rank→budget weight
- [ ] All functions ≤40 lines, ≥2 assertions, Result returns, structlog
- [ ] Hypothesis test: cost_adjusted_weights_sum_to_1
- [ ] Unit tests for each AC criterion

## Agent Instructions
Create cbr.py. Depends on ScoringWeights from scoring.py (RT-003 graft) and MBR output format (RT-013). Create tests/unit/test_cbr.py alongside. Follow coding standards strictly.