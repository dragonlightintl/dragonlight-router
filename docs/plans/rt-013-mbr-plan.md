# Implementation Plan: RT-013 - Create selection/mbr.py — MBR capability filtering stage

## Task Description
**RT-013**: Create selection/mbr.py — MBR capability filtering stage — critical, complex effort, parallelizable=True

- Depends on: RT-003
- Targets: `src/dragonlight_router/selection/mbr.py`

## Changes
- Create filter_by_capability(candidates, tier, health_cache) -> Result[list[Candidate], MBRNoCandidatesError]
- Create estimate_complexity(order: DispatchOrder) -> BackendTier
- Implement adjacent-tier graceful upgrade logic
- Implement circuit_open exclusion
- Implement never-downgrade invariant
- Implement local-provider unlimited-rate passthrough

## Acceptance Criteria
- [ ] All 5 TM-001 acceptance criteria met
- [ ] Function bodies ≤40 lines
- [ ] All functions have ≥2 assertions
- [ ] structlog logging at entry/exit
- [ ] Hypothesis property test: tier_never_downgrades
- [ ] Unit tests for each AC criterion

## Agent Instructions
Create mbr.py from scratch. Implement filter_by_capability and estimate_complexity. Use Result type for return. Add guard clauses, assertions, structlog. Follow the complexity.py pattern for tier classification. Create tests/unit/test_mbr.py alongside.