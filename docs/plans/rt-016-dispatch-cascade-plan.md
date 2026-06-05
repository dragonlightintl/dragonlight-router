# Implementation Plan: RT-016 - Create dispatch/cascade.py — MBR→CBR→LBR composition

## Task Description
**RT-016**: Create dispatch/cascade.py — MBR→CBR→LBR composition — critical, complex effort, parallelizable=False

- Depends on: RT-013, RT-014, RT-015
- Targets: `src/dragonlight_router/dispatch/cascade.py`

## Changes
- Create route(order, registry, budget, rate, health, cost_governor, queue_depths, config) -> Result[EngineResponse, DispatchFailure]
- Implement MBR→CBR→LBR pipeline with Result composition
- Each stage returns Result[CandidateSet, StageError]
- Empty candidate set → Err with diagnostics
- Construct fallback chain from LBR survivors

## Acceptance Criteria
- [ ] All 5 TM-004 acceptance criteria met
- [ ] Pipeline composition order: MBR then CBR then LBR (load-bearing, not configurable)
- [ ] Each stage failure produces typed Err with context
- [ ] Fallback chain preserves LBR ordering
- [ ] Unit tests for each stage failure path
- [ ] Integration test for full pipeline

## Agent Instructions
Create dispatch/cascade.py with route() function. Compose MBR→CBR→LBR using guard-clause Result composition. Each stage failure short-circuits. Create tests/unit/test_cascade.py alongside.

## Dependencies Status Check
- RT-013 (MBR): ❌ NOT_STARTED - needs to be created first
- RT-014 (CBR): ❌ NOT_STARTED - needs to be created first  
- RT-015 (LBR): ❌ NOT_STARTED - needs to be created first

Since RT-016 depends on RT-013, RT-014, RT-015, these must be completed first. However, for maximum parallelism, I'll create implementation plans for all three MBR/CBR/LBR stages first, then execute them in parallel subagents, followed by RT-016.