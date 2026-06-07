# Implementation Plan for Remaining Dragonlight Router Tasks

## Overview
This plan implements the remaining tasks from the delta-spec in parallel waves, respecting dependencies, managing rate limits (<38 req/min), and ensuring seamless integration into main branch.

## Task Dependencies Analysis

### Completed Tasks (from delta-spec):
- RT-003, RT-012, RT-013, RT-014, RT-015

### Remaining Tasks from live-spec task_map:
1. TM-001: MBR (Model Based Router) - capability filtering stage
2. TM-002: CBR (Cost Balancing Router) - cost scoring stage  
3. TM-003: LBR (Load Balancing Router) - rate enforcement + final selection
4. TM-004: Cascade dispatch (MBR→CBR→LBR composition + execution)
5. TM-005: GenerativeBackend adapters for all 8 providers
6. TM-006: Context trust tier filtering (DIAN CECHT)
7. TM-007: Cost governor + ScoringWeights per canonical spec
8. TM-008: Health check background loop
9. TM-009: HTTP API enhancements (dispatch endpoint, trust-tier headers)
10. TM-010: RouterEngine.dispatch() method for engine consumers
11. TM-011: Integration test suite for end-to-end cascade routing
12. TM-012: BudgetTracker TPM and daily token cap enforcement

## Wave-Based Implementation Strategy

### Wave 1: Foundation Components (Independent, can run in parallel)
**Target: <8 min wall-clock per subagent**
- TM-001: MBR (Model Based Router) - selection/mbr.py
- TM-006: Context trust tier filtering (DIAN CECHT) - selection/context_filter.py  
- TM-007: Cost governor + ScoringWeights - selection/scoring.py
- TM-012: BudgetTracker TPM/daily cap - budget/tracker.py

### Wave 2: Dependent on Wave 1
**Target: <8 min wall-clock per subagent**
- TM-002: CBR (Cost Balancing Router) - selection/cbr.py [depends on TM-001, TM-007]
- TM-003: LBR (Load Balancing Router) - selection/lbr.py [depends on TM-002]
- TM-008: Health check background loop - health/check_loop.py [independent]

### Wave 3: Dependent on Waves 1-2
**Target: <8 min wall-clock per subagent**
- TM-004: Cascade dispatch - dispatch/cascade.py [depends on TM-001, TM-002, TM-003, TM-006]
- TM-005: GenerativeBackend adapters - adapters/* [independent but substantial]
- TM-009: HTTP API enhancements - server/routes.py, server/app.py [depends on TM-004]

### Wave 4: Final Integration
**Target: <8 min wall-clock per subagent**
- TM-010: RouterEngine.dispatch() method - router.py [depends on TM-004, TM-009]
- TM-011: Integration test suite - tests/integration/test_cascade_dispatch.py [depends on TM-010]

## Rate Limit Management
- Max concurrent subagents: 3 (to stay under ~38 req/min)
- Each subagent gets ~2-3 minutes of active work time
- Checkpoint every 2 minutes with progress assessment
- Automatic handoff if wall-clock approaches 8 minutes

## Progress Tracking & Auto-Merge
- After each subagent completes: commit to feature branch
- After each wave: create PR and merge to main
- Final validation: run all tests before pushing to origin

## Checkpoint Strategy
Each subagent will:
1. Work for max 6 minutes
2. At 6-minute mark: save progress, commit, push to feature branch
3. If not complete: create handoff document for next subagent
4. If complete: signal for wave completion

## Files to Update in delta-spec.jsonl
As tasks complete, update the completed_tasks array in:
docs/live-specs/delta-spec.jsonl

Let's begin with Wave 1...