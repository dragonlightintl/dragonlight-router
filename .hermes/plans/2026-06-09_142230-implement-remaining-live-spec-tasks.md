# Plan: Implement Remaining Live Spec Tasks (TM-006, TM-008)

## Goal
Complete implementation of TM-006 (Context Trust Tier Filtering) and TM-008 (Health Check Background Loop) to bring codebase fully compliant with live-spec v0.2.0.

## Current Context / Assumptions
- **Git Status**: On branch `main` with modifications to:
  - `src/dragonlight_router/health/check_loop.py` (TM-008 implementation)
  - `src/dragonlight_router/router.py` (TM-008 integration)
  - `src/dragonlight_router/server/app.py` (TM-008 integration)
  - Untracked: `current-status-06-08-2026.md`
- **Live Spec Compliance**: 
  - Per delta-spec.jsonl: TM-001,002,003,004,005,007,009,010,011,012 marked complete
  - Analysis shows TM-006 and TM-008 require test completion
- **Dependencies**: 
  - TM-006 depends on TM-004 (Cascade dispatch) - assumed complete
  - TM-008 depends on TM-005 (GenerativeBackend adapters) - assumed complete
- **Estimated Effort**: Both tasks marked "standard" complexity in live spec

## Proposed Approach
Complete missing unit tests for both tasks to verify implementation correctness and achieve full live spec compliance. Focus on test-driven validation rather than re-implementing existing code.

### TM-006: Context Trust Tier Filtering (DIAN CECHT)
- **Status**: Implementation complete in `src/dragonlight_router/selection/context_filter.py`
- **Gap**: Missing tests for main function `filter_context_for_provider`
- **Action**: Write comprehensive unit tests covering all 5 acceptance criteria

### TM-008: Health Check Background Loop
- **Status**: Implementation largely complete (SLO enforcement, degraded status, 404 handling)
- **Gap**: Missing tests for SLO/degraded status logic and ranking penalty
- **Action**: Write missing unit tests to validate acceptance criteria

## Step-by-Step Plan

### Phase 1: TM-006 Test Implementation (~15 minutes)
1. Examine existing `tests/unit/test_context_filter.py` structure
2. Add test class/functions for `filter_context_for_provider`:
   - Test TRUSTED providers receive full system-level context
   - Test SEMI_TRUSTED providers: 
     * Remove behavioral rules
     * Replace persona names with "[REDACTED PERSONA]"
     * Limit history to last 3 turns
   - Test UNTRUSTED providers receive task-specific instruction only
   - Test LOCAL providers receive full context (no network egress)
   - Test PII never present (validate pre-redaction assumption)
3. Run tests to verify implementation correctness

### Phase 2: TM-008 Test Completion (~20 minutes)  
1. Examine existing `tests/unit/test_check_loop.py` structure
2. Add missing test functions:
   - Test SLO enforcement: providers exceeding latency SLO for 3 consecutive checks transition to DEGRADED status
   - Test degraded providers receive ranking penalty (verify scoring integration)
   - Test combined SLO/failure scenarios
   - Test edge cases (boundary conditions, recovery from degraded status)
3. Run tests to verify SLO enforcement and degraded status logic

### Phase 3: Verification (~10 minutes)
1. Run all tests for both modules to ensure no regressions
2. Verify test coverage meets project standards
3. Confirm live spec compliance for TM-006 and TM-008

## Files Likely to Change
- `tests/unit/test_context_filter.py` - Add tests for `filter_context_for_provider`
- `tests/unit/test_check_loop.py` - Add tests for SLO/degraded/ranking logic

## Tests / Validation
- **TM-006**: 
  - `test_filter_context_for_provider_trusted_full_context`
  - `test_filter_context_for_provider_semitrusted_no_behavioral_rules`
  - `test_filter_context_for_provider_semitrusted_no_persona_names` 
  - `test_filter_context_for_provider_semitrusted_limited_history`
  - `test_filter_context_for_provider_untrusted_task_only`
  - `test_filter_context_for_provider_local_full_context`
  - `test_filter_context_for_provider_pii_never_present`
- **TM-008**:
  - `test_slo_enforcement_transitions_to_degraded_after_three_violations`
  - `test_degraded_providers_receive_ranking_penalty_not_excluded`
  - `test_slo_violation_count_reset_on_successful_compliant_check`
  - ` test_latency_slo_boundary_conditions`

## Risks, Tradeoffs, and Open Questions
### Risks
- **TM-006**: Implementation may have subtle bugs in regex patterns for behavioral rules/persona replacement
- **TM-008**: Ranking penalty implementation may be incomplete if not verified in scoring logic
- Test implementation may reveal actual implementation gaps requiring code fixes

### Tradeoffs
- Focus on test completion rather than implementation avoids potential regressions
- Test-first approach validates existing implementation without modification
- Time-boxed approach prevents over-engineering test suites

### Open Questions
1. **TM-008 Ranking Penalty**: Where is ranking penalty applied? Need to verify if:
   - Scoring system reduces score for DEGRADED backends
   - Or if DEGRADED status is treated differently in dispatch logic
   - May need to examine `src/dragonlight_router/selection/scoring.py` or `dispatch/cascade.py`
2. **Test Data**: What constitutes a behavioral rule or persona name in test contexts?
3. **SLO Values**: What latency SLO values should be used in tests for deterministic results?

## Git State Assessment
- **Branch**: main (current)
- **Uncommitted Changes**: 3 modified files (TM-008 implementation)
- **Untracked Files**: 1 (status document)
- **Ready State**: Changes should be committed after test implementation

## Dependency Assessment
- **TM-006**: Depends on TM-004 (Cascade dispatch) - assumed complete per delta-spec
- **TM-008**: Depends on TM-005 (GenerativeBackend adapters) - assumed complete per delta-spec
- **No Blocking Dependencies**: Both tasks can be worked on in parallel
- **Downstream Impact**: 
  - TM-006 enables secure context handling for all provider trust tiers
  - TM-008 enables reliable health checking with graceful degradation

## Live Spec Assessment
- **Source**: ./docs/live-specs/dragonlight-router.jsonl (live-spec v0.2.0)
- **Task Map**: Contains 12 tasks (TM-001 through TM-012)
- **Remaining Per Analysis**: TM-006 and TM-008 require test completion
- **Completeness Impact**: 
  - Completing these tasks achieves full test coverage for critical path
  - Enables confident verification of live spec compliance
  - Supports upcoming waves in execution plan (Waves 3-7 depend on foundation)