# Implementation Plan: RT-012 - Add TPM + daily token cap tracking to BudgetTracker

## Task Description
**RT-012**: Add TPM + daily token cap tracking to BudgetTracker — critical, standard effort, parallelizable=False

- Depends on: RT-003
- Targets: `src/dragonlight_router/budget/tracker.py`

## Changes
- Add _tpm_windows: dict[str, deque[float]] sliding window (same pattern as RPM)
- Add _tokens_today: dict[str, int] counter
- Add _tpm_remaining(provider) method
- Extend has_capacity() to check RPM + RPD + TPM + token_cap
- Extend score() to include TPM ratio in composite
- Extend record_request() to track tokens in TPM window and daily token counter

## Acceptance Criteria
- [ ] TPM sliding window tracking works identically to RPM tracking
- [ ] Daily token cap enforced — has_capacity() returns False when cap reached
- [ ] score() returns min(rpm_ratio, rpd_ratio, tpm_ratio, token_ratio) * 100
- [ ] 0 token_cap means unlimited (ratio=1.0)
- [ ] Unit tests for all new capacity checks
- [ ] Property test: score() in [0, 100] for all valid inputs

## Agent Instructions
Add TPM sliding window to BudgetTracker following the exact pattern of RPM tracking. Add daily token cap counter following RPD pattern. Extend has_capacity() and score() to include these new dimensions. The ProviderConfig already has tpm_limit field. Add comprehensive tests.