# ADR-001: Result Type Pattern

## Status
Accepted

## Context
The router cascade pipeline composes multiple fallible stages (MBR filtering, CBR scoring, LBR rate-limit checks, adapter dispatch). Each stage can fail for distinct, expected reasons: no candidates meet capability requirements, budget is exhausted, rate limits are hit, or a provider returns an error. Using Python exceptions for these expected failure modes creates several problems: exception handlers are easy to omit, error types are invisible in function signatures, and broad `except` clauses can silently swallow unrelated bugs. In a cascade where failures are normal (and trigger the next fallback candidate), conflating expected failures with unexpected exceptions makes the fallback logic fragile.

## Decision
All fallible operations return `Result[T, E]` — a union of `Ok[T]` and `Err[E]` frozen dataclasses defined in `core.types`. Callers pattern-match on the result type with `isinstance` checks. Python exceptions are reserved for truly unexpected conditions (programming errors, I/O failures outside the cascade). The `result` module provides helper functions (`ok()`, `err()`, `is_ok()`, `unwrap()`) for ergonomic use.

## Consequences
**Positive:**
- Error paths are explicit in every function signature.
- The cascade can branch on `Ok`/`Err` without try/except nesting.
- Impossible to accidentally catch an unrelated exception during fallback.
- Frozen dataclasses make results safe to pass through async boundaries.

**Negative:**
- Adds a pattern unfamiliar to most Python developers.
- Requires `isinstance` checks at every call site instead of implicit propagation.
- Type checkers need careful annotation to track the union correctly.
