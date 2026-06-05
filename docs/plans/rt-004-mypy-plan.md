# Implementation Plan: RT-004 - Fix mypy strict violations (13 errors)

## Task Description
**RT-004**: Fix mypy strict violations (13 errors) — hard, standard effort, parallelizable=True

- Depends on: —
- Targets: `src/dragonlight_router/core/registry.py`, `src/dragonlight_router/caching/simple.py`, `src/dragonlight_router/catalog/cache.py`, `src/dragonlight_router/budget/persistence.py`, `src/dragonlight_router/catalog/refresher.py`, `src/dragonlight_router/server/app.py`

## Changes
- registry.py:41,43 — add dict[str, GenerativeBackend] and dict[str, BackendState] type params
- caching/simple.py:48 — add explicit cast for Any return from sqlite3
- caching/simple.py:68 — add dict[str, Any] type param
- catalog/cache.py:78,83,94 — add proper dict type params, fix Any return
- budget/persistence.py:18,47,60 — add dict type params, fix Any return
- catalog/refresher.py:44 — add isinstance narrow before assignment
- server/app.py:23 — add parameter type annotation
- Install types-PyYAML stubs

## Acceptance Criteria
- [ ] mypy --strict src/ produces zero errors
- [ ] pip install types-PyYAML added to project deps or dev deps
- [ ] All existing tests pass

## Agent Instructions
Fix each mypy error by adding proper type annotations. Run mypy after each fix to confirm. Install types-PyYAML. Target: mypy src/ — zero errors.