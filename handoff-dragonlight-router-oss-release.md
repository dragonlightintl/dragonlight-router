# Handoff: dragonlight-router — Open-Source Release Checklist

> Generated from ground-truth audit of `/Users/coryflanigan/dragonlight-ops/dragonlight-router`.  
> All findings are rooted in direct file inspection. Author metadata: `Korrigon @ Dragonlight International`.

---

## What This Project Is

**dragonlight-router** is a Python multi-provider LLM routing engine. It sits between your application and a fleet of LLM providers (NVIDIA NIM, Groq, OpenRouter, Cerebras, Gemini, Mistral, Anthropic, Ollama) and intelligently selects + ranks models by role, live availability, budget headroom, and health state. It does no inference itself — it routes to whoever should do it.

**Key subsystems:**

| Subsystem | What it does |
|---|---|
| `RoleMatrix` | Hot-reloadable JSON file mapping roles → ranked model IDs |
| `BudgetTracker` | Sliding-window RPM + daily RPD tracking per provider |
| `HealthTracker` | Per-model error counting + EWMA latency |
| `CircuitBreaker` | CLOSED→OPEN→HALF_OPEN state machine, configurable thresholds |
| `CatalogCache` | File-backed TTL cache of live provider model lists |
| `CatalogRefresher` | Concurrent async fetch from provider `/v1/models` endpoints |
| `RouterEngine` | Wires everything; dual interface: `select_models()` + `record_request()` |
| `Server` | Starlette HTTP API: `/v1/select`, `/v1/record`, `/v1/health`, `/v1/catalog` |
| `SimpleCache` | SHA-256 exact-match response cache (SQLite, WAL) |
| `SemanticCache` | Character n-gram Jaccard similarity cache for near-duplicates |
| `ComplexityEstimator` | Heuristic that maps intent + context size → tier (LOCAL/HAIKU/SONNET/OPUS) |

**Code quality baseline:** Architecture is clean and well above average for this stage. 1,990 lines of unit tests across 15 files. `mypy --strict`, `ruff`, `pytest-asyncio`. The gaps are all *around* the code, not inside it.

---

## Current State (as audited)

| Area | Status |
|---|---|
| Package metadata (`pyproject.toml`) | 🟡 Partial — missing license, authors, classifiers, keywords, URLs |
| `README.md` | 🔴 Missing |
| `LICENSE` | 🔴 Missing — legally "all rights reserved" without it |
| `CHANGELOG.md` | 🔴 Missing |
| `CONTRIBUTING.md` | 🔴 Missing |
| `SECURITY.md` | 🔴 Missing |
| `.gitignore` | 🔴 Missing — secrets exposure risk if git-inited |
| `.env.example` | 🟡 Missing — 7 API key env vars undocumented |
| CI/CD (GitHub Actions) | 🔴 Missing — no automated test runner |
| Git repository | 🔴 Not initialized — no history, no remote |
| Unit tests | 🟢 Solid — 15 files, 1,990 lines |
| Integration tests | 🔴 Empty stubs |
| Coverage config | 🟡 Dep declared, not wired to pytest run |
| Static typing (`mypy --strict`) | 🟢 Configured |
| Linting (`ruff`) | 🟡 Configured but minimal rule set |
| Secrets in code | 🟢 Clean — env_key pattern only, no hardcoded values |

---

## Priority 1 — Blockers (must fix before any public commit)

### 1. `.gitignore` — Create before `git init`

**Risk:** Without this, `git add .` would commit `__pycache__/`, `.pytest_cache/`, `*.egg-info/`, and any future `.env` file.

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.eggs/
*.egg

# Secrets — never commit
.env
.env.*
!.env.example

# Runtime state (budget persistence, SQLite caches)
router_state/
*.db
*.sqlite

# Tool caches
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage
htmlcov/

# OS / Editor
.DS_Store
Thumbs.db
.vscode/
.idea/
```

### 2. `LICENSE` — MIT Recommended

Without a license, copyright law defaults to "all rights reserved." Other developers **cannot legally use, modify, or distribute** the code regardless of it being public. MIT is the path of least resistance for a developer tool — maximally permissive, universally understood, encourages adoption.

**Author:** `Korrigon @ Dragonlight International`  
**Year:** 2025

### 3. Complete `pyproject.toml` Metadata

Currently missing (found at `pyproject.toml`):

```toml
[project]
# Add these fields:
license = {text = "MIT"}
authors = [{name = "Korrigon @ Dragonlight International"}]
keywords = ["llm", "routing", "ai", "multi-provider", "openai", "anthropic"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

[project.urls]
Homepage = "https://github.com/YOUR_USERNAME/dragonlight-router"
Repository = "https://github.com/YOUR_USERNAME/dragonlight-router"
Issues = "https://github.com/YOUR_USERNAME/dragonlight-router/issues"
```

---

## Priority 2 — Required for a Maintainable OSS Project

### 4. `README.md`

The single file that determines adoption. Developers decide within 30 seconds whether a repo is worth their time. Must include:

- **What it is** (one sentence)
- **Why it exists** — the problem: provider churn, budget enforcement, circuit breaking
- **Install** — `pip install dragonlight-router[all]`
- **Quickstart** — copy-pasteable working example with `get_router()` + `select_models()`
- **Configuration** — `router.yaml` format + the 7 required env vars
- **Architecture overview** — subsystem table
- **Scope of v0.1** — dispatch/ is intentionally empty (see note below)
- **Contributing** pointer
- **License** badge + statement

### 5. `.env.example`

The project resolves 7 API keys from environment variables (`config/router.yaml` lines 11, 22, 30, 36, 44, 62, 81). No documentation exists for new users.

```bash
# .env.example — copy to .env and fill in your keys
NVIDIA_NIM_API_KEY=
GROQ_API_KEY=
OPENROUTER_API_KEY=
CEREBRAS_API_KEY=
GEMINI_API_KEY=
MISTRAL_API_KEY=
ANTHROPIC_API_KEY=

# Optional overrides
DRAGONLIGHT_ROUTER_CONFIG=./config/router.yaml
DRAGONLIGHT_HOST=127.0.0.1
DRAGONLIGHT_PORT=8100
```

### 6. GitHub Actions CI (`.github/workflows/ci.yml`)

No automated test runner exists. PRs can silently break tests.

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -e ".[dev]"
      - run: ruff check src tests
      - run: mypy src
      - run: pytest --cov=dragonlight_router --cov-report=term-missing
```

---

## Priority 3 — Code Fixes Before Tagging v0.1

All findings are from direct source inspection.

### 7. Fix `assert` in production paths (`src/dragonlight_router/core/state.py`)

`assert` is stripped when Python runs with `-O`. Several `BackendState` capacity-check methods use `assert` for invariant enforcement. Replace with explicit raises:

```python
# Before:
assert provider_id in self._budgets
# After:
if provider_id not in self._budgets:
    raise ValueError(f"Unknown provider_id: {provider_id!r}")
```

### 8. Add env-var overrides for host/port (`src/dragonlight_router/server/app.py:48`)

Currently hardcoded: `uvicorn.run(app, host="127.0.0.1", port=8100)`. Container deployments can't configure without patching source.

```python
import os
host = os.environ.get("DRAGONLIGHT_HOST", "127.0.0.1")
port = int(os.environ.get("DRAGONLIGHT_PORT", "8100"))
uvicorn.run(app, host=host, port=port)
```

### 9. Expand `ruff` lint rules (`pyproject.toml`)

Currently only `E` (pycodestyle errors) is active by default. For a project claiming `mypy --strict`:

```toml
[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "C4", "SIM"]
```

### 10. Wire coverage into pytest defaults (`pyproject.toml`)

`pytest-cov` is listed as a dev dependency but not connected. Any `pytest` invocation skips coverage silently.

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "--cov=dragonlight_router --cov-report=term-missing --cov-fail-under=80"
```

---

## Priority 4 — Helpful, Not Blocking

### 11. `CONTRIBUTING.md`
Minimum viable: dev setup (`pip install -e ".[dev]"`), how to run tests, how to run lint, PR conventions.

### 12. `CHANGELOG.md`
Start at v0.1.0 with `### Added` listing core subsystems. Use [Keep a Changelog](https://keepachangelog.com) format. GitHub surfaces this automatically.

### 13. `SECURITY.md`
Three lines: report vulnerabilities privately (email or GitHub private reporting), expected response time. GitHub shows a banner on the Issues tab when this exists.

### 14. Document incomplete subsystems in README

`dispatch/` and `adapters/` are empty stubs. The cascade dispatch pipeline (actually calling an LLM after routing) is not implemented. `ComplexityEstimator` has no caller. This is fine for v0.1 — but it **must** be clearly documented to prevent spurious bug reports:

> **Scope of v0.1:** Model selection and health routing only. Dispatch (actually calling the provider) is the application's responsibility. The `dispatch/` module is reserved for a future release.

---

## Execution Order (Path of Least Resistance)

```
Step  Action                                        Time
───────────────────────────────────────────────────────
1.    Create .gitignore                             2 min
2.    Create LICENSE (MIT, 2025)                   2 min
3.    Complete pyproject.toml metadata             5 min
4.    Create .env.example                          3 min
5.    Write README.md                              45 min  ← the real work
6.    Create .github/workflows/ci.yml              10 min
7.    Fix assert → raise in state.py               10 min
8.    Fix hardcoded host/port in app.py            5 min
9.    Expand ruff rules + wire coverage            5 min
10.   Create CONTRIBUTING.md                       15 min
11.   Create CHANGELOG.md                          5 min
12.   Create SECURITY.md                           5 min
13.   git init + git add . + git commit            2 min
14.   Create GitHub repo + git push                5 min
15.   Tag v0.1.0 + create GitHub Release           3 min
───────────────────────────────────────────────────────
Total: ~2 hours of focused work
```

---

## Notes on What NOT to Change

- **`config/router.yaml` env_key pattern** — correct by design. API keys are referenced by env var name, never by value. Do not "fix" this.
- **`asyncio.gather(..., return_exceptions=True)` in `CatalogRefresher`** — correct. One failing provider must not kill the whole refresh cycle.
- **Atomic file writes (tmp → rename + fsync)** in `catalog/cache.py` and `budget/persistence.py` — correct. Do not simplify.
- **`asyncio_mode = "auto"` in pytest config** — correct for the async-heavy codebase.

---

## Legal Note

Until a LICENSE file is added and committed to the public repository, **all code is legally "all rights reserved"** under copyright law, regardless of whether the repository is public. The MIT License is recommended. Do not push to a public GitHub repository before this file exists.
