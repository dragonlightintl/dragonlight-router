# Contributing

## Development setup

```bash
git clone https://github.com/dragonlightintl/dragonlight-router
cd dragonlight-router
pip install -e ".[all,dev]"
```

This installs the router with all provider extras plus development tools: pytest, mypy, ruff, hypothesis, and bandit.

## Running tests

The test suite is organized into four layers:

```
tests/
  unit/              Unit tests (isolated, fast, no I/O)
    adapters/        Provider adapter unit tests
    selection/       MBR/CBR/LBR stage tests
  integration/       Integration tests (subsystem interactions)
  contracts/         Contract tests (protocol conformance)
  acceptance/        Acceptance tests (end-to-end through the HTTP API)
  smoke/             Smoke tests (basic startup and connectivity)
```

### Commands

```bash
make test                      # Full suite (no coverage)
make test-cov                  # Full suite with coverage report

# Or directly:
python3 -m pytest --no-cov -q                    # All tests
python3 -m pytest tests/unit/ --no-cov -q        # Unit tests only
python3 -m pytest tests/integration/ --no-cov -q # Integration tests only
```

The test suite uses `pytest-asyncio` for async test support and `hypothesis` for property-based testing. Tests that hit real provider APIs are marked with `@pytest.mark.live` and excluded by default. To run them:

```bash
python3 -m pytest -m live --no-cov
```

### Coverage

Coverage is configured in `pyproject.toml` with an 80% minimum threshold. The default `pytest` invocation (without `--no-cov`) generates a terminal coverage report with missing lines highlighted.

### Timeouts

All tests have a 60-second timeout (configurable in `pyproject.toml`). Individual tests can override with `@pytest.mark.timeout(seconds)`.

## Linting and type checking

```bash
make lint                      # ruff check src/ tests/
make typecheck                 # mypy src/dragonlight_router/
make security                  # bandit security scanner

# Or directly:
python3 -m ruff check src/ tests/
python3 -m ruff format src/ tests/        # Auto-format
python3 -m mypy src/dragonlight_router/
python3 -m bandit -r src/dragonlight_router/ -s B101,B603
```

### Ruff

The project uses ruff for linting and formatting. Configuration is in `pyproject.toml`:

- Target: Python 3.11
- Line length: 100
- Selected rules: `E`, `F`, `I`, `UP`, `B`, `C4`, `SIM`, `D100`, `D101`
- Docstring convention: Google style
- Test files are exempt from docstring rules

### mypy

Type checking runs in strict mode. All public functions and module-level variables require type annotations.

## Pull request conventions

- One logical change per PR
- All tests must pass (`make test`)
- No new mypy errors (`make typecheck`)
- No new ruff errors (`make lint`)
- Add a `CHANGELOG.md` entry under `## [Unreleased]`

## Code standards

- **Function length:** 40 lines maximum. Longer functions require an explicit `DEVIATION CS-004` comment with justification, architect approval, and a review expiration date.
- **Result types:** All fallible operations return `Result[T, E]` instead of raising exceptions. See [ADR-001](adr/001-result-type-pattern.md).
- **Frozen dataclasses:** Core data types are immutable after construction.
- **Assertions:** Used for invariant checking in non-test code (bandit skips `B101`).

## Reporting bugs

Open an issue on GitHub. Include: Python version, OS, reproduction steps, and full traceback.
