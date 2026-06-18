# Contributing to dragonlight-router

Contributions are welcome. This document covers the conventions and workflow you need to get started.

## Development setup

```bash
git clone https://github.com/dragonlightintl/dragonlight-router.git
cd dragonlight-router
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[all,dev]"
make all                # lint + typecheck + full test suite
```

If `make all` passes, your environment is correct.

## Code style

This project enforces strict quality gates:

- **Linting** — [Ruff](https://docs.astral.sh/ruff/) with `E`, `F`, `I`, `UP`, `B`, `C4`, `SIM`, `D100`, `D101` rules. Run: `make lint`
- **Formatting** — Ruff format, 100-char line length. Run: `make format`
- **Type checking** — [mypy](https://mypy-lang.org/) in `--strict` mode. Run: `make typecheck`
- **Security** — [Bandit](https://bandit.readthedocs.io/) SAST scanner. Run: `make security`

All four must pass before a PR will be reviewed.

## Testing

```bash
make test              # fast run, no coverage
make test-cov          # with coverage report (80% minimum enforced)
```

Guidelines:
- All new code requires tests. Coverage must not decrease.
- Property-based tests via [Hypothesis](https://hypothesis.readthedocs.io/) are encouraged for invariant-heavy logic.
- Unit tests go in `tests/unit/`, integration tests in `tests/integration/`.
- Tests marked `@pytest.mark.live` hit real provider APIs and are skipped by default.

## Pull request conventions

1. **One logical change per PR.** Split unrelated changes into separate PRs.
2. **Descriptive title** — prefix with `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`, or `security:`.
3. **Link to an issue** if one exists (e.g., `Closes #42`).
4. **All CI checks must pass** — lint, typecheck, security scan, tests.
5. **Add a CHANGELOG.md entry** under `## [Unreleased]` for user-facing changes.

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add weighted random selection to LBR stage
fix: prevent double-pathing in Gemini base URL
docs: add deployment runbook
```

Keep the subject line under 72 characters. Use the body for context when the "why" is not obvious from the subject.

## Architecture

The codebase is organized as a pipeline:

```
src/dragonlight_router/
  adapters/     Provider adapters (11 providers + OpenAI-compat base)
  budget/       Sliding-window rate tracking and budget scoring
  caching/      Exact-match + semantic response cache
  catalog/      Provider model catalog with TTL refresh
  config/       YAML config loader and schema
  core/         Shared types, Result pattern, registry, errors
  dispatch/     MBR→CBR→LBR cascade composition + fallback
  health/       Health tracking, circuit breaker
  roles/        Hot-reloadable role-to-model matrix
  selection/    MBR, CBR, LBR stages and scoring functions
  server/       HTTP server (Starlette), routes, middleware
```

For design decisions and rationale, see [ARCHITECTURE.md](ARCHITECTURE.md) and the [ADRs](docs/adr/).

## Reporting bugs

Start a [Discussion](https://github.com/dragonlightintl/dragonlight-router/discussions/categories/bug-reports) with your Python version, OS, reproduction steps, and full traceback. Confirmed bugs get promoted to Issues.

## Reporting security vulnerabilities

Do **not** open a public issue. See [SECURITY.md](SECURITY.md) for the disclosure process.

## Code of conduct

Be respectful and constructive. A formal Code of Conduct is being adopted; in the meantime, the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) applies.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
