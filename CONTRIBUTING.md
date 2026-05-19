# Contributing to dragonlight-router

## Development Setup

```bash
git clone https://github.com/dragonlightintl/dragonlight-router
cd dragonlight-router
pip install -e ".[dev,server,cache]"
```

## Running Tests

```bash
pytest                          # run all tests with coverage
pytest tests/unit/              # unit tests only
```

## Linting and Type Checking

```bash
ruff check src tests            # lint
ruff format src tests           # format
mypy src                        # type check
```

## Pull Request Conventions

- One logical change per PR
- All tests must pass (`pytest`)
- No new mypy errors (`mypy src`)
- No new ruff errors (`ruff check src tests`)
- Add a `CHANGELOG.md` entry under `## [Unreleased]`

## Reporting Bugs

Open an issue at GitHub Issues. Include: Python version, OS, reproduction steps, and full traceback.
