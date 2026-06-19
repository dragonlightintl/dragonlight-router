.PHONY: help install dev test test-cov test-unit test-integration test-contract test-property test-smoke test-acceptance lint format format-check typecheck security dep-audit docs docs-serve run docker-build docker-run clean all

PYTHON ?= python3
PORT   ?= 8100
HOST   ?= 127.0.0.1

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install package with all extras
	$(PYTHON) -m pip install -e ".[all]"

dev: ## Install with all extras + dev tools
	$(PYTHON) -m pip install -e ".[all,dev]"

test: ## Run test suite (fast, no coverage)
	$(PYTHON) -m pytest --no-cov -q

test-cov: ## Run tests with coverage report
	$(PYTHON) -m pytest --cov-report=term-missing

test-unit: ## Run unit tests only
	$(PYTHON) -m pytest tests/unit/ --no-cov -q

test-integration: ## Run integration tests only
	$(PYTHON) -m pytest tests/integration/ --no-cov -q

test-contract: ## Run contract tests only
	$(PYTHON) -m pytest tests/contracts/ --no-cov -q

test-property: ## Run property-based tests only
	$(PYTHON) -m pytest tests/unit/test_properties.py --no-cov -q

test-smoke: ## Run smoke tests only
	$(PYTHON) -m pytest tests/smoke/ --no-cov -q

test-acceptance: ## Run acceptance tests only
	$(PYTHON) -m pytest tests/acceptance/ --no-cov -q

lint: ## Run ruff linter
	$(PYTHON) -m ruff check src/ tests/

format: ## Format code with ruff
	$(PYTHON) -m ruff format src/ tests/

format-check: ## Check code formatting without changes
	$(PYTHON) -m ruff format --check src/ tests/

typecheck: ## Run mypy strict type checking
	$(PYTHON) -m mypy src/dragonlight_router/

security: ## Run bandit security scanner
	$(PYTHON) -m bandit -r src/dragonlight_router/ -c pyproject.toml

dep-audit: ## Audit dependencies for known vulnerabilities
	$(PYTHON) -m pip_audit

docs: ## Build documentation site
	$(PYTHON) -m mkdocs build --strict

docs-serve: ## Serve documentation locally
	$(PYTHON) -m mkdocs serve

run: ## Start the router server locally
	DRAGONLIGHT_ROUTER_CONFIG=./config/router.yaml \
		DRAGONLIGHT_HOST=$(HOST) DRAGONLIGHT_PORT=$(PORT) \
		$(PYTHON) -m dragonlight_router.server.app

docker-build: ## Build the Docker image
	docker build -t dragonlight-router:latest .

docker-run: ## Run the router in Docker (port 8100)
	docker run --rm -p $(PORT):8100 \
		--env-file .env \
		dragonlight-router:latest

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info src/*.egg-info site/ htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f .coverage coverage.xml

all: lint format-check typecheck security test ## Run lint + format-check + typecheck + security + tests (CI in a box)
