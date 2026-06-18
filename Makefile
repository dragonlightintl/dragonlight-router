.PHONY: help install dev test test-cov lint format typecheck security docs docs-serve run docker-build docker-run clean all

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

lint: ## Run ruff linter
	$(PYTHON) -m ruff check src/ tests/

format: ## Format code with ruff
	$(PYTHON) -m ruff format src/ tests/

typecheck: ## Run mypy strict type checking
	$(PYTHON) -m mypy src/dragonlight_router/

security: ## Run bandit security scanner
	$(PYTHON) -m bandit -r src/dragonlight_router/ -c pyproject.toml

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

all: lint typecheck security test ## Run lint + typecheck + security + tests (CI in a box)
