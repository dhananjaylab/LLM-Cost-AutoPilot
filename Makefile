# ── LLM Cost Autopilot — Developer Makefile ──────────────────────────────────
.DEFAULT_GOAL := help
SHELL := /bin/bash

UV := uv
DC := docker compose
DC_FILE := docker-compose.yml

.PHONY: help install dev-install lint format typecheck test test-unit test-integration \
        up up-core down logs migrate migrate-create shell-api shell-db \
        ollama-pull clean reset

# ── Help ──────────────────────────────────────────────────────────────────────
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Environment ───────────────────────────────────────────────────────────────
install: ## Install production dependencies for all workspace members
	$(UV) sync --no-dev

dev-install: ## Install all dependencies including dev + pre-commit hooks
	$(UV) sync --dev
	$(UV) run pre-commit install
	@echo "✓ Dev environment ready"

# ── Code quality ──────────────────────────────────────────────────────────────
lint: ## Run ruff linter (auto-fix)
	$(UV) run ruff check . --fix
	$(UV) run ruff format .

format: ## Run black formatter
	$(UV) run black .

typecheck: ## Run mypy type checker
	$(UV) run mypy libs/ apps/

check: lint typecheck ## Run all static analysis

# ── Tests ─────────────────────────────────────────────────────────────────────
test: ## Run all tests (requires Postgres + Redis running)
	$(UV) run pytest tests/ -v --tb=short

test-unit: ## Run unit tests only (no external services needed)
	$(UV) run pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests (requires Postgres + Redis)
	$(UV) run pytest tests/integration/ -v --tb=short

test-cov: ## Run tests with HTML coverage report
	$(UV) run pytest tests/ \
	  --cov=libs/core/llm_autopilot_core \
	  --cov=apps/api/llm_autopilot_api \
	  --cov=apps/worker/llm_autopilot_worker \
	  --cov-report=html \
	  --cov-report=term-missing

# ── Docker ────────────────────────────────────────────────────────────────────
up: ## Start all services
	$(DC) -f $(DC_FILE) up -d --build
	@echo "✓ Services up"
	@echo "  API:       http://localhost:8000/docs"
	@echo "  Grafana:   http://localhost:3000  (admin / autopilot)"
	@echo "  Flower:    http://localhost:5555"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  RedisInsight: http://localhost:8001"

up-core: ## Start only Postgres + Redis (for local FastAPI dev)
	$(DC) -f $(DC_FILE) up -d postgres redis
	@echo "✓ Core services up (Postgres + Redis)"

up-infra: ## Start infra services only (no app containers)
	$(DC) -f $(DC_FILE) up -d postgres redis prometheus grafana
	@echo "✓ Infra services up"

down: ## Stop all services (keep volumes)
	$(DC) -f $(DC_FILE) down

down-v: ## Stop all services AND remove volumes (destructive!)
	$(DC) -f $(DC_FILE) down -v
	@echo "⚠ Volumes removed"

logs: ## Tail logs for all services
	$(DC) -f $(DC_FILE) logs -f --tail=100

logs-api: ## Tail API logs
	$(DC) -f $(DC_FILE) logs -f --tail=100 api

logs-worker: ## Tail worker logs
	$(DC) -f $(DC_FILE) logs -f --tail=100 worker

# ── Migrations ────────────────────────────────────────────────────────────────
migrate: ## Apply all pending Alembic migrations
	$(UV) run alembic upgrade head

migrate-create: ## Create a new migration (usage: make migrate-create MSG="add foo table")
	$(UV) run alembic revision --autogenerate -m "$(MSG)"

migrate-history: ## Show migration history
	$(UV) run alembic history --verbose

migrate-rollback: ## Roll back one migration
	$(UV) run alembic downgrade -1

# ── Shells ────────────────────────────────────────────────────────────────────
shell-api: ## Open a shell in the running API container
	$(DC) -f $(DC_FILE) exec api /bin/bash

shell-db: ## Open a psql shell in the Postgres container
	$(DC) -f $(DC_FILE) exec postgres psql -U autopilot -d autopilot

shell-redis: ## Open redis-cli in the Redis container
	$(DC) -f $(DC_FILE) exec redis redis-cli

# ── Local dev server (without Docker) ────────────────────────────────────────
dev-api: ## Run FastAPI dev server locally (needs Postgres + Redis running)
	PYTHONPATH=libs/core:apps/api \
	$(UV) run uvicorn llm_autopilot_api.main:app \
	  --reload \
	  --host 0.0.0.0 \
	  --port 8000

dev-worker: ## Run Celery worker locally
	PYTHONPATH=libs/core:apps/api:apps/worker \
	$(UV) run celery -A llm_autopilot_worker.main worker \
	  -Q verification,retraining \
	  -c 2 \
	  --loglevel DEBUG

dev-beat: ## Run Celery beat scheduler locally (one instance only!)
	PYTHONPATH=libs/core:apps/api:apps/worker \
	$(UV) run celery -A llm_autopilot_worker.main beat \
	  --loglevel DEBUG

dev-flower: ## Run Flower locally
	PYTHONPATH=libs/core:apps/api:apps/worker \
	$(UV) run celery -A llm_autopilot_worker.main flower --port=5555

# ── Ollama ────────────────────────────────────────────────────────────────────
ollama-pull: ## Pull required Ollama models (run after `make up`)
	$(DC) -f $(DC_FILE) exec ollama ollama pull llama3.1
	$(DC) -f $(DC_FILE) exec ollama ollama pull mistral
	@echo "✓ Ollama models pulled"

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove Python cache files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null; true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	@echo "✓ Cache cleaned"

reset: down-v clean ## Full reset: stop services, remove volumes, clean cache
	@echo "✓ Full reset complete — run 'make dev-install && make up' to start fresh"
