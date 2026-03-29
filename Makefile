# =============================================================================
# Telecom Customer Churn MLOps — Makefile
# Consolidates dev, test, pipeline, and Docker workflows into a single entry point.
# Rule 2.10 — Onboarding Hygiene: Makefile is a mandatory Day-One artifact.
#
# Usage:
#   make lint          Run ruff linting and format check
#   make typecheck     Run pyright static type checking
#   make test          Run pytest with coverage gate
#   make pipeline      Run the full DVC pipeline (dvc repro)
#   make build         Build all Docker images
#   make up            Start all services in detached mode
#   make down          Stop and remove containers
#   make logs          Stream logs from all services
#   make health        Check health endpoints on both API services
#   make validate      Run the full Multi-Point Validation Gate
# =============================================================================

.PHONY: lint typecheck test pipeline build up down down-v logs \
        health health-embed health-pred validate clean help

# ---------------------------------------------------------------------------
# Code Quality
# ---------------------------------------------------------------------------

lint:
	@echo ">>> Running ruff lint..."
	uv run ruff check src/ tests/
	@echo ">>> Running ruff format check..."
	uv run ruff format --check src/ tests/
	@echo ">>> Lint passed."

typecheck:
	@echo ">>> Running pyright..."
	uv run pyright
	@echo ">>> Type check passed."

# ---------------------------------------------------------------------------
# Testing
# ---------------------------------------------------------------------------

test:
	@echo ">>> Running pytest with coverage gate (65%)..."
	uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-fail-under=65
	@echo ">>> Tests passed."

test-fast:
	@echo ">>> Running pytest (no coverage)..."
	uv run pytest tests/ -v

# ---------------------------------------------------------------------------
# DVC Pipeline
# ---------------------------------------------------------------------------

pipeline:
	@echo ">>> Reproducing DVC pipeline..."
	uv run dvc repro
	@echo ">>> Pipeline complete."

pipeline-force:
	@echo ">>> Force-reproducing DVC pipeline (ignoring cache)..."
	uv run dvc repro --force

dvc-status:
	@echo ">>> DVC pipeline status..."
	uv run dvc status

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

build:
	@echo ">>> Building all Docker images..."
	docker compose build
	@echo ">>> Build complete."

build-no-cache:
	@echo ">>> Building all Docker images (no cache)..."
	docker compose build --no-cache

up:
	@echo ">>> Starting all services (detached)..."
	docker compose up -d
	@echo ">>> Services started. Run 'make logs' to stream output."

up-build:
	@echo ">>> Building and starting all services..."
	docker compose up --build -d

down:
	@echo ">>> Stopping and removing containers..."
	docker compose down

down-v:
	@echo ">>> Stopping containers and removing named volumes..."
	docker compose down -v

logs:
	docker compose logs -f

logs-embed:
	docker compose logs -f embedding-service

logs-pred:
	docker compose logs -f prediction-api

# ---------------------------------------------------------------------------
# Health Checks
# ---------------------------------------------------------------------------

health-embed:
	@echo ">>> Embedding service health (port 8001)..."
	@curl -sf http://localhost:8001/v1/health | python -m json.tool || \
		(echo "ERROR: Embedding service not responding." && exit 1)

health-pred:
	@echo ">>> Prediction API health (port 8000)..."
	@curl -sf http://localhost:8000/v1/health | python -m json.tool || \
		(echo "ERROR: Prediction API not responding." && exit 1)

health: health-embed health-pred
	@echo ">>> All services healthy."

# ---------------------------------------------------------------------------
# Multi-Point Validation Gate (Rule 6.4)
# ---------------------------------------------------------------------------

validate:
	@echo ">>> Running Multi-Point Validation Gate..."
	@bash validate_system.sh

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

clean:
	@echo ">>> Removing Python cache files..."
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo ">>> Clean complete."

help:
	@echo ""
	@echo "Telecom Churn MLOps — Available targets:"
	@echo ""
	@echo "  Code Quality:"
	@echo "    lint          Run ruff check + format check"
	@echo "    typecheck     Run pyright"
	@echo ""
	@echo "  Testing:"
	@echo "    test          Run pytest with 65% coverage gate"
	@echo "    test-fast     Run pytest without coverage"
	@echo ""
	@echo "  DVC Pipeline:"
	@echo "    pipeline      uv run dvc repro"
	@echo "    pipeline-force  dvc repro --force"
	@echo "    dvc-status    Check pipeline cache status"
	@echo ""
	@echo "  Docker:"
	@echo "    build         docker compose build"
	@echo "    up            docker compose up -d"
	@echo "    up-build      docker compose up --build -d"
	@echo "    down          docker compose down"
	@echo "    down-v        docker compose down -v"
	@echo "    logs          Stream all service logs"
	@echo "    logs-embed    Stream embedding-service logs"
	@echo "    logs-pred     Stream prediction-api logs"
	@echo ""
	@echo "  Health:"
	@echo "    health        Check both API services"
	@echo "    health-embed  Check embedding-service only"
	@echo "    health-pred   Check prediction-api only"
	@echo ""
	@echo "  Validation:"
	@echo "    validate      Run full Rule 6.4 validation gate"
	@echo ""
	@echo "  Utilities:"
	@echo "    clean         Remove Python cache files"
	@echo "    help          Show this message"
	@echo ""