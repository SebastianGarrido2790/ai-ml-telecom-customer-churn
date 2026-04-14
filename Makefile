# =============================================================================
# Telecom Customer Churn MLOps — Makefile
# Consolidates dev, test, pipeline, Docker, and AWS deployment workflows.
# Onboarding Hygiene: Makefile is a mandatory Day-One artifact.
# =============================================================================

.PHONY: lint typecheck test pipeline build up down down-v logs \
        health health-embed health-pred pre-commit-install pre-commit \
        ecr-login ecr-push deploy-embed deploy-pred deploy-gradio deploy \
        artifacts-push artifacts-pull validate clean help

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
# Pre-commit
# ---------------------------------------------------------------------------

pre-commit-install:
	@echo ">>> Installing pre-commit hooks..."
	uv run pre-commit install
	@echo ">>> Pre-commit hooks installed. Hooks run on every 'git commit'."

pre-commit:
	@echo ">>> Running pre-commit on all files..."
	uv run pre-commit run --all-files

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
# Docker (local)
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

logs-gradio:
	docker compose logs -f gradio-ui

# ---------------------------------------------------------------------------
# Health Checks
# ---------------------------------------------------------------------------

health-embed:
	@echo ">>> Embedding service health (port 8001)..."
	@curl -sf http://localhost:8001/v1/health | python3 -m json.tool || \
		(echo "ERROR: Embedding service not responding." && exit 1)

health-pred:
	@echo ">>> Prediction API health (port 8000)..."
	@curl -sf http://localhost:8000/v1/health | python3 -m json.tool || \
		(echo "ERROR: Prediction API not responding." && exit 1)

health-gradio:
	@echo ">>> Gradio UI health (port 7860)..."
	@curl -sf http://localhost:7860/ > /dev/null || \
		(echo "ERROR: Gradio UI not responding." && exit 1)

health: health-embed health-pred health-gradio
	@echo ">>> All services healthy."

# ---------------------------------------------------------------------------
# AWS — ECR
# ---------------------------------------------------------------------------

ecr-login:
	@echo ">>> Logging into Amazon ECR..."
	aws ecr get-login-password --region $(AWS_DEFAULT_REGION) | \
		docker login --username AWS --password-stdin $(ECR_REGISTRY)

ecr-push: ecr-login
	@echo ">>> Pushing all images to ECR..."
	@for svc in embedding-service prediction-api gradio-ui mlflow-server; do \
		echo "  Pushing telecom-churn/$$svc..."; \
		docker tag telecom-churn/$$svc:latest $(ECR_REGISTRY)/telecom-churn/$$svc:latest; \
		docker push $(ECR_REGISTRY)/telecom-churn/$$svc:latest; \
	done
	@echo ">>> ECR push complete."

# ---------------------------------------------------------------------------
# AWS — ECS Deploy (individual services)
# ---------------------------------------------------------------------------

deploy-embed:
	@echo ">>> Deploying embedding-service to ECS..."
	aws ecs update-service \
		--cluster $(ECS_CLUSTER) \
		--service telecom-churn-embedding-service \
		--force-new-deployment \
		--region $(AWS_DEFAULT_REGION)
	@echo ">>> Waiting for embedding-service stability..."
	aws ecs wait services-stable \
		--cluster $(ECS_CLUSTER) \
		--services telecom-churn-embedding-service \
		--region $(AWS_DEFAULT_REGION)

deploy-pred:
	@echo ">>> Deploying prediction-api to ECS..."
	aws ecs update-service \
		--cluster $(ECS_CLUSTER) \
		--service telecom-churn-prediction-api \
		--force-new-deployment \
		--region $(AWS_DEFAULT_REGION)
	@aws ecs wait services-stable \
		--cluster $(ECS_CLUSTER) \
		--services telecom-churn-prediction-api \
		--region $(AWS_DEFAULT_REGION)

deploy-gradio:
	@echo ">>> Deploying gradio-ui to ECS..."
	aws ecs update-service \
		--cluster $(ECS_CLUSTER) \
		--service telecom-churn-gradio-ui \
		--force-new-deployment \
		--region $(AWS_DEFAULT_REGION)
	aws ecs wait services-stable \
		--cluster $(ECS_CLUSTER) \
		--services telecom-churn-gradio-ui \
		--region $(AWS_DEFAULT_REGION)

# Full sequential deploy — mirrors CD workflow service ordering
deploy: deploy-embed deploy-pred deploy-gradio
	@echo ">>> Full deployment complete."

# ---------------------------------------------------------------------------
# AWS — Artifacts S3 Sync (Decision L1)
# ---------------------------------------------------------------------------

artifacts-push:
	@echo ">>> Pushing DVC artifacts to S3 ($(ARTIFACTS_S3_BUCKET))..."
	aws s3 sync artifacts/ s3://$(ARTIFACTS_S3_BUCKET)/artifacts/ \
		--exclude "*.gitignore" \
		--region $(AWS_DEFAULT_REGION)
	@echo ">>> Artifacts pushed."

artifacts-pull:
	@echo ">>> Pulling DVC artifacts from S3 ($(ARTIFACTS_S3_BUCKET))..."
	aws s3 sync s3://$(ARTIFACTS_S3_BUCKET)/artifacts/ artifacts/ \
		--region $(AWS_DEFAULT_REGION)
	@echo ">>> Artifacts pulled."

# ---------------------------------------------------------------------------
# AWS — LocalStack Emulation
# ---------------------------------------------------------------------------

localstack-seed:
	@echo ">>> Creating mock bucket and syncing artifacts to LocalStack..."
	docker exec localstack awslocal s3 mb s3://telecom-churn-artifacts-local || true
	docker exec localstack awslocal s3 sync /workspace/artifacts/ s3://telecom-churn-artifacts-local/artifacts/
	@echo ">>> LocalStack seed complete."

localstack-deploy:
	@echo ">>> Deploying services to LocalStack ECS (mock)..."
	@echo "For actual deployment logic to LocalStack, see .github/workflows/cd.yml. In local dev, Docker Compose replaces this."

# ---------------------------------------------------------------------------
# Multi-Point Validation Gate
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
	@echo "    lint              ruff check + format check"
	@echo "    typecheck         pyright"
	@echo "    pre-commit        Run all pre-commit hooks on all files"
	@echo "    pre-commit-install  Install git hooks (run once per machine)"
	@echo ""
	@echo "  Testing:"
	@echo "    test              pytest with 65% coverage gate"
	@echo "    test-fast         pytest without coverage"
	@echo ""
	@echo "  DVC Pipeline:"
	@echo "    pipeline          dvc repro"
	@echo "    pipeline-force    dvc repro --force"
	@echo "    dvc-status        Check pipeline cache status"
	@echo ""
	@echo "  Docker (local):"
	@echo "    build             docker compose build"
	@echo "    up                docker compose up -d"
	@echo "    up-build          docker compose up --build -d"
	@echo "    down / down-v     Stop containers (with/without volumes)"
	@echo "    logs              Stream all service logs"
	@echo ""
	@echo "  Health:"
	@echo "    health            Check all running services"
	@echo "    health-embed      Check embedding-service only"
	@echo "    health-pred       Check prediction-api only"
	@echo "    health-gradio     Check gradio-ui only"
	@echo ""
	@echo "  AWS — ECR:"
	@echo "    ecr-login         Authenticate Docker to ECR"
	@echo "    ecr-push          Tag + push all images to ECR"
	@echo ""
	@echo "  AWS — ECS Deploy:"
	@echo "    deploy-embed      Rolling update: embedding-service"
	@echo "    deploy-pred       Rolling update: prediction-api"
	@echo "    deploy-gradio     Rolling update: gradio-ui"
	@echo "    deploy            Full sequential deploy (all three)"
	@echo ""
	@echo "  AWS — Artifacts:"
	@echo "    artifacts-push    Sync artifacts/ → S3"
	@echo "    artifacts-pull    Sync S3 → artifacts/"
	@echo ""
	@echo "  Validation:"
	@echo "    validate          Multi-Point Validation Gate"
	@echo ""
	@echo "  Utilities:"
	@echo "    clean             Remove Python cache files"
	@echo "    help              Show this message"
	@echo ""
