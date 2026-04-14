## Phase 8: CI/CD & Cloud Deployment — Full Implementation Plan

> Confirmed Decisions: **Decision 1 (I1 — OIDC)**, **Decision 2 (J1 — rolling update)**, **Decision 3 (K2 — three services)**, **Decision 4 (L1 — S3 fetch at startup)**.

---

### Files to Deliver (10 total)

| # | File | Type |
|---|---|---|
| 1 | `.github/workflows/ci.yml` | New |
| 2 | `.github/workflows/cd.yml` | New |
| 3 | `task-definitions/embedding-service.json` | New |
| 4 | `task-definitions/prediction-api.json` | New |
| 5 | `task-definitions/mlflow-server.json` | New |
| 6 | `task-definitions/gradio-ui.json` | New |
| 7 | `.pre-commit-config.yaml` | Create at project root |
| 8 | `docker-compose.yaml` | Modified (activate gradio-ui + add ECR image tags) |
| 9 | `.env.example` | Modified (add AWS + ECR vars) |
| 10 | `Makefile` | Modified (add ECR push + ECS deploy targets) |
| 11 | `validate_system.sh` | Modified (add ECR connectivity pre-check) |

---

### CI/CD Pipeline Architecture

Two workflows following the **trunk-based development** model:

```
Every push / PR to any branch:
    ci.yml
    ├── Pillar 1: ruff check + ruff format --check + pyright
    ├── Pillar 2: pytest --cov=src --cov-fail-under=65
    └── Result: blocks merge if any pillar fails

Merge to main only:
    cd.yml
    ├── Stage 1: Build all four Docker images (parallel matrix)
    ├── Stage 2: Vulnerability scan (Docker Scout per image)
    ├── Stage 3: Push to AWS ECR (all four repos)
    └── Stage 4: Deploy to ECS Fargate (rolling update, service by service)
                  embedding-service → prediction-api → mlflow-server → gradio-ui
```

---

### GitHub Actions Workflow Design

**`ci.yml` — Continuous Integration**

Triggers: `push` (all branches) + `pull_request` (targeting `main`).

```
jobs:
  quality:                        # single job — fast feedback
    runs-on: ubuntu-latest
    steps:
      - checkout
      - setup uv (cached)
      - uv sync --all-extras
      - ruff check src/ tests/
      - ruff format --check src/ tests/
      - pyright
      - pytest --cov=src --cov-fail-under=65 --tb=short
```

Single job by design — the quality checks are fast (< 3 min) and share the same venv. Splitting into parallel jobs adds matrix overhead that exceeds the time saved.

**`cd.yml` — Continuous Deployment**

Triggers: `push` to `main` only (post-merge).

```
jobs:
  build-and-push:                 # parallel matrix over 3 services
    strategy:
      matrix:
        service: [embedding-service, prediction-api, gradio-ui]
    steps:
      - checkout
      - configure AWS credentials (OIDC)
      - login to ECR
      - build image (context: ., dockerfile: docker/{service}/Dockerfile)
      - docker scout CVEs (fail on critical)
      - push to ECR with two tags: {SHA} + latest

  deploy:
    needs: build-and-push
    steps:
      - download task definitions from repo
      - register new task definition revision per service (with new image SHA)
      - update ECS service (rolling update, wait for stability)
      - run validate_system.sh health check against deployed endpoints
```

---

### ECS Task Definition Design

Each task definition follows a consistent structure:

```json
{
  "family": "telecom-churn-{service}",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",           // 0.5 vCPU
  "memory": "1024",       // 1 GB (embedding-service: 2048 for SentenceTransformer)
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn":      "arn:aws:iam::ACCOUNT_ID:role/ecsTaskRole",
  "containerDefinitions": [{
    "name": "{service}",
    "image": "ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com/telecom-churn/{service}:latest",
    "portMappings": [{"containerPort": PORT, "protocol": "tcp"}],
    "environment": [...],   // service-specific env vars
    "secrets": [...],       // SSM Parameter Store references for sensitive values
    "logConfiguration": {   // CloudWatch Logs
      "logDriver": "awslogs",
      "options": {
        "awslogs-group": "/ecs/telecom-churn/{service}",
        "awslogs-region": "us-east-1",
        "awslogs-stream-prefix": "ecs"
      }
    },
    "healthCheck": {...}
  }]
}
```

**Embedding service** gets `"cpu": "1024", "memory": "2048"` — the SentenceTransformer model is baked into the image but still requires ~1.3 GB resident memory during inference.

**Model artifacts on Fargate:** ECS Fargate cannot use local bind mounts. Artifacts are stored in S3 and fetched at container startup via an init container or startup script. This is the key infrastructure difference from local Docker Compose. The task definition includes an `ARTIFACTS_S3_BUCKET` environment variable and the `taskRoleArn` grants `s3:GetObject` permissions on that bucket.

---

### Required GitHub Secrets

```
AWS_ROLE_ARN          IAM Role ARN for OIDC federation
AWS_ACCOUNT_ID        12-digit AWS account ID
AWS_REGION            e.g., us-east-1
ECR_REGISTRY          ACCOUNT_ID.dkr.ecr.REGION.amazonaws.com
ARTIFACTS_S3_BUCKET   S3 bucket name containing DVC-tracked artifacts
ECS_CLUSTER           ECS cluster name
```

No `AWS_ACCESS_KEY_ID` or `AWS_SECRET_ACCESS_KEY` — OIDC only.

---

### Pre-requisites (AWS infrastructure to create before running CD)

These are one-time setup steps outside the CI/CD code:

1. **ECR repositories:** Three repos — `telecom-churn/embedding-service`, `telecom-churn/prediction-api`, `telecom-churn/gradio-ui`.
2. **S3 bucket:** One bucket for DVC artifacts. Run `aws s3 sync artifacts/ s3://bucket/artifacts/` once to seed it.
3. **ECS cluster:** One Fargate cluster (`telecom-churn-cluster`).
4. **IAM roles:** `ecsTaskExecutionRole` (AWS managed) + `ecsTaskRole` (custom, with S3 read permission).
5. **GitHub OIDC provider:** Add `token.actions.githubusercontent.com` as an identity provider in IAM. One-time setup.
6. **IAM Role for GitHub Actions:** Trust policy scoped to your repository, with `ecr:*`, `ecs:*`, and `s3:GetObject` permissions.

---

### One-time AWS setup (before first CD run)

```bash
# 1. Create ECR repositories
aws ecr create-repository --repository-name telecom-churn/embedding-service
aws ecr create-repository --repository-name telecom-churn/prediction-api
aws ecr create-repository --repository-name telecom-churn/gradio-ui
aws ecr create-repository --repository-name telecom-churn/mlflow-server

# 2. Seed S3 with current artifacts
make artifacts-push   # requires ARTIFACTS_S3_BUCKET in .env

# 3. Add GitHub OIDC provider in AWS IAM (one-time per account)
# Console: IAM → Identity providers → Add provider
# Provider URL: https://token.actions.githubusercontent.com
# Audience: sts.amazonaws.com

# 4. Create IAM Role for GitHub Actions with trust policy for your repo
# Then add to GitHub: Settings → Secrets → AWS_ROLE_ARN
```

### One-time developer setup (per machine)

```bash
# Install pre-commit hooks — runs automatically on every git commit after this
make pre-commit-install

# Verify hooks work
make pre-commit
```

### What each pre-commit hook catches locally

| Hook | Blocks |
|---|---|
| `ruff` + `ruff-format` | Lint errors, import order, formatting — same as Pillar 1 CI |
| `pyright` | Type errors across all staged files |
| `trailing-whitespace`, `end-of-file-fixer` | Cosmetic issues before review |
| `check-yaml`, `check-json`, `check-toml` | Syntax errors in config files |
| `detect-private-key` | Accidental credential commits |
| `check-added-large-files` | Accidental model/data file commits (> 5 MB) |
| `no-artifacts-in-git` | Direct `artifacts/*.pkl|csv` commits — forces DVC workflow |
| `no-env-files` | `.env` commits — enforces `.env.example` pattern |

### Task definition placeholders to replace

Each JSON contains these literal strings that must be replaced before registering with ECS:

- `ACCOUNT_ID` → your 12-digit AWS account ID
- `REGION` → your AWS region (e.g., `us-east-1`)
- `REPLACE_WITH_BUCKET_NAME` → your S3 artifact bucket name
- `REPLACE_WITH_EMBEDDING_SERVICE_INTERNAL_DNS` → the internal ALB or service discovery DNS for `embedding-service`
- `REPLACE_WITH_PREDICTION_API_INTERNAL_URL` → the internal URL for `prediction-api`

---

## CD Workflow

```yaml
# =============================================================================
# CD — Continuous Deployment
# Triggers only on push to main (post-merge from a reviewed PR).
# Builds, scans, pushes to ECR, and deploys to ECS Fargate (rolling update).
#
# Deployment scope (Decision K2):
#   embedding-service  → ECR + ECS
#   prediction-api     → ECR + ECS
#   gradio-ui          → ECR + ECS
#   mlflow-server      → ECR push only (ECS deployment deferred to Phase 9)
#
# Authentication: OIDC federation — no long-lived AWS credentials (Decision I1).
# Artifact delivery: S3 fetch at container startup (Decision L1).
# Deployment strategy: ECS rolling update (Decision J1).
# =============================================================================

name: CD

on:
  push:
    branches:
      - main

# Only one deployment may run at a time to prevent race conditions.
concurrency:
  group: cd-main
  cancel-in-progress: false

env:
  AWS_REGION:       ${{ secrets.AWS_REGION }}
  ECR_REGISTRY:     ${{ secrets.ECR_REGISTRY }}
  ECS_CLUSTER:      ${{ secrets.ECS_CLUSTER }}

jobs:
  # ---------------------------------------------------------------------------
  # Stage 1: Build, scan, and push all images in parallel (matrix).
  # Each service builds from its own Dockerfile with the shared context root.
  # Images are tagged with both the Git SHA (immutable) and 'latest' (mutable).
  # ---------------------------------------------------------------------------
  build-and-push:
    name: "Build & Push — ${{ matrix.service }}"
    runs-on: ubuntu-latest
    timeout-minutes: 60

    strategy:
      fail-fast: false          # allow other images to complete if one fails
      matrix:
        include:
          - service: embedding-service
            dockerfile: docker/embedding_service/Dockerfile
            ecr_repo: telecom-churn/embedding-service
          - service: prediction-api
            dockerfile: docker/prediction_api/Dockerfile
            ecr_repo: telecom-churn/prediction-api
          - service: gradio-ui
            dockerfile: docker/gradio_ui/Dockerfile
            ecr_repo: telecom-churn/gradio-ui
          - service: mlflow-server
            dockerfile: docker/mlflow_server/Dockerfile
            ecr_repo: telecom-churn/mlflow-server

    permissions:
      id-token: write     # required for OIDC token request
      contents: read

    outputs:
      # Export the SHA-tagged image URI for each service so the deploy job
      # can reference the exact immutable image rather than 'latest'.
      embedding-service-image: ${{ steps.image-uri.outputs.embedding-service }}
      prediction-api-image:    ${{ steps.image-uri.outputs.prediction-api }}
      gradio-ui-image:         ${{ steps.image-uri.outputs.gradio-ui }}

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      # OIDC authentication — exchanges GitHub's JWT for temporary AWS credentials.
      # The IAM Role trust policy must allow this repository and ref.
      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume:    ${{ secrets.AWS_ROLE_ARN }}
          aws-region:        ${{ env.AWS_REGION }}
          role-session-name: github-cd-${{ github.run_id }}

      - name: Login to Amazon ECR
        id: ecr-login
        uses: aws-actions/amazon-ecr-login@v2

      # Set up Docker Buildx for efficient multi-layer caching.
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build and push image
        id: docker-build
        uses: docker/build-push-action@v6
        with:
          context: .
          file: ${{ matrix.dockerfile }}
          push: true
          tags: |
            ${{ env.ECR_REGISTRY }}/${{ matrix.ecr_repo }}:${{ github.sha }}
            ${{ env.ECR_REGISTRY }}/${{ matrix.ecr_repo }}:latest
          # Layer cache stored in ECR — no separate cache backend needed.
          cache-from: type=registry,ref=${{ env.ECR_REGISTRY }}/${{ matrix.ecr_repo }}:buildcache
          cache-to:   type=registry,ref=${{ env.ECR_REGISTRY }}/${{ matrix.ecr_repo }}:buildcache,mode=max
          labels: |
            org.opencontainers.image.source=${{ github.repositoryUrl }}
            org.opencontainers.image.revision=${{ github.sha }}
            org.opencontainers.image.created=${{ github.event.head_commit.timestamp }}

      # Docker Scout vulnerability scan — fails the job on CRITICAL CVEs.
      # HIGH CVEs are reported but non-blocking (portfolio context).
      - name: Docker Scout CVE scan
        uses: docker/scout-action@v1
        with:
          command: cves
          image: ${{ env.ECR_REGISTRY }}/${{ matrix.ecr_repo }}:${{ github.sha }}
          exit-code: true
          severity: critical        # block on CRITICAL only
          only-fixed: true          # ignore CVEs with no available fix
        env:
          DOCKER_SCOUT_HUB_USER:  ${{ secrets.DOCKER_HUB_USERNAME }}
          DOCKER_SCOUT_HUB_PASSWORD: ${{ secrets.DOCKER_HUB_TOKEN }}

      # Export image URI for the deploy job (only applicable services).
      - name: Export image URI
        id: image-uri
        run: |
          echo "${{ matrix.service }}=${{ env.ECR_REGISTRY }}/${{ matrix.ecr_repo }}:${{ github.sha }}" \
            >> "$GITHUB_OUTPUT"

  # ---------------------------------------------------------------------------
  # Stage 2: Deploy to ECS Fargate (rolling update, Decision J1).
  # Runs after ALL matrix builds succeed (needs: build-and-push).
  # mlflow-server is pushed to ECR but not deployed to ECS (Decision K2).
  # ---------------------------------------------------------------------------
  deploy:
    name: "Deploy to ECS Fargate"
    runs-on: ubuntu-latest
    timeout-minutes: 30
    needs: build-and-push

    permissions:
      id-token: write
      contents: read

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume:    ${{ secrets.AWS_ROLE_ARN }}
          aws-region:        ${{ env.AWS_REGION }}
          role-session-name: github-deploy-${{ github.run_id }}

      # -----------------------------------------------------------------------
      # embedding-service — deployed first (prediction-api depends on it)
      # -----------------------------------------------------------------------
      - name: Render ECS task definition — embedding-service
        id: task-def-embed
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition:    task-definitions/embedding-service.json
          container-name:     embedding-service
          image:              ${{ env.ECR_REGISTRY }}/telecom-churn/embedding-service:${{ github.sha }}

      - name: Deploy embedding-service to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition:    ${{ steps.task-def-embed.outputs.task-definition }}
          service:            telecom-churn-embedding-service
          cluster:            ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true
          wait-for-minutes:   10

      # -----------------------------------------------------------------------
      # prediction-api — deployed second (depends on embedding-service health)
      # -----------------------------------------------------------------------
      - name: Render ECS task definition — prediction-api
        id: task-def-pred
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition:    task-definitions/prediction-api.json
          container-name:     prediction-api
          image:              ${{ env.ECR_REGISTRY }}/telecom-churn/prediction-api:${{ github.sha }}

      - name: Deploy prediction-api to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition:    ${{ steps.task-def-pred.outputs.task-definition }}
          service:            telecom-churn-prediction-api
          cluster:            ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true
          wait-for-minutes:   10

      # -----------------------------------------------------------------------
      # gradio-ui — deployed last (depends on prediction-api health)
      # -----------------------------------------------------------------------
      - name: Render ECS task definition — gradio-ui
        id: task-def-gradio
        uses: aws-actions/amazon-ecs-render-task-definition@v1
        with:
          task-definition:    task-definitions/gradio-ui.json
          container-name:     gradio-ui
          image:              ${{ env.ECR_REGISTRY }}/telecom-churn/gradio-ui:${{ github.sha }}

      - name: Deploy gradio-ui to ECS
        uses: aws-actions/amazon-ecs-deploy-task-definition@v2
        with:
          task-definition:    ${{ steps.task-def-gradio.outputs.task-definition }}
          service:            telecom-churn-gradio-ui
          cluster:            ${{ env.ECS_CLUSTER }}
          wait-for-service-stability: true
          wait-for-minutes:   10

      # -----------------------------------------------------------------------
      # Post-deploy health validation
      # Runs validate_system.sh Pillar 4 against the ALB DNS endpoints.
      # ALB_DNS_PRED and ALB_DNS_GRADIO are set as GitHub Actions variables
      # (non-secret configuration) pointing to the ALB DNS names.
      # -----------------------------------------------------------------------
      - name: Post-deploy health check
        run: |
          echo "Checking prediction-api health..."
          curl -sf --max-time 10 \
            "http://${{ vars.ALB_DNS_PRED }}/v1/health" | python3 -m json.tool

          echo "Checking gradio-ui health..."
          curl -sf --max-time 10 \
            "http://${{ vars.ALB_DNS_GRADIO }}/" > /dev/null && \
            echo "Gradio UI: 200 OK"

      - name: Deployment summary
        run: |
          echo "=== Deployment complete ==="
          echo "Commit : ${{ github.sha }}"
          echo "Actor  : ${{ github.actor }}"
          echo "Time   : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
          echo "Images :"
          echo "  embedding-service : ${{ env.ECR_REGISTRY }}/telecom-churn/embedding-service:${{ github.sha }}"
          echo "  prediction-api    : ${{ env.ECR_REGISTRY }}/telecom-churn/prediction-api:${{ github.sha }}"
          echo "  gradio-ui         : ${{ env.ECR_REGISTRY }}/telecom-churn/gradio-ui:${{ github.sha }}"
```