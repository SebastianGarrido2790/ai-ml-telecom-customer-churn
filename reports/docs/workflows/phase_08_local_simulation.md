## Phase 8: CI/CD & Cloud Deployment (LocalStack Simulation)

> Confirmed Decisions: **Decision 1 (I1 — OIDC)**, **Decision 2 (J1 — rolling update)**, **Decision 3 (K2 — three services)**, **Decision 4 (L1 — S3 fetch at startup)**, **Decision 5 (M1 — LocalStack simulation)**.

---

### Mock Cloud Infrastructure Ready

The entire cloud deployment automation strategy will be containerized and sandboxed perfectly. The agents and ML models can act and deploy natively as if they're in Fargate without you ever having to open an AWS account window.

We need to build a cloud-agnostic simulation environment that handles:

- **Mock Cloud Infrastructure**: Using LocalStack 3.1.0 (Stable).
- **Automated Artifact Pulling**: Services can be starting-up with an "empty" state and pull their models directly from the cloud.
- **Environment Parity**: The logic used here will work in a real AWS ECS/Fargate deployment.

---

### Files to Deliver (11 total)

| # | File | Type |
|---|---|---|
| 1 | `.github/workflows/ci.yml` | New |
| 2 | `.github/workflows/cd.yml` | New (LocalStack configuration) |
| 3 | `task-definitions/embedding-service.json` | New |
| 4 | `task-definitions/prediction-api.json` | New |
| 5 | `task-definitions/mlflow-server.json` | New |
| 6 | `task-definitions/gradio-ui.json` | New |
| 7 | `.pre-commit-config.yaml` | New |
| 8 | `docker-compose.yaml` | Modified (add LocalStack and activate gradio-ui) |
| 9 | `.env.example` | Modified (add LocalStack endpoint vars) |
| 10 | `Makefile` | Modified (add LocalStack + AWS targets) |
| 11 | `validate_system.sh` | Modified |

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
    cd.yml (Using LocalStack Action)
    ├── Stage 1: Setup LocalStack emulator in GitHub Actions
    ├── Stage 2: Create local ECR repos, ECS clusters, S3 buckets
    ├── Stage 3: Build & Push all four Docker images to LocalStack ECR
    └── Stage 4: Deploy to LocalStack ECS Fargate (rolling update, service by service)
```

---

### GitHub Actions Workflow Design

The **GitHub Actions CI/CD workflow** is will be completely wired to run a full production-like deployment entirely using your mock LocalStack cloud.

Here's a breakdown of the automated workflows safely built for this phase:

#### 1. The CI Pipeline (`.github/workflows/ci.yml`)
Operating on trunk-based development principles, this strictly gates any code merged into `main`. It implements a two-pillar validation process directly simulating your local `validate_system.sh`:
*   **Pillar 1 (Code Quality):** Runs `ruff check`, `ruff format`, and `pyright` to ensure strict typing and code style.
*   **Pillar 2 (Unit Testing):** Runs `pytest` and enforces a hard `65%` test coverage threshold. 

#### 2. The CD Pipeline & Mock Cloud (`.github/workflows/cd.yml`)
The engine of your new continuous deployment process. On pushes to `main`, this workflow now completely isolates your architecture from real AWS costs while executing full deployments:
*   **Stage 1 - Provisioning:** It sets up `LocalStack` 3.1.0 in the GitHub runner natively and immediately provisions a mock S3 bucket, three ECR registries, and a mock ECS default cluster.
*   **Stage 2 - Building & Pushing:** It iterates over your `prediction-api`, `embedding-service`, and `gradio-ui` containers, tagging them and pushing them directly to the mock LocalStack ECR (`localhost:4566/telecom-churn/...`).
*   **Stage 3 - Mock Fargate Deployment:** Uses dynamic `sed` commands to hijack your `task-definitions/*.json` (rewriting the real AWS regions and account IDs to your mock sandbox), then calls `awslocal ecs create-service` with Fargate mock networking.

**`ci.yml` — Continuous Integration**
(Remains identical to the original plan, checking code quality and tests natively.)

**`cd.yml` — Continuous Deployment (LocalStack)**
Triggers: `push` to `main` only.
```yaml
jobs:
  localstack-deploy:
    runs-on: ubuntu-latest
    steps:
      - checkout
      - Start LocalStack: `localstack/setup-localstack@v0.2.2`
      - Setup AWS Credentials (dummy): `AWS_ACCESS_KEY_ID=test, AWS_SECRET_ACCESS_KEY=test, AWS_DEFAULT_REGION=us-east-1`
      - Init LocalStack Resources: Create ECR repos, S3 bucket, ECS Cluster via `aws --endpoint-url=http://localhost:4566 ...`
      - Build images
      - Push images to `localhost:4566`
      - Register task definitions
      - Update ECS services
```

---

### ECS Task Definition Design

Remains largely the same, but S3 fetching is done targeting the localstack endpoint.

**Model artifacts on Fargate:** The ECS containers fetch artifacts via `aws s3 sync ... --endpoint-url=http://localstack:4566`.

---

### LocalStack Specific Setup

1. **Docker Compose:** Append LocalStack to the main stack to allow local testing of the emulation.
2. **Docker Entrypoints:** When running `aws s3 sync`, detect if `AWS_ENDPOINT_URL` is set, and if so, append `--endpoint-url $AWS_ENDPOINT_URL` to the command.

---

## Implementation Results

The application files were updated to support the LocalStack simulation! 

Here is what was implemented:
1. **`.env.example`**: Configured with the dummy `AWS_ACCESS_KEY_ID=test`, `AWS_SECRET_ACCESS_KEY=test`, `AWS_DEFAULT_REGION=us-east-1`, and `AWS_ENDPOINT_URL=http://localstack:4566`.
2. **`docker-compose.yaml`**: 
   - Added the `localstack` container definition.
   - Wired the `AWS_ENDPOINT_URL` and `ARTIFACTS_S3_BUCKET` environment variables into `embedding-service`, `prediction-api`, and `gradio-ui`.
3. **`docker/entrypoint.sh`**: Created a centralized shell script that optionally synchronizes artifacts from the S3 bucket via the `aws s3 sync` command before starting up the Uvicorn/Gradio servers. It explicitly checks for the `AWS_ENDPOINT_URL` to easily route traffic to LocalStack.
4. **`Dockerfiles` (x3)**: Updated the building phase for `embedding_service`, `prediction_api`, and `gradio_ui` to install `awscli` and copy in the new `entrypoint.sh`.

### Steps to Launch LocalStack
To test the environment, let's now start the emulator and see if your local cloud behaves properly. 

1. **Update your `.env`**: Copy the new variables from `.env.example` to your `.env` file (if you haven't already).
2. **Start the containers** (This will rebuild the Docker images and start LocalStack):
   ```bash
   docker compose up --build -d
   ```
3. **Configure the LocalStack CLI**:
   Since we are using LocalStack in Docker, we use a helper function to run the CLI *inside* the container (so you don't need to install anything on your host).
   
   **For PowerShell (pwsh):**
   ```powershell
   function awslocal { docker exec -it localstack awslocal $args }
   ```
   **For Bash/Zsh:**
   ```bash
   alias awslocal="docker exec -it localstack awslocal"
   ```

4. **Seed the Emulator** (Create a bucket and push your current models into it):
   ```powershell
   # Create the bucket
   awslocal s3 mb s3://telecom-churn-artifacts-local
   
   # Sync the models to the mock S3 folder (/workspace is mapped to your project root)
   awslocal s3 sync /workspace/artifacts/ s3://telecom-churn-artifacts-local/artifacts/
   ```

5. **Restart Services to Pull from S3**:
   Once the bucket is seeded, restart your app containers so their `entrypoint.sh` scripts can pull the files:
   ```bash
   docker compose restart prediction-api embedding-service gradio-ui
   ```
