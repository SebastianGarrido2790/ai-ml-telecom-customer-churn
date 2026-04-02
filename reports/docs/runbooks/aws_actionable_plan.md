## AWS Actionable Plan (LocalStack Simulation)

Here is the updated actionable plan to complete **Phase 8: CI/CD & Cloud Deployment** using **LocalStack** to emulate AWS infrastructure. This approach requires zero cloud costs and zero credit cards, but teaches the same architectural patterns.

### Step 1: LocalStack Setup & Infrastructure Emulation

1. **Start LocalStack Environment**:
   We will add `localstack` to our `docker-compose.yaml` file so it runs continually during development alongside our app containers. Alternatively, run it detached manually:
   ```bash
   docker run --rm -d -p 4566:4566 -p 4510-4559:4510-4559 --name localstack localstack/localstack
   ```

2. **Configure Dummy AWS Credentials**:
   LocalStack doesn't require real IAM credentials, but the AWS CLI requires the variables to be set.
   ```bash
   export AWS_ACCESS_KEY_ID=test
   export AWS_SECRET_ACCESS_KEY=test
   export AWS_DEFAULT_REGION=us-east-1
   # Create a helpful alias
   alias awslocal="aws --endpoint-url=http://localhost:4566"
   ```

3. **Create Emulated ECR Repositories**:
   ```bash
   awslocal ecr create-repository --repository-name telecom-churn/embedding-service
   awslocal ecr create-repository --repository-name telecom-churn/prediction-api
   awslocal ecr create-repository --repository-name telecom-churn/gradio-ui
   awslocal ecr create-repository --repository-name telecom-churn/mlflow-server
   ```

4. **Create Emulated S3 Bucket**:
   ```bash
   awslocal s3 mb s3://telecom-churn-artifacts-local
   # Seed it once
   awslocal s3 sync artifacts/ s3://telecom-churn-artifacts-local/artifacts/
   ```

5. **Create Emulated ECS Cluster**:
   ```bash
   awslocal ecs create-cluster --cluster-name telecom-churn-cluster
   ```

### Step 2: Update Application Files

- **Docker Compose (`docker-compose.yaml`)**:
  Add a `localstack` service block. Ensure the other services have `AWS_ENDPOINT_URL=http://localstack:4566` in their environment variables.

- **Entrypoint Script (S3 Fetch)**:
  We need to download artifacts from S3 at startup. We'll use an `entrypoint.sh` for our containers that respects the `AWS_ENDPOINT_URL`:
  ```bash
  ENDPOINT_FLAG=""
  if [ ! -z "$AWS_ENDPOINT_URL" ]; then
      ENDPOINT_FLAG="--endpoint-url $AWS_ENDPOINT_URL"
  fi
  aws s3 sync s3://${ARTIFACTS_S3_BUCKET}/artifacts/feature_engineering/ /app/artifacts/feature_engineering/ $ENDPOINT_FLAG
  aws s3 sync s3://${ARTIFACTS_S3_BUCKET}/artifacts/model_training/ /app/artifacts/model_training/ $ENDPOINT_FLAG
  ```

- **Makefile**:
  Update AWS targets (`make ecr-push`, `make deploy`) to use the localstack endpoint.

- **Task Definitions**:
  Create JSON representations of our services pointing to dummy ARNs representing the localstack resources. Be aware that LocalStack's ECS implementation has varying feature completeness in the free tier, but is enough for testing basic definitions.

### Step 3: Configure GitHub Actions for LocalStack

We will build the CI/CD pipeline (`.github/workflows/cd.yml`) using the `localstack/setup-localstack` GitHub Action.
This will spin up an ephemeral LocalStack instance inside the GitHub runner, execute the infrastructure creation commands, build the Docker images, and simulate an ECS deployment.

---

# AWS Actionable Plan (Real AWS)

Here's a clear, actionable plan to complete **Phase 8: CI/CD & Cloud Deployment** based on the confirmed decisions (I1 OIDC, J1 rolling updates, K2 three services, L1 S3 fetch at startup) and the provided documents.

### Step 1: One-Time AWS Infrastructure Setup (Do This First)
Before any GitHub workflow runs, create these resources manually (or via IaC in a future phase). Use the AWS Console or CLI.

1. **Create ECR Repositories** (one per service):
   ```bash
   aws ecr create-repository --repository-name telecom-churn/embedding-service --region us-east-1
   aws ecr create-repository --repository-name telecom-churn/prediction-api --region us-east-1
   aws ecr create-repository --repository-name telecom-churn/gradio-ui --region us-east-1
   aws ecr create-repository --repository-name telecom-churn/mlflow-server --region us-east-1
   ```

2. **Create S3 Bucket for Artifacts** (DVC models/preprocessors):
   ```bash
   aws s3 mb s3://your-unique-telecom-churn-artifacts-bucket --region us-east-1
   ```
   Then seed it once:
   ```bash
   # From your local machine (after make artifacts-push is ready)
   aws s3 sync artifacts/ s3://your-unique-telecom-churn-artifacts-bucket/artifacts/ --exclude "*.gitignore"
   ```

3. **Create ECS Cluster** (Fargate):
   ```bash
   aws ecs create-cluster --cluster-name telecom-churn-cluster --region us-east-1
   ```

4. **IAM Roles**:
   - Use the AWS-managed `ecsTaskExecutionRole` (or create one with `AmazonECSTaskExecutionRolePolicy`).
   - Create a custom `ecsTaskRole` with at minimum:
     - `s3:GetObject` (and `s3:ListBucket`) on your artifacts bucket.
     - Optionally `logs:*` if needed beyond execution role.

5. **GitHub OIDC Provider** (one-time per AWS account):
   - In AWS IAM → Identity providers → Add provider.
   - Provider URL: `https://token.actions.githubusercontent.com`
   - Audience: `sts.amazonaws.com`

6. **IAM Role for GitHub Actions** (critical for OIDC — Decision I1):
   Create a role with:
   - Trust policy (replace `YOUR_GITHUB_ORG` and `YOUR_REPO_NAME`):
     ```json
     {
       "Version": "2012-10-17",
       "Statement": [
         {
           "Effect": "Allow",
           "Principal": {
             "Federated": "arn:aws:iam::YOUR_AWS_ACCOUNT_ID:oidc-provider/token.actions.githubusercontent.com"
           },
           "Action": "sts:AssumeRoleWithWebIdentity",
           "Condition": {
             "StringEquals": {
               "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
             },
             "StringLike": {
               "token.actions.githubusercontent.com:sub": "repo:YOUR_GITHUB_ORG/YOUR_REPO_NAME:ref:refs/heads/main"
             }
           }
         }
       ]
     }
     ```
   - Attach policies for: `AmazonEC2ContainerRegistryFullAccess`, `AmazonECS_FullAccess`, `AmazonS3ReadOnlyAccess` (scoped to your bucket), and any needed for task registration.

   Add the role ARN as GitHub repo secret: `AWS_ROLE_ARN`.

7. **GitHub Repository Settings**:
   - Secrets: `AWS_ROLE_ARN`, `AWS_ACCOUNT_ID`, `AWS_REGION`, `ECR_REGISTRY` (e.g., `123456789012.dkr.ecr.us-east-1.amazonaws.com`), `ARTIFACTS_S3_BUCKET`, `ECS_CLUSTER`.
   - Variables (non-secret): `ALB_DNS_PRED` and `ALB_DNS_GRADIO` (once you create ALBs for public access to prediction-api and gradio-ui).

**Note**: You'll need an Application Load Balancer (ALB) + target groups for public access (especially gradio-ui). Create these in the AWS Console and point them to your ECS services. The internal DNS placeholders in task defs will reference service discovery or ALB internal URLs.

### Step 2: Update Local Files (Deliverables)
The documents already provide most of the content. Here's what to do:

- **Create these new files** exactly as provided in the documents:
  - `.github/workflows/ci.yml`
  - `.github/workflows/cd.yml`
  - `task-definitions/embedding-service.json`
  - `task-definitions/prediction-api.json`
  - `task-definitions/gradio-ui.json`
  - `task-definitions/mlflow-server.json` (for reference only)

- **Update these files**:
  - `docker-compose.yaml` — Activate gradio-ui and ensure image tags can be overridden.
  - `.env.example` — Add the AWS/ECR sections.
  - `Makefile` — Add ECR + ECS + artifacts targets (the provided version already includes them).
  - `.pre-commit-config.yaml` — Add the full config (including no-artifacts and no-env hooks).

**Important Placeholder Replacements** (in task definition JSONs):
- `ACCOUNT_ID` → your 12-digit AWS account ID
- `REGION` → `us-east-1` (or your region)
- `REPLACE_WITH_BUCKET_NAME` → your S3 bucket
- `REPLACE_WITH_EMBEDDING_SERVICE_INTERNAL_DNS` → e.g., the private DNS of the embedding-service (use AWS Cloud Map service discovery or ALB internal endpoint)
- `REPLACE_WITH_PREDICTION_API_INTERNAL_URL` → similar for prediction-api

For Fargate (no bind mounts):
- Your Dockerfiles must include an **entrypoint script** that does `aws s3 sync s3://$ARTIFACTS_S3_BUCKET/artifacts/... /app/artifacts/...` at startup (Decision L1). The task defs already pass `ARTIFACTS_S3_BUCKET` as an env var, and the task role has S3 read permissions.
- Add `awscli` to your Docker images if not already present.

### Step 3: Local Testing Before Pushing
1. Install pre-commit: `make pre-commit-install`
2. Run full validation: `make validate` (should pass all pillars locally)
3. Test Docker locally: `make up-build` then `make health`
4. Test artifact sync: `make artifacts-push` (after setting `.env`)
5. Test ECR push locally: `make ecr-login && make ecr-push` (requires AWS creds in `.env` for local CLI)

### Step 4: Commit, Merge, and Trigger CD
- Commit all changes (pre-commit hooks will run automatically).
- Open a PR → CI should pass (quality + tests).
- Merge to `main` → CD workflow triggers automatically.
- Monitor the CD run in GitHub Actions. It builds/pushes all four images (parallel), then deploys the three services sequentially with rolling updates and a final health check.

### Step 5: Post-Deployment
- Verify services in AWS ECS Console (tasks should be running, logs in CloudWatch).
- Test public endpoints via your ALBs.
- Monitor costs (Fargate is pay-per-use; embedding-service uses more CPU/memory).
- If health checks fail or rolling updates get stuck, check VPC/subnet/security group configuration for awsvpc mode.

### Potential Gotchas & Tips
- **Networking**: Ensure your ECS services are in private subnets with NAT gateway for outbound (S3/ECR pulls) and security groups allowing ALB → tasks on the right ports.
- **S3 Fetch at Startup**: Make the entrypoint robust (retries, timeouts). For ~5MB artifacts it should be fast.
- **Dependencies**: embedding-service deploys first, then prediction-api, then gradio-ui — matches the CD order.
- **mlflow-server**: Pushed to ECR but not deployed (per Decision K2). Good for future Phase 9 with EFS.
- **Security**: OIDC avoids long-lived secrets. Docker Scout blocks only on critical CVEs (good for portfolio).
- **Rollback**: If a deployment fails stability, ECS keeps the previous task definition revision.

Once the infrastructure is ready and files are in place, merging to main should give you a fully automated deployment.
