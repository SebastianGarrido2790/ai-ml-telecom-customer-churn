# AWS Actionable Plan (LocalStack Simulation)

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
