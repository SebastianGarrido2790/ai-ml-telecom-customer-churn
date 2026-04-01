## AWS Deployment Decisions

### Decision 1 — AWS Authentication Strategy

**Option I1 — OIDC (OpenID Connect) — Recommended**
GitHub Actions federates with AWS IAM via OIDC. No long-lived credentials stored in GitHub Secrets. The workflow assumes an IAM Role via `aws-actions/configure-aws-credentials@v4` with `role-to-assume`. AWS validates the JWT token issued by GitHub's OIDC provider for each workflow run. Credentials are ephemeral — valid for the duration of the job only.

**Option I2 — IAM User with `AWS_ACCESS_KEY_ID` + `AWS_SECRET_ACCESS_KEY`**
Static credentials stored in GitHub Secrets. Simpler to set up but violates the principle of least privilege — secrets are long-lived and must be rotated manually. Recommended for secure secrets management, OIDC: *"Use GitHub Secrets or OIDC for handling credentials, API keys, and sensitive data."*

**Recommendation and default: Option I1 (OIDC).** The IAM Role ARN is stored in GitHub Secrets as `AWS_ROLE_ARN` — a non-sensitive reference, not a credential.

---

### Decision 2 — ECS Deployment Strategy

**Option J1 — Rolling update (default ECS behavior)**
ECS replaces running task instances one at a time. Zero downtime for stateless services. Simple configuration — no additional infrastructure required.

**Option J2 — Blue/Green deployment via CodeDeploy**
Two complete environments run simultaneously; traffic shifts after health validation. True zero-downtime with instant rollback capability. Requires AWS CodeDeploy, Application Load Balancer, and target group configuration — significant infrastructure overhead for a portfolio project.

**Recommendation: Option J1 (rolling update).** Blue/Green is appropriate for production SLAs above 99.9%. For this project's portfolio context, rolling update provides the correct balance of production-readiness demonstration without introducing unnecessary infrastructure complexity. Phase 9 can migrate to Blue/Green when SLA requirements are defined.

---

### Decision 3 — ECS Service Scoping

Four services in `docker-compose.yaml`. Two options for ECS deployment scope:

**Option K1 — Deploy all four services to ECS**
All four containers run as ECS services. `mlflow-server` requires persistent EFS storage for `mlruns/`. `gradio-ui` is publicly accessible behind an ALB. Full cloud parity with the local Docker Compose stack.

**Option K2 — Deploy the two API services + gradio-ui; keep MLflow on EC2/local**
`mlflow-server` is an internal tool — not customer-facing. Running it in ECS with EFS adds cost and complexity (EFS mount target configuration, VPC subnet routing) for minimal production benefit at this stage. The three customer-facing services (`embedding-service`, `prediction-api`, `gradio-ui`) are the portfolio-demonstrable components.

**Recommendation: Option K2.** MLflow ECS deployment is a Phase 9 concern. The CD pipeline deploys three services. Task definitions for all four are delivered as artifacts (so the fourth can be activated without code changes), but the ECS deploy step targets three services only.

---

### Decision 4 — Artifact Delivery to Fargate

**Option L1 — S3 fetch at container startup**
An entrypoint script runs `aws s3 sync s3://bucket/artifacts/ /app/artifacts/` before starting uvicorn. Simple, no infrastructure changes to the application code, artifacts always match the latest DVC-tracked version in S3.

**Option L2 — Bake artifacts into the image at CD time**
The CD pipeline runs `aws s3 sync` during `docker build`, copying artifacts into the image layer. The image is fully self-contained. Larger images, longer build times, artifacts tightly coupled to image versions.

**Option L3 — EFS (Elastic File System) mount**
EFS provides a persistent shared filesystem mounted into all containers. Mirrors the local bind mount strategy exactly. Requires VPC configuration, EFS mount targets per subnet, and access point configuration.

**Recommendation: Option L1 (S3 fetch at startup).** Cleanest separation of concerns — artifacts are DVC-managed data, not application code. The entrypoint script is a small shell addition to each Dockerfile. EFS (L3) is reserved for Phase 9 if latency of the S3 fetch becomes a concern (it won't for ~5 MB of pkl files).

---

### Decision 5 — Cloud Provider Target

**Option M1 — LocalStack Simulation (Recommended)**
Use LocalStack via Docker Compose to emulate AWS services (S3, ECR, ECS) locally. This avoids cloud costs and credit card requirements while still demonstrating the exact IaC and CI/CD pipelines needed for real AWS. The AWS CLI commands will be appended with `--endpoint-url=http://localhost:4566`.

**Option M2 — Live AWS Account**
Deploy to a real AWS account. Requires a credit card and incurs costs for Fargate compute, ECR storage, and ALB provisioning.

**Recommendation: Option M1 (LocalStack).** See `cloud_bill_hurdle.md` for discussion. This allows us to proceed with Phase 8 locally without losing the AWS architectural patterns.