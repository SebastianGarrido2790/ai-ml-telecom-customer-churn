## Cloud Bill Hurdle

That is a common hurdle when starting with cloud providers like AWS. Since Phase 8 is focused on **CI/CD and Cloud Deployment**, we don't want to stall our progress. We can pivot to a **"Cloud-Agnostic / Local-First"** approach for Phase 8 that still demonstrates 100% of the MLOps engineering requirements without needing a live AWS bill.

Here are two viable paths to proceed:

### Option 1: The "LocalStack" Path (Recommended for AWS Simulation)
**LocalStack** is a high-fidelity cloud service emulator that runs in a single container on your local machine. It allows you to run your Terraform scripts, deploy to a "local" ECR, and even simulate AWS App Runner or Lambda.
*   **Pros:** You get to keep all your AWS-specific code (Terraform, CI/CD logic); it costs $0 and requires no credit card.
*   **Cons:** Requires running one extra Docker container locally; some advanced features might be limited.

### Option 2: The "Registry-Only & CI Hardening" Path
We focus on the **CI (Continuous Integration)** part of the pipeline. We configure GitHub Actions to build your Docker images and push them to **GHCR (GitHub Container Registry)** instead of AWS ECR. We then use a "Pull-based" deployment model onto a local server or a sandbox environment.
*   **Pros:** Very fast to set up; uses standard GitHub tools; no cloud account needed at all.
*   **Cons:** Replaces the "CD to AWS" step with a local deployment simulation; you won't get to see the live AWS Console.

---

### Suggested Decision
Since we have already started drafting the **AWS Actionable Plan**, I recommend **Option 1 (LocalStack)**. It allows us to keep your current architecture intact. We will simply point our Terraform provider and GitHub Actions to `localhost:4566` instead of the public AWS endpoints.

**How we would adapt Phase 8:**
1.  **Local Cloud Infrastructure:** Use Docker Compose to spin up LocalStack.
2.  **Terraform:** Modify the `provider "aws"` block to use the LocalStack endpoint.
3.  **CI/CD:** Update the GitHub Actions workflow to run against the LocalStack container (using a runner that has access to your local environment or a mock runner).
