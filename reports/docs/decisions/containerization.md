## Containerization Decisions

### Decision 1 — SentenceTransformer Model Caching Strategy

The `all-MiniLM-L6-v2` model (~90MB) must be available inside the embedding service container. Three options:

**Option E1 — Bake into image at build time**
A `RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"` layer in the Dockerfile downloads and caches the model during `docker build`. The image is ~90MB larger but is fully self-contained and offline-capable. Every `docker compose up` starts instantly.

**Option E2 — Download on first container startup**
No model download at build time. The warmup in `main.py` downloads the model on first startup if not cached. Requires internet access on first run. The volume mount approach in E3 is more practical than this option for local development.

**Option E3 — Mount host HuggingFace cache as a volume**
```yaml
volumes:
  - ${HF_HOME:-~/.cache/huggingface}:/root/.cache/huggingface
```
Reuses whatever the host has already downloaded. Zero network cost if the model is cached locally. Fails silently if the host has never run the model (falls back to download at startup). The least reproducible option across machines.

**Recommendation: Option E1.** For a portfolio project and for Phase 8 AWS deployment, the image must be self-contained. A 90MB size increase is acceptable. E3's host-path dependency breaks reproducibility across environments. E2 requires outbound internet access in the container at startup, which is unacceptable in production network configurations.

---

### Decision 2 — Artifact Mount Strategy

Model artifacts (`*.pkl` files) are DVC-tracked and not in the Git repository. Two options:

**Option F1 — Bind mount from host at runtime**
```yaml
volumes:
  - ./artifacts:/app/artifacts:ro
```
The container reads artifacts directly from the host filesystem. Simple, no image bloat, artifacts stay in sync with DVC. The `:ro` (read-only) flag prevents container writes to the artifact directory.

**Option F2 — COPY into image at build time**
Artifacts are copied into the image during `docker build`. Fully self-contained but produces large images (~500MB+ per service) and requires a rebuild whenever artifacts change (i.e., after every training run).

**Recommendation: Option F1.** Artifacts are the output of the DVC pipeline, not application code. Treating them as runtime data (bind-mounted) is architecturally correct and aligns with how ECS Fargate will consume them from S3 in Phase 8 (EFS or S3-mounted volumes). This keeps images lean and decoupled from the training cycle.

---

### Decision 3 — MLflow Server Storage

The `mlflow-server` service needs a backend store. Two options:

**Option G1 — File store (current: `file:./mlruns`)**
Mount `./mlruns` as a volume. Simple, zero dependencies, consistent with the current `params.yaml` URI.

**Option G2 — SQLite backend**
`sqlite:///mlflow.db` in a named volume. Better performance for concurrent reads, supports model registry queries. Requires a one-time migration from the file store.

**Recommendation: Option G1** for Phase 7. The file store is already operational and the `mlruns/` directory contains existing run history. Phase 8 can migrate to RDS PostgreSQL when deploying to AWS.
