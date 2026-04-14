# =============================================================================
# Embedding Microservice — Multi-Stage Dockerfile
# Stage 1 (builder): installs dependencies + bakes SentenceTransformer model
# Stage 2 (runtime): lean production image with non-root user
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder
# Installs all Python dependencies and downloads the SentenceTransformer model
# into the image cache at build time (Decision E1 — offline-capable runtime).
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL stage="builder"

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv==0.6.14

WORKDIR /app

# Copy dependency manifest first — maximises layer cache reuse.
# Re-installing only when pyproject.toml changes.
COPY pyproject.toml ./

# Install all project dependencies into an isolated venv
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
        fastapi \
        uvicorn[standard] \
        pydantic \
        joblib \
        scikit-learn \
        sentence-transformers \
        numpy \
        pandas \
        python-dotenv \
        pyyaml \
        python-box \
        rich \
        httpx

# Bake the SentenceTransformer model into the image cache.
# This eliminates the ~13s cold-start download on every container start
# and makes the image fully self-contained for offline / air-gapped environments.
# Model size: ~90MB. HF_HOME controls the cache directory.
ENV HF_HOME=/app/.cache/huggingface
RUN .venv/bin/python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('Model cached successfully.')"

# ---------------------------------------------------------------------------
# Stage 2: runtime
# Copies only the venv, HF model cache, application source, and config.
# No build tools, no pip, no uv in the final image.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Sebastian" \
      service="embedding-service" \
      version="1.0.0"

# Create non-root user — least-privilege execution
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

WORKDIR /app

# Copy the venv from builder (no build tools carried over)
COPY --from=builder /app/.venv /app/.venv

# Copy the baked HuggingFace model cache from builder
COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface

# Copy application source and configuration
# artifacts/ is intentionally excluded — bind-mounted at runtime
COPY src/api/embedding_service/ ./src/api/embedding_service/
COPY src/api/__init__.py         ./src/api/__init__.py
COPY src/config/                 ./src/config/
COPY src/entity/                 ./src/entity/
COPY src/utils/                  ./src/utils/
COPY src/__init__.py             ./src/__init__.py
COPY src/constants/              ./src/constants/
COPY config/                     ./config/

# Make the venv the active Python environment
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface

EXPOSE 8001

# Health check — curl not available in slim; use python httpx instead.
# /v1/health returns 200 only after the warmup completes (after yield in lifespan),
# so this probe correctly gates the depends_on condition in docker-compose.yaml.
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=60s \
    CMD python -c "import httpx; httpx.get('http://localhost:8001/v1/health').raise_for_status()"

# Pre-create all directories referenced in src/constants/__init__.py
# (Decision R1) so that the appuser has write access at startup when the
# package is imported and mkdir(exist_ok=True) is called.
RUN mkdir -p logs reports/figures reports/docs data/raw data/external artifacts

# Install awscli for Phase 8 S3 artifacts fetch at startup
RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Ensure /app and all its children are owned by the non-root user
RUN chown -R appuser:appgroup /app

USER appuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "src.api.embedding_service.main:app", \
     "--host", "0.0.0.0", "--port", "8001", \
     "--workers", "1", "--log-level", "info"]
