# =============================================================================
# Prediction API — Multi-Stage Dockerfile
# Stage 1 (builder): installs all ML + API dependencies
# Stage 2 (runtime): lean production image with non-root user
# Model artifacts are NOT copied into the image — bind-mounted at runtime
# (Decision F1) so the image stays lean and decoupled from the training cycle.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL stage="builder"

RUN pip install --no-cache-dir uv==0.6.14

WORKDIR /app

COPY pyproject.toml ./

# Install all prediction API dependencies
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
        fastapi \
        uvicorn[standard] \
        pydantic \
        joblib \
        scikit-learn \
        xgboost \
        numpy \
        pandas \
        httpx \
        python-dotenv \
        pyyaml \
        python-box \
        rich

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Sebastian" \
      service="prediction-api" \
      version="1.0.0"

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

# Application source and configuration only.
# artifacts/ is bind-mounted at runtime via docker-compose.yaml volumes.
COPY src/api/prediction_service/ ./src/api/prediction_service/
COPY src/api/embedding_service/schemas.py ./src/api/embedding_service/schemas.py
COPY src/api/__init__.py                  ./src/api/__init__.py
COPY src/config/                          ./src/config/
COPY src/entity/                          ./src/entity/
COPY src/utils/                           ./src/utils/
COPY src/__init__.py                      ./src/__init__.py
COPY src/constants/                       ./src/constants/
COPY config/                              ./config/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# EMBEDDING_SERVICE_HOST is overridden by docker-compose.yaml to the
# container DNS name "embedding-service". Default is local development value.
ENV EMBEDDING_SERVICE_HOST=127.0.0.1

EXPOSE 8000

# Health check mirrors the embedding service pattern.
# Returns 200 only after all four artifacts are loaded (after yield in lifespan).
HEALTHCHECK --interval=10s --timeout=5s --retries=3 --start-period=30s \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/v1/health').raise_for_status()"

# Pre-create all directories referenced in src/constants/__init__.py
# so that the appuser has write access at startup.
RUN mkdir -p logs reports/figures reports/docs data/raw data/external artifacts

# Install awscli for Phase 8 S3 artifacts fetch at startup
RUN apt-get update && apt-get install -y awscli && rm -rf /var/lib/apt/lists/*
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Ensure /app and all its children are owned by the non-root user
RUN chown -R appuser:appgroup /app

USER appuser

ENTRYPOINT ["/entrypoint.sh"]
CMD ["uvicorn", "src.api.prediction_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--log-level", "info"]
