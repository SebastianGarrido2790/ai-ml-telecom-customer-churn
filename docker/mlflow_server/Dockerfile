# =============================================================================
# MLflow Tracking Server — Single-Stage Dockerfile
# Serves the MLflow UI and tracking API.
# The ./mlruns directory is bind-mounted at runtime (Decision G1 — file store).
# =============================================================================

FROM python:3.11-slim AS runtime

LABEL maintainer="Sebastian" \
      service="mlflow-server" \
      version="1.0.0"

RUN pip install --no-cache-dir uv==0.6.14 && \
    uv pip install --system mlflow==2.19.0 httpx

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

WORKDIR /app

# mlruns/ is bind-mounted from the host at runtime — preserving full run history
# without inflating the image. No source code or config is needed for the
# MLflow server; it reads only from the backend store path.

ENV PYTHONUNBUFFERED=1

EXPOSE 5000

# Health check polls the MLflow REST API ping endpoint.
HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=20s \
    CMD python -c "import httpx; httpx.get('http://localhost:5000/health').raise_for_status()"

USER appuser

# --backend-store-uri and --default-artifact-root both point to the bind-mounted
# mlruns directory so existing experiments and registered models are preserved.
CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "file:///app/mlruns", \
     "--default-artifact-root", "file:///app/mlruns"]
