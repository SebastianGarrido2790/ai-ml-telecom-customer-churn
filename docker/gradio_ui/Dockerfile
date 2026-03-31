# =============================================================================
# Dockerfile: Gradio UI
# Provides the interactive web dashboard for the Telecom Customer Churn ML system.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

LABEL stage="builder"

RUN pip install --no-cache-dir uv==0.6.14

WORKDIR /app

COPY pyproject.toml ./

# Install UI + SHAP/matplotlib dependencies
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
        gradio \
        httpx \
        pandas \
        numpy \
        shap \
        joblib \
        matplotlib \
        xgboost \
        scikit-learn \
        seaborn \
        python-dotenv \
        python-box \
        pyyaml \
        rich

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

LABEL maintainer="Sebastian" \
      service="gradio-ui" \
      version="1.0.0"

RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup --no-create-home appuser

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

COPY src/ui/                              ./src/ui/
COPY src/config/                          ./src/config/
COPY src/entity/                          ./src/entity/
COPY src/utils/                           ./src/utils/
COPY src/__init__.py                      ./src/__init__.py
COPY src/constants/                       ./src/constants/
COPY config/                              ./config/

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PREDICTION_API_URL="http://prediction-api:8000"

EXPOSE 7860

# Health check logic - Gradio runs on 0.0.0.0:7860 by default
HEALTHCHECK --interval=15s --timeout=5s --retries=3 --start-period=30s \
    CMD python -c "import httpx; httpx.get('http://localhost:7860/').raise_for_status()"

# Pre-create all directories referenced in src/constants/__init__.py
# so that the appuser has write access at startup.
RUN mkdir -p logs reports/figures reports/docs data/raw data/external artifacts mlruns

# Ensure /app and all its children are owned by the non-root user (Rule 6.1)
RUN chown -R appuser:appgroup /app

USER appuser

ENV GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_NAME=0.0.0.0

CMD ["python", "-m", "src.ui.app"]
