"""
Prediction API — FastAPI application factory.

Serves Late Fusion churn predictions via a stateless REST API. All four model
artifacts (structured_preprocessor.pkl, structured_model.pkl, nlp_model.pkl,
meta_model.pkl) are loaded once at startup and held in an InferenceService
instance on app.state.

The Embedding Microservice URL is configured at startup from config.yaml.
In local development the URL points to localhost:8001; inside Docker Compose
it resolves to the embedding-service container (Phase 7).

Run locally (requires embedding-service running on port 8001):
    uv run uvicorn src.api.prediction_service.main:app --host 0.0.0.0 --port 8000 --reload

Run via Docker (Phase 7):
    docker compose up prediction-api
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import joblib
from fastapi import FastAPI

from src.api.prediction_service.inference import InferenceService
from src.api.prediction_service.router import router
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="Prediction API")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle.

    On startup:
        - Resolves all artifact paths via ConfigurationManager.
        - Deserializes all four model artifacts with joblib.
        - Instantiates InferenceService with loaded artifacts and embedding URL.
        - Stores InferenceService on app.state for request-scoped access.

    On shutdown:
        - Clears app.state to release memory.

    Args:
        application: The FastAPI application instance.

    Yields:
        None — control returns to FastAPI to handle requests.
    """
    logger.info("Prediction API starting up.")

    config_mgr = ConfigurationManager()
    pred_cfg = config_mgr.get_prediction_api_config()

    logger.info("Loading model artifacts.")
    structured_preprocessor = joblib.load(pred_cfg.structured_preprocessor_path)
    structured_model = joblib.load(pred_cfg.structured_model_path)
    nlp_model = joblib.load(pred_cfg.nlp_model_path)
    meta_model = joblib.load(pred_cfg.meta_model_path)

    logger.info(f"Artifacts loaded. Embedding service: {pred_cfg.embedding_service_url}")

    embed_cfg = config_mgr.get_embedding_service_config()

    application.state.inference_service = InferenceService(
        structured_preprocessor=structured_preprocessor,
        structured_model=structured_model,
        nlp_model=nlp_model,
        meta_model=meta_model,
        embedding_service_url=pred_cfg.embedding_service_url,
        model_version=pred_cfg.model_version,
        pca_components=pred_cfg.pca_components,
        timeout_seconds=embed_cfg.timeout_seconds,
    )

    logger.info(f"Prediction API ready. Model: {pred_cfg.model_version} | Port: {pred_cfg.port}")

    yield

    logger.info("Prediction API shutting down.")
    del application.state.inference_service


def create_app() -> FastAPI:
    """Creates and configures the Prediction API FastAPI application.

    Returns:
        FastAPI application instance with lifespan, metadata, and router mounted.
    """
    application = FastAPI(
        title="Telecom Churn — Prediction API",
        description=(
            "Late Fusion churn risk scoring API. Combines structured tabular features "
            "(Branch 1) with PCA-reduced NLP ticket note embeddings (Branch 2) via a "
            "Logistic Regression meta-learner. Implements circuit-breaker fallback "
            "when the Embedding Microservice is unavailable."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
