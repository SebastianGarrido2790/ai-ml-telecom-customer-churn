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

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import joblib
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

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

    application.state.api_key = pred_cfg.api_key
    embed_cfg = config_mgr.get_embedding_service_config()

    application.state.inference_service = InferenceService(
        structured_preprocessor=structured_preprocessor,
        structured_model=structured_model,
        nlp_model=nlp_model,
        meta_model=meta_model,
        embedding_service_url=pred_cfg.embedding_service_url,
        model_version=pred_cfg.model_version,
        pca_components=pred_cfg.pca_components,
        api_key=pred_cfg.api_key,
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

    # 1. CORS Middleware
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Restrict this in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 2. Global Exception Handler
    @application.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Global exception caught: {type(exc).__name__}: {exc!s}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error. Please check logs for correlation-id."},
        )

    # 3. API Key Validation Dependency
    async def validate_api_key(request: Request, x_api_key: str = Header(...)):
        if x_api_key != request.app.state.api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API Key",
            )

    application.include_router(router, dependencies=[Depends(validate_api_key)])
    return application


app = create_app()
