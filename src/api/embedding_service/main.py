"""
Embedding Microservice — FastAPI application factory.

Serves PCA-reduced ticket note embeddings via a stateless REST API.
The nlp_preprocessor.pkl artifact (TextEmbedder + PCA, fitted on the Phase 4
training set) is loaded once at startup via the lifespan context manager and
stored on app.state for zero-overhead access per request.

Run locally:
    uv run uvicorn src.api.embedding_service.main:app --host 0.0.0.0 --port 8001 --reload

Run via Docker (Phase 7):
    docker compose up embedding-service
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import joblib
from fastapi import FastAPI

from src.api.embedding_service.router import router
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="Embedding Microservice")


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle.

    On startup:
        - Loads ConfigurationManager to resolve artifact paths.
        - Deserializes nlp_preprocessor.pkl from the feature engineering artifact
          directory.
        - Warms the SentenceTransformer model by running one dummy transform()
          call. This eliminates cold-start latency on the first real inference
          request, keeping the prediction API timeout at 5s. Without warmup,
          the first transform() triggers PyTorch model loading (~2s), which
          risks breaching the inter-service timeout threshold.
        - Stores the preprocessor and model version string on app.state.

    On shutdown:
        - Clears app.state to release memory.

    Args:
        application: The FastAPI application instance.

    Yields:
        None — control returns to FastAPI to handle requests.
    """
    logger.info("Embedding Microservice starting up.")

    config_mgr = ConfigurationManager()
    embed_cfg = config_mgr.get_embedding_service_config()

    logger.info(f"Loading NLP preprocessor from: {embed_cfg.nlp_preprocessor_path}")
    nlp_preprocessor = joblib.load(embed_cfg.nlp_preprocessor_path)

    application.state.nlp_preprocessor = nlp_preprocessor
    application.state.model_version = embed_cfg.model_version
    application.state.pca_components = embed_cfg.pca_components

    # Warm the SentenceTransformer to eliminate cold-start latency on the first
    # real request. The TextEmbedder uses lazy loading — the PyTorch model is
    # not initialised until the first transform() call. Running a dummy call
    # here triggers that initialisation once, at startup, before any client
    # traffic arrives. Without this, the first /v1/embed call takes ~2s extra,
    # risking a timeout breach on the prediction API's httpx client.
    logger.info("Warming NLP preprocessor (first transform initialises SentenceTransformer).")
    import pandas as pd

    nlp_preprocessor.transform(pd.DataFrame({"ticket_note": ["warmup"]}))
    logger.info("NLP preprocessor warmed up — SentenceTransformer loaded into memory.")

    logger.info(
        f"Embedding Microservice ready. "
        f"Model: {embed_cfg.model_version} | Dim: {embed_cfg.pca_components}"
    )

    yield

    logger.info("Embedding Microservice shutting down.")
    del application.state.nlp_preprocessor


def create_app() -> FastAPI:
    """Creates and configures the Embedding Microservice FastAPI application.

    Returns:
        FastAPI application instance with lifespan, metadata, and router mounted.
    """
    application = FastAPI(
        title="Telecom Churn — Embedding Microservice",
        description=(
            "Converts raw customer ticket notes into PCA-reduced embedding vectors "
            "using the Phase 4 NLP preprocessor. Part of the Late Fusion inference stack."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )
    application.include_router(router)
    return application


app = create_app()
