"""
Route handlers for the Embedding Microservice.

Endpoints:
    POST /v1/embed   — Convert ticket notes to PCA-reduced embeddings.
    GET  /v1/health  — Container readiness probe.

This module is a pure HTTP conductor. All transformation logic is delegated
to the fitted nlp_preprocessor pipeline loaded at startup (Brain vs.
Brawn separation). The router never touches sklearn or numpy directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Request
from pandas import DataFrame, Series

from src.api.embedding_service.schemas import EmbedRequest, EmbedResponse, HealthResponse
from src.utils.array_utils import ensure_ndarray
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1")


@router.get("/health", response_model=HealthResponse, summary="Readiness probe")
async def health(request: Request) -> HealthResponse:
    """Returns service health status and the loaded model version.

    Used by Docker Compose health checks and container orchestrators to
    determine when the service is ready to receive traffic.

    Args:
        request: FastAPI request object (provides access to app state).

    Returns:
        HealthResponse with status='healthy' and the loaded model_version.
    """
    state: Any = request.app.state
    return HealthResponse(model_version=state.model_version)


@router.post("/embed", response_model=EmbedResponse, summary="Generate PCA-reduced embeddings")
async def embed(payload: EmbedRequest, request: Request) -> EmbedResponse:
    """Converts a batch of ticket notes into PCA-reduced embedding vectors.

    Applies the fitted nlp_preprocessor pipeline (TextEmbedder → PCA) to
    the input notes. The preprocessor was fitted exclusively on the training
    set during Phase 4, ensuring inference-time Anti-Skew compliance.

    Args:
        payload: EmbedRequest containing one or more ticket note strings.
        request: FastAPI request object (provides access to app state).

    Returns:
        EmbedResponse with embeddings, model_version, and dim.

    Raises:
        HTTPException 422: Raised automatically by FastAPI if payload
                           fails Pydantic validation.
        HTTPException 500: Raised if the preprocessor transform fails
                           for an unexpected reason.
    """
    state: Any = request.app.state
    preprocessor = state.nlp_preprocessor

    # ColumnTransformer expects a DataFrame with the 'ticket_note' column
    notes_df = pd.DataFrame({"ticket_note": payload.ticket_notes})
    transformed: np.ndarray | DataFrame | Series = preprocessor.transform(notes_df)

    # Ensure numpy array for serialization
    transformed_np = ensure_ndarray(transformed)

    embeddings: list[list[float]] = transformed_np.tolist()
    dim = transformed_np.shape[1]

    return EmbedResponse(
        embeddings=embeddings,
        model_version=state.model_version,
        dim=dim,
    )
