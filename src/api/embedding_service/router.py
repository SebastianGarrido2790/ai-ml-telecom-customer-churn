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

from typing import TYPE_CHECKING, Any

import numpy as np
from fastapi import APIRouter, HTTPException, Request

from src.api.embedding_service.schemas import EmbedRequest, EmbedResponse, HealthResponse
from src.utils.logger import get_logger

if TYPE_CHECKING:
    pass

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

    try:
        import pandas as pd
        from pandas import DataFrame, Series

        # ColumnTransformer expects a DataFrame with the 'ticket_note' column
        notes_df = pd.DataFrame({"ticket_note": payload.ticket_notes})
        transformed: np.ndarray | DataFrame | Series = preprocessor.transform(notes_df)

        # Ensure numpy array for serialization
        from typing import Any, cast

        if hasattr(transformed, "toarray"):
            transformed_np: np.ndarray = cast(Any, transformed).toarray()
        elif hasattr(transformed, "to_numpy"):
            transformed_np: np.ndarray = cast(Any, transformed).to_numpy()
        elif hasattr(transformed, "values"):
            transformed_np: np.ndarray = cast(Any, transformed).values
        else:
            transformed_np = np.asarray(transformed)

        embeddings: list[list[float]] = transformed_np.tolist()
        dim = transformed_np.shape[1]

    except Exception as exc:
        logger.error(f"Embedding transform failed: {exc!s}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding generation failed: {exc!s}",
        ) from exc

    return EmbedResponse(
        embeddings=embeddings,
        model_version=state.model_version,
        dim=dim,
    )
