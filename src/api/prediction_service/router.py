"""
Route handlers for the Prediction API microservice.

Endpoints:
    POST /v1/predict         — Real-time single-customer churn scoring.
    POST /v1/predict/batch   — Bulk customer scoring.
    GET  /v1/health          — Container readiness probe.

This module is a pure HTTP conductor. All inference logic — DataFrame
reconstruction, preprocessing, embedding calls, and model prediction —
is delegated to InferenceService (Decision D2). Each endpoint handler
is intentionally concise: validate input, call service, return result.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from src.api.prediction_service.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ChurnPredictionResponse,
    CustomerFeatureRequest,
    PredictionHealthResponse,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/v1")


@router.get(
    "/health",
    response_model=PredictionHealthResponse,
    summary="Readiness probe",
)
async def health(request: Request) -> PredictionHealthResponse:
    """Returns service health status and the loaded model version.

    Args:
        request: FastAPI request object (provides access to app state).

    Returns:
        PredictionHealthResponse with status='healthy' and model_version.
    """
    state: Any = request.app.state
    return PredictionHealthResponse(model_version=state.inference_service.model_version)


@router.post(
    "/predict",
    response_model=ChurnPredictionResponse,
    summary="Real-time single-customer churn prediction",
)
async def predict(
    payload: CustomerFeatureRequest,
    request: Request,
) -> ChurnPredictionResponse:
    """Scores a single customer for churn risk using the Late Fusion model.

    Delegates entirely to InferenceService.predict_batch (called with a
    one-item list). The circuit breaker inside InferenceService ensures this
    endpoint never returns 5xx due to Embedding Microservice unavailability —
    it sets nlp_branch_available=False and continues with structured prediction.

    Args:
        payload: CustomerFeatureRequest with all 19 structured fields + ticket_note.
        request: FastAPI request object (provides access to app state).

    Returns:
        ChurnPredictionResponse with churn probability, prediction, branch
        probabilities, and nlp_branch_available flag.
    """
    state: Any = request.app.state
    batch_response = await state.inference_service.predict_batch([payload])
    return batch_response.predictions[0]


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    summary="Bulk customer churn scoring",
)
async def predict_batch(
    payload: BatchPredictRequest,
    request: Request,
) -> BatchPredictResponse:
    """Scores a batch of customers for churn risk using the Late Fusion model.

    Embedding calls are batched into a single POST /v1/embed request to the
    Embedding Microservice, reducing inter-service round-trips. The circuit
    breaker applies to the entire batch: if the embedding service is unreachable,
    all customers in the batch receive nlp_branch_available=False.

    Args:
        payload: BatchPredictRequest containing one or more CustomerFeatureRequest.
        request: FastAPI request object (provides access to app state).

    Returns:
        BatchPredictResponse with predictions list, total count, and
        nlp_branch_available flag for the batch.
    """
    state: Any = request.app.state
    return await state.inference_service.predict_batch(payload.customers)
