"""
Unit tests for the FastAPI inference services.

This module validates the REST API endpoints for both the prediction service
and the embedding microservice, ensuring correct handling of health checks,
churn prediction payloads, and NLP embedding requests.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.embedding_service.router import router as embedding_router
from src.api.prediction_service.router import router as prediction_router
from src.api.prediction_service.schemas import BatchPredictResponse, ChurnPredictionResponse


@pytest.fixture
def app():
    """Provides a mocked FastAPI application for the prediction service."""
    app = FastAPI()
    app.include_router(prediction_router)
    app.state.inference_service = AsyncMock()
    app.state.inference_service.model_version = "1.0.0"
    return app


@pytest.fixture
def embed_app():
    """Provides a mocked FastAPI application for the embedding service."""
    app = FastAPI()
    app.include_router(embedding_router)
    # The router expects these in state
    app.state.nlp_preprocessor = MagicMock()
    app.state.model_version = "v1"
    return app


@pytest.mark.asyncio
async def test_prediction_health(app):
    """Verifies that the prediction service health endpoint returns the correct version."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/v1/health")
    assert response.status_code == 200
    assert response.json()["model_version"] == "1.0.0"


@pytest.mark.asyncio
async def test_prediction_predict(app):
    """Verifies that the churn prediction endpoint correctly processes a sample request."""
    mock_pred = ChurnPredictionResponse(
        customerID="ID-1",
        churn_probability=0.8,
        churn_prediction=True,
        p_structured=0.7,
        p_nlp=0.9,
        nlp_branch_available=True,
        model_version="1.0.0",
    )
    app.state.inference_service.predict_batch.return_value = BatchPredictResponse(
        predictions=[mock_pred], total=1, nlp_branch_available=True
    )

    payload = {
        "customerID": "ID-1",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": "29.85",
        "ticket_note": "Test note",
    }

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/v1/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["churn_probability"] == 0.8


@pytest.mark.asyncio
async def test_embed_endpoint(embed_app):
    """Verifies that the embedding service endpoint returns high-dimensional NLP features."""
    # Mocking the nlp_preprocessor.transform
    import numpy as np

    embed_app.state.nlp_preprocessor.transform.return_value = np.array([[0.1] * 20])  # After PCA

    payload = {"ticket_notes": ["test note"]}
    async with AsyncClient(transport=ASGITransport(app=embed_app), base_url="http://test") as ac:
        response = await ac.post("/v1/embed", json=payload)
    assert response.status_code == 200
    assert response.json()["dim"] == 20
    assert response.json()["model_version"] == "v1"
