"""
Unit tests for the Prediction Service API.

This module validates the end-to-end inference logic for churn prediction,
including the orchestration of structural and NLP branches, fallback behaviors
when microservices are unavailable, and schema-based request validation.

Key validations:
    - /v1/predict: Single customer churn risk evaluation.
    - /v1/predict/batch: High-throughput batch processing.
    - Orchestration: Graceful recovery when the embedding service times out.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.prediction_service.main import app


@pytest.fixture
def mock_artifacts():
    """Provides a dictionary of mocked model artifacts."""
    mock_struct_pre = MagicMock()
    mock_struct_pre.transform.return_value = np.zeros((1, 10))

    mock_struct_model = MagicMock()
    mock_struct_model.predict_proba.return_value = np.array([[0.8, 0.2]])

    mock_nlp_model = MagicMock()
    mock_nlp_model.predict_proba.return_value = np.array([[0.7, 0.3]])

    mock_meta_model = MagicMock()
    mock_meta_model.predict_proba.return_value = np.array([[0.6, 0.4]])

    return {
        "structured_preprocessor": mock_struct_pre,
        "structured_model": mock_struct_model,
        "nlp_model": mock_nlp_model,
        "meta_model": mock_meta_model,
        "api_key": "test-api-key",
    }


@pytest.fixture
def client(mock_artifacts):
    """Provides a TestClient with mocked dependencies."""
    mock_config = MagicMock()
    mock_config.structured_preprocessor_path = "mock_path_sp.pkl"
    mock_config.structured_model_path = "mock_path_sm.pkl"
    mock_config.nlp_model_path = "mock_path_nm.pkl"
    mock_config.meta_model_path = "mock_path_mm.pkl"
    mock_config.embedding_service_url = "http://mock-embed:8001"
    mock_config.model_version = "test-v1"
    mock_config.pca_components = 20
    mock_config.api_key = "test-api-key"
    mock_config.timeout_seconds = 1.0

    def mock_joblib_load(path):
        if "sp" in path:
            return mock_artifacts["structured_preprocessor"]
        if "sm" in path:
            return mock_artifacts["structured_model"]
        if "nm" in path:
            return mock_artifacts["nlp_model"]
        if "mm" in path:
            return mock_artifacts["meta_model"]
        return MagicMock()

    with (
        patch("src.api.prediction_service.main.ConfigurationManager") as mock_config_mgr,
        patch("src.api.prediction_service.main.joblib.load", side_effect=mock_joblib_load),
    ):
        mock_v = mock_config_mgr.return_value
        mock_v.get_prediction_api_config.return_value = mock_config
        mock_v.get_embedding_service_config.return_value = mock_config
        with TestClient(app, raise_server_exceptions=False) as c:
            c.headers.update({"X-API-Key": "test-api-key"})  # Default valid header
            yield c


def test_health_endpoint(client):
    """Test the /v1/health readiness probe."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "test-v1"


@pytest.fixture
def sample_payload():
    """Provides a sample prediction payload."""
    return {
        "customers": [
            {
                "customerID": "1234-ABCD",
                "tenure": 12,
                "MonthlyCharges": 70.0,
                "TotalCharges": "840.0",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
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
                "ticket_note": "I want to cancel my subscription.",
            }
        ]
    }


def test_predict_endpoint_success(client, sample_payload):
    """Test successful prediction with mock embedding response."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "embeddings": [[0.1] * 20],
        "model_version": "embed-v1",
        "dim": 20,
    }
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        # sample_payload is a BatchPredictRequest (has "customers" key)
        response = client.post("/v1/predict/batch", json=sample_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert "predictions" in data
        assert data["predictions"][0]["customerID"] == "1234-ABCD"
        assert data["predictions"][0]["nlp_branch_available"] is True


def test_predict_endpoint_single_success(client, sample_payload):
    """Test successful single-customer prediction."""
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "embeddings": [[0.1] * 20],
        "model_version": "embed-v1",
        "dim": 20,
    }
    mock_resp.raise_for_status.return_value = None

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.return_value = mock_resp
        # Single customer payload
        single_payload = sample_payload["customers"][0]
        response = client.post("/v1/predict", json=single_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["customerID"] == "1234-ABCD"
        assert data["nlp_branch_available"] is True


def test_predict_endpoint_fallback(client, sample_payload):
    """Test prediction fallback when embedding service fails."""
    import httpx

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
        mock_post.side_effect = httpx.ConnectError("Mock Connection Refused")
        # sample_payload is BatchPredictRequest
        response = client.post("/v1/predict/batch", json=sample_payload)

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["predictions"][0]["nlp_branch_available"] is False
        assert data["nlp_branch_available"] is False


def test_predict_endpoint_validation_error(client):
    """Test that missing required fields fail."""
    payload = {"customers": [{"customerID": "MISSING_FIELDS"}]}
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422


def test_auth_missing_header(client):
    """Test that requests without X-API-Key are rejected."""
    # Temporarily remove default header
    client.headers.pop("X-API-Key")
    response = client.get("/v1/health")
    assert response.status_code == 422
    assert "x-api-key" in response.text.lower()


def test_auth_invalid_key(client):
    """Test that requests with wrong X-API-Key are rejected."""
    client.headers.update({"X-API-Key": "wrong-key"})
    response = client.get("/v1/health")
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API Key"


def test_batch_limit_exceeded(client, sample_payload):
    """Test that batches exceeding 1000 items are rejected (Schema Hardening)."""
    # Duplicate the sample customer 1001 times
    customer = sample_payload["customers"][0]
    large_payload = {"customers": [customer] * 1001}
    response = client.post("/v1/predict/batch", json=large_payload)
    assert response.status_code == 422
    # Pydantic v2 error message for max_length
    assert "1000" in response.text
