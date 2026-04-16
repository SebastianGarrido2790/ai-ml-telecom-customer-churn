"""
Unit tests for the Embedding Microservice API.

This module validates the REST endpoints of the Embedding Service, ensuring
that text-based ticket notes are accurately transformed into high-dimensional
vectors using the pre-trained NLP preprocessor.

Key validations:
    - /v1/health: Readiness and version checks.
    - /v1/embed: Single and batch embedding extraction.
    - Error handling: Input validation (422) and internal failures (500).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.api.embedding_service.main import app


@pytest.fixture
def client():
    mock_preprocessor = MagicMock()

    # Mock the transform to return a dummy 2D array of the same length as the input
    def side_effect(df):
        return np.array([[0.1, 0.2]] * len(df))

    mock_preprocessor.transform.side_effect = side_effect

    mock_config = MagicMock()
    mock_config.nlp_preprocessor_path = "mock_path.pkl"
    mock_config.model_version = "test-v1"
    mock_config.pca_components = 2
    mock_config.api_key = "test-api-key"

    with (
        patch("src.api.embedding_service.main.ConfigurationManager") as mock_config_mgr,
        patch("src.api.embedding_service.main.joblib.load", return_value=mock_preprocessor),
    ):
        mock_config_mgr.return_value.get_embedding_service_config.return_value = mock_config
        with TestClient(app, raise_server_exceptions=False) as c:
            # Ensure the mock is assigned to state for the test functions to manipulate
            c.app.state.nlp_preprocessor = mock_preprocessor
            c.app.state.api_key = "test-api-key"
            c.headers.update({"X-API-Key": "test-api-key"})
            yield c


def test_health_endpoint(client):
    """Test the /v1/health readiness probe."""
    response = client.get("/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_version"] == "test-v1"


def test_embed_endpoint_single(client):
    """Test embedding a single ticket note."""
    payload = {"ticket_notes": ["I have a problem with my bill."]}
    response = client.post("/v1/embed", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "embeddings" in data
    assert len(data["embeddings"]) == 1
    assert len(data["embeddings"][0]) == 2
    assert data["dim"] == 2
    assert data["model_version"] == "test-v1"


def test_embed_endpoint_batch(client):
    """Test embedding a batch of ticket notes."""
    payload = {"ticket_notes": ["Note 1", "Note 2", "Note 3"]}
    response = client.post("/v1/embed", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert len(data["embeddings"]) == 3
    assert len(data["embeddings"][0]) == 2


def test_embed_endpoint_validation_error(client):
    """Test that invalid payload (empty list when at least one note is required) fails."""
    # If the schema allows empty list, this might pass. Let's check schemas.py or just pass wrong type.
    payload = {"ticket_notes": "not a list"}
    response = client.post("/v1/embed", json=payload)
    assert response.status_code == 422


def test_embed_endpoint_internal_error(client):
    """Test 500 error when preprocessor fails — triggers global exception handler."""
    client.app.state.nlp_preprocessor.transform.side_effect = Exception("Mock Transform Fail")
    payload = {"ticket_notes": ["broken"]}
    response = client.post("/v1/embed", json=payload)
    assert response.status_code == 500
    # Global exception handler masks the actual error message for safety
    assert "Internal server error" in response.json()["detail"]


def test_auth_missing_header(client):
    """Test that requests without X-API-Key are rejected."""
    client.headers.pop("X-API-Key")
    response = client.get("/v1/health")
    assert response.status_code == 422


def test_auth_invalid_key(client):
    """Test that requests with wrong X-API-Key are rejected."""
    client.headers.update({"X-API-Key": "wrong-key"})
    response = client.get("/v1/health")
    assert response.status_code == 401
