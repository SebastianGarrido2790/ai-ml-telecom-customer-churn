"""
Root conftest.py for centralized pytest fixtures.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_sentence_transformer():
    """Mock for SentenceTransformer to avoid downloading models during tests."""
    with patch("sentence_transformers.SentenceTransformer") as mock:
        # Create a mock instance
        mock_instance = MagicMock()
        # Mock the get_sentence_embedding_dimension
        mock_instance.get_sentence_embedding_dimension.return_value = 384

        # Mock encoding to return dummy vectors of shape (len(texts), 384)
        def mock_encode(texts, **kwargs):
            return np.random.rand(len(texts), 384)

        mock_instance.encode.side_effect = mock_encode

        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def sample_telco_df():
    """Provides a small, valid Telco Customer Churn DataFrame for testing."""
    return pd.DataFrame(
        {
            "customerID": [f"ID_{i}" for i in range(20)],
            "gender": ["Male", "Female"] * 10,
            "SeniorCitizen": [0, 1] * 10,
            "Partner": ["Yes", "No"] * 10,
            "Dependents": ["Yes", "No"] * 10,
            "tenure": [12, 1, 0, 72, 50] * 4,
            "PhoneService": ["Yes", "No"] * 10,
            "MultipleLines": ["Yes", "No"] * 10,
            "InternetService": ["DSL", "Fiber optic"] * 10,
            "OnlineSecurity": ["Yes", "No"] * 10,
            "OnlineBackup": ["Yes", "No"] * 10,
            "DeviceProtection": ["Yes", "No"] * 10,
            "TechSupport": ["Yes", "No"] * 10,
            "StreamingTV": ["Yes", "No"] * 10,
            "StreamingMovies": ["Yes", "No"] * 10,
            "Contract": ["Month-to-month", "One year"] * 10,
            "PaperlessBilling": ["Yes", "No"] * 10,
            "PaymentMethod": ["Electronic check", "Mailed check"] * 10,
            "MonthlyCharges": [50.5] * 20,
            "TotalCharges": [str(i * 10.5) for i in range(20)],
            "ticket_note": ["dummy text"] * 20,
            "primary_sentiment_tag": ["Positive", "Negative", "Neutral", "Positive"] * 5,
            "Churn": ["Yes", "No", "Yes", "No", "No"] * 4,
        }
    )
