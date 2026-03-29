"""
Unit Tests: Phase 6 API Schemas and InferenceService Logic.

Tests cover five deterministic guarantees of the inference pipeline:

    1. EmbedRequest/EmbedResponse schema contracts.
    2. CustomerFeatureRequest field validation and constraints.
    3. ChurnPredictionResponse field constraints and circuit-breaker flag.
    4. BatchPredictRequest/BatchPredictResponse structure.
    5. InferenceService._build_structured_df column order and TotalCharges handling.
    6. InferenceService circuit breaker: zero-vector fallback on embedding failure.

No live HTTP calls, no model artifacts, and no sklearn/xgboost imports are
required — all external dependencies are replaced with mock objects or
minimal fixtures.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Shared fixture: minimal valid CustomerFeatureRequest payload
# ---------------------------------------------------------------------------

VALID_CUSTOMER_PAYLOAD: dict[str, Any] = {
    "customerID": "1234-ABCD",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.50,
    "TotalCharges": "1026.0",
    "ticket_note": "Customer called to inquire about billing charges.",
}


# ---------------------------------------------------------------------------
# Test 1: EmbedRequest / EmbedResponse schema contracts
# ---------------------------------------------------------------------------


class TestEmbeddingSchemas:
    """Validates the I/O contracts of the Embedding Microservice."""

    def test_embed_request_valid(self) -> None:
        """A list with at least one note must be accepted."""
        from src.api.embedding_service.schemas import EmbedRequest

        req = EmbedRequest(ticket_notes=["Customer called about billing."])
        assert len(req.ticket_notes) == 1

    def test_embed_request_empty_list_fails(self) -> None:
        """An empty ticket_notes list must fail validation (min_length=1)."""
        from src.api.embedding_service.schemas import EmbedRequest

        with pytest.raises(ValidationError):
            EmbedRequest(ticket_notes=[])

    def test_embed_request_batch(self) -> None:
        """Multiple notes in a single request must be accepted."""
        from src.api.embedding_service.schemas import EmbedRequest

        notes = ["Note A.", "Note B.", "Note C."]
        req = EmbedRequest(ticket_notes=notes)
        assert len(req.ticket_notes) == 3

    def test_embed_response_valid(self) -> None:
        """A valid embeddings matrix with matching dim must be accepted."""
        from src.api.embedding_service.schemas import EmbedResponse

        resp = EmbedResponse(
            embeddings=[[0.1] * 20],
            model_version="all-MiniLM-L6-v2-pca20",
            dim=20,
        )
        assert resp.dim == 20
        assert len(resp.embeddings[0]) == 20

    def test_embed_response_invalid_dim(self) -> None:
        """dim=0 must fail schema validation (gt=0 constraint)."""
        from src.api.embedding_service.schemas import EmbedResponse

        with pytest.raises(ValidationError):
            EmbedResponse(
                embeddings=[],
                model_version="all-MiniLM-L6-v2-pca20",
                dim=0,
            )


# ---------------------------------------------------------------------------
# Test 2: CustomerFeatureRequest validation
# ---------------------------------------------------------------------------


class TestCustomerFeatureRequest:
    """Validates the Prediction API input schema."""

    def test_valid_payload_accepted(self) -> None:
        """A fully valid payload must be accepted without errors."""
        from src.api.prediction_service.schemas import CustomerFeatureRequest

        req = CustomerFeatureRequest(**VALID_CUSTOMER_PAYLOAD)
        assert req.customerID == "1234-ABCD"
        assert req.tenure == 12
        assert req.MonthlyCharges == pytest.approx(85.50)

    def test_customer_id_is_optional(self) -> None:
        """customerID must default to None when omitted."""
        from src.api.prediction_service.schemas import CustomerFeatureRequest

        payload = {k: v for k, v in VALID_CUSTOMER_PAYLOAD.items() if k != "customerID"}
        req = CustomerFeatureRequest(**payload)
        assert req.customerID is None

    def test_negative_tenure_fails(self) -> None:
        """Negative tenure must raise ValidationError."""
        from src.api.prediction_service.schemas import CustomerFeatureRequest

        with pytest.raises(ValidationError):
            CustomerFeatureRequest(**{**VALID_CUSTOMER_PAYLOAD, "tenure": -1})

    def test_invalid_senior_citizen_fails(self) -> None:
        """SeniorCitizen > 1 must raise ValidationError."""
        from src.api.prediction_service.schemas import CustomerFeatureRequest

        with pytest.raises(ValidationError):
            CustomerFeatureRequest(**{**VALID_CUSTOMER_PAYLOAD, "SeniorCitizen": 5})

    def test_negative_monthly_charges_fails(self) -> None:
        """Negative MonthlyCharges must raise ValidationError."""
        from src.api.prediction_service.schemas import CustomerFeatureRequest

        with pytest.raises(ValidationError):
            CustomerFeatureRequest(**{**VALID_CUSTOMER_PAYLOAD, "MonthlyCharges": -10.0})

    def test_total_charges_none_accepted(self) -> None:
        """TotalCharges=None must be accepted (tenure=0 customers)."""
        from src.api.prediction_service.schemas import CustomerFeatureRequest

        req = CustomerFeatureRequest(**{**VALID_CUSTOMER_PAYLOAD, "TotalCharges": None})
        assert req.TotalCharges is None


# ---------------------------------------------------------------------------
# Test 3: ChurnPredictionResponse constraints
# ---------------------------------------------------------------------------


class TestChurnPredictionResponse:
    """Validates the Prediction API output schema."""

    def test_valid_response_accepted(self) -> None:
        """A valid prediction response must be accepted."""
        from src.api.prediction_service.schemas import ChurnPredictionResponse

        resp = ChurnPredictionResponse(
            customerID="1234-ABCD",
            churn_probability=0.72,
            churn_prediction=True,
            p_structured=0.65,
            p_nlp=0.58,
            nlp_branch_available=True,
            model_version="late-fusion-v2",
        )
        assert resp.churn_prediction is True
        assert resp.nlp_branch_available is True

    def test_circuit_breaker_flag_false(self) -> None:
        """nlp_branch_available=False must be accepted (circuit breaker state)."""
        from src.api.prediction_service.schemas import ChurnPredictionResponse

        resp = ChurnPredictionResponse(
            customerID=None,
            churn_probability=0.55,
            churn_prediction=True,
            p_structured=0.55,
            p_nlp=0.0,
            nlp_branch_available=False,
            model_version="late-fusion-v2",
        )
        assert resp.nlp_branch_available is False
        assert resp.p_nlp == pytest.approx(0.0)

    def test_probability_out_of_range_fails(self) -> None:
        """churn_probability > 1.0 must raise ValidationError."""
        from src.api.prediction_service.schemas import ChurnPredictionResponse

        with pytest.raises(ValidationError):
            ChurnPredictionResponse(
                customerID=None,
                churn_probability=1.5,
                churn_prediction=True,
                p_structured=0.5,
                p_nlp=0.5,
                nlp_branch_available=True,
                model_version="late-fusion-v2",
            )


# ---------------------------------------------------------------------------
# Test 4: BatchPredictRequest / BatchPredictResponse
# ---------------------------------------------------------------------------


class TestBatchSchemas:
    """Validates the batch prediction request and response schemas."""

    def test_batch_request_valid(self) -> None:
        """A batch request with one customer must be accepted."""
        from src.api.prediction_service.schemas import (
            BatchPredictRequest,
            CustomerFeatureRequest,
        )

        req = BatchPredictRequest(customers=[CustomerFeatureRequest(**VALID_CUSTOMER_PAYLOAD)])
        assert req.customers[0].tenure == 12

    def test_batch_request_empty_fails(self) -> None:
        """An empty customers list must fail validation (min_length=1)."""
        from src.api.prediction_service.schemas import BatchPredictRequest

        with pytest.raises(ValidationError):
            BatchPredictRequest(customers=[])

    def test_batch_response_total_matches_predictions(self) -> None:
        """BatchPredictResponse.total must equal len(predictions)."""
        from src.api.prediction_service.schemas import (
            BatchPredictResponse,
            ChurnPredictionResponse,
        )

        pred = ChurnPredictionResponse(
            customerID="X",
            churn_probability=0.3,
            churn_prediction=False,
            p_structured=0.3,
            p_nlp=0.25,
            nlp_branch_available=True,
            model_version="late-fusion-v2",
        )
        resp = BatchPredictResponse(
            predictions=[pred],
            total=1,
            nlp_branch_available=True,
        )
        assert resp.total == len(resp.predictions)


# ---------------------------------------------------------------------------
# Test 5: InferenceService._build_structured_df
# ---------------------------------------------------------------------------


class TestInferenceServiceDataFrame:
    """Validates DataFrame reconstruction from raw Pydantic request fields."""

    def _make_service(self) -> Any:
        """Creates a minimal InferenceService with mock model artifacts."""
        from src.api.prediction_service.inference import InferenceService

        return InferenceService(
            structured_preprocessor=MagicMock(),
            structured_model=MagicMock(),
            nlp_model=MagicMock(),
            meta_model=MagicMock(),
            embedding_service_url="http://localhost:8001",
            model_version="late-fusion-v2",
            pca_components=20,
            timeout_seconds=5.0,
        )

    def test_dataframe_column_count(self) -> None:
        """Reconstructed DataFrame must have exactly 19 structured columns."""
        from src.api.prediction_service.inference import STRUCTURED_RAW_COLS
        from src.api.prediction_service.schemas import CustomerFeatureRequest

        service = self._make_service()
        customer = CustomerFeatureRequest(**VALID_CUSTOMER_PAYLOAD)
        df = service._build_structured_df([customer])

        assert df.shape == (1, len(STRUCTURED_RAW_COLS))
        assert list(df.columns) == STRUCTURED_RAW_COLS

    def test_total_charges_none_becomes_empty_string(self) -> None:
        """TotalCharges=None must be converted to '' for NumericCleaner compatibility."""
        from src.api.prediction_service.schemas import CustomerFeatureRequest

        service = self._make_service()
        customer = CustomerFeatureRequest(**{**VALID_CUSTOMER_PAYLOAD, "TotalCharges": None})
        df = service._build_structured_df([customer])

        assert df["TotalCharges"].iloc[0] == ""

    def test_batch_dataframe_row_count(self) -> None:
        """Batch of 3 customers must produce a DataFrame with 3 rows."""
        from src.api.prediction_service.schemas import CustomerFeatureRequest

        service = self._make_service()
        customers = [CustomerFeatureRequest(**VALID_CUSTOMER_PAYLOAD) for _ in range(3)]
        df = service._build_structured_df(customers)

        assert df.shape[0] == 3


# ---------------------------------------------------------------------------
# Test 6: Circuit Breaker — zero-vector fallback
# ---------------------------------------------------------------------------


class TestCircuitBreaker:
    """Validates the embedding service circuit breaker logic in InferenceService."""

    def _make_service(self) -> Any:
        """Creates InferenceService with mock model artifacts."""
        from src.api.prediction_service.inference import InferenceService

        return InferenceService(
            structured_preprocessor=MagicMock(),
            structured_model=MagicMock(),
            nlp_model=MagicMock(),
            meta_model=MagicMock(),
            embedding_service_url="http://localhost:8001",
            model_version="late-fusion-v2",
            pca_components=20,
            timeout_seconds=1.0,
        )

    @pytest.mark.asyncio
    async def test_timeout_triggers_zero_vector_fallback(self) -> None:
        """A TimeoutException must return a zero-vector and nlp_available=False."""
        import httpx

        service = self._make_service()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("timeout")
            embeddings, available = await service._get_embeddings(["test note"])

        assert available is False
        assert embeddings.shape == (1, 20)
        assert np.all(embeddings == 0.0)

    @pytest.mark.asyncio
    async def test_connection_error_triggers_zero_vector_fallback(self) -> None:
        """A connection error must return a zero-vector and nlp_available=False."""
        import httpx

        service = self._make_service()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("connection refused")
            embeddings, available = await service._get_embeddings(["test note"])

        assert available is False
        assert embeddings.shape == (1, 20)

    @pytest.mark.asyncio
    async def test_successful_embedding_call_returns_available_true(self) -> None:
        """A successful HTTP call must return the embeddings and nlp_available=True."""

        service = self._make_service()

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "embeddings": [[0.1] * 20],
            "model_version": "all-MiniLM-L6-v2-pca20",
            "dim": 20,
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = mock_response
            embeddings, available = await service._get_embeddings(["test note"])

        assert available is True
        assert embeddings.shape == (1, 20)
        assert embeddings[0][0] == pytest.approx(0.1)

    @pytest.mark.asyncio
    async def test_zero_vector_fallback_shape_matches_pca_components(self) -> None:
        """Zero-vector fallback shape must equal (n_customers, pca_components)."""
        import httpx

        service = self._make_service()
        notes = ["note 1", "note 2", "note 3"]

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.TimeoutException("timeout")
            embeddings, available = await service._get_embeddings(notes)

        assert embeddings.shape == (3, 20), (
            f"Expected (3, 20), got {embeddings.shape}. Zero-vector fallback must match (n_customers, pca_components)."
        )
        assert available is False
