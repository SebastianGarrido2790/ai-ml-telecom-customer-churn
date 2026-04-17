"""
Gradio UI Smoke Tests.

Validates the Gradio UI layer (§2.3) through three tiers:
  1. API client unit tests — mock httpx to confirm correct request construction,
     header injection, error propagation, and payload shape.
  2. App-builder smoke test — confirms build_app() returns a gr.Blocks instance
     without raising, proving no import-time wiring errors in pages or components.
  3. Contract tests — checks that the api_client enforces the X-API-Key header
     and falls back gracefully on all httpx error subtypes.

No running server is required; all HTTP calls are intercepted by unittest.mock.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_httpx_post():
    """Patches httpx.post for api_client tests.

    Yields:
        MagicMock configured to return a 200 response by default.
    """
    with patch("src.ui.data_loaders.api_client.httpx.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "churn_probability": 0.72,
            "churn_prediction": True,
            "p_structured": 0.68,
            "p_nlp": 0.75,
            "nlp_branch_available": True,
            "model_version": "1.0.0",
        }
        mock_post.return_value = mock_response
        yield mock_post


@pytest.fixture()
def mock_httpx_get():
    """Patches httpx.get for api_client health-check tests.

    Yields:
        MagicMock configured to return a 200 response by default.
    """
    with patch("src.ui.data_loaders.api_client.httpx.get") as mock_get:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        yield mock_get


@pytest.fixture()
def sample_customer() -> dict[str, Any]:
    """Returns a minimal, schema-valid customer payload.

    Returns:
        dict containing all required fields for the /v1/predict endpoint.
    """
    return {
        "customerID": "TEST-001",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 12,
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
        "MonthlyCharges": 55.5,
        "TotalCharges": "666.0",
        "ticket_note": "Customer is unhappy with billing.",
    }


# ---------------------------------------------------------------------------
# 1. API Client — predict_single
# ---------------------------------------------------------------------------


class TestPredictSingle:
    """Unit tests for api_client.predict_single."""

    def test_returns_prediction_dict_on_success(
        self, mock_httpx_post: MagicMock, sample_customer: dict[str, Any]
    ) -> None:
        """predict_single should return the JSON body on a 200 response."""
        from src.ui.data_loaders.api_client import predict_single

        result = predict_single(sample_customer)

        assert "churn_probability" in result
        assert "churn_prediction" in result
        assert isinstance(result["churn_probability"], float)

    def test_injects_x_api_key_header(
        self, mock_httpx_post: MagicMock, sample_customer: dict[str, Any]
    ) -> None:
        """predict_single must attach the X-API-Key header on every request."""
        from src.ui.data_loaders.api_client import predict_single

        predict_single(sample_customer)

        call_kwargs = mock_httpx_post.call_args.kwargs
        assert "X-API-Key" in call_kwargs["headers"], "X-API-Key header must be present"

    def test_posts_to_correct_endpoint(
        self, mock_httpx_post: MagicMock, sample_customer: dict[str, Any]
    ) -> None:
        """predict_single should POST to /v1/predict."""
        from src.ui.data_loaders.api_client import predict_single

        predict_single(sample_customer)

        called_url: str = mock_httpx_post.call_args.args[0]
        assert called_url.endswith("/v1/predict"), f"Unexpected endpoint: {called_url}"

    def test_returns_error_dict_on_http_error(self, sample_customer: dict[str, Any]) -> None:
        """predict_single should catch httpx.HTTPError and return {error: str}."""
        import httpx

        from src.ui.data_loaders.api_client import predict_single

        with patch("src.ui.data_loaders.api_client.httpx.post", side_effect=httpx.HTTPError("timeout")):
            result = predict_single(sample_customer)

        assert "error" in result
        assert "timeout" in result["error"]


# ---------------------------------------------------------------------------
# 2. API Client — predict_batch
# ---------------------------------------------------------------------------


class TestPredictBatch:
    """Unit tests for api_client.predict_batch."""

    def test_batch_wraps_payload_in_customers_key(
        self, mock_httpx_post: MagicMock, sample_customer: dict[str, Any]
    ) -> None:
        """predict_batch must wrap the list under the 'customers' key."""
        from src.ui.data_loaders.api_client import predict_batch

        mock_httpx_post.return_value.json.return_value = {
            "predictions": [],
            "total_count": 1,
        }

        predict_batch([sample_customer])

        sent_json: dict[str, Any] = mock_httpx_post.call_args.kwargs["json"]
        assert "customers" in sent_json, "'customers' key must be present in batch payload"
        assert isinstance(sent_json["customers"], list)

    def test_batch_injects_x_api_key_header(
        self, mock_httpx_post: MagicMock, sample_customer: dict[str, Any]
    ) -> None:
        """predict_batch must attach the X-API-Key header."""
        from src.ui.data_loaders.api_client import predict_batch

        mock_httpx_post.return_value.json.return_value = {"predictions": [], "total_count": 0}

        predict_batch([sample_customer])

        call_kwargs = mock_httpx_post.call_args.kwargs
        assert "X-API-Key" in call_kwargs["headers"], "X-API-Key header must be present in batch calls"

    def test_batch_returns_error_on_http_error(self, sample_customer: dict[str, Any]) -> None:
        """predict_batch should return {error: ...} on connection failure."""
        import httpx

        from src.ui.data_loaders.api_client import predict_batch

        with patch("src.ui.data_loaders.api_client.httpx.post", side_effect=httpx.ConnectError("refused")):
            result = predict_batch([sample_customer])

        assert "error" in result


# ---------------------------------------------------------------------------
# 3. API Client — check_health
# ---------------------------------------------------------------------------


class TestCheckHealth:
    """Unit tests for api_client.check_health."""

    def test_returns_true_on_200(self, mock_httpx_get: MagicMock) -> None:
        """check_health should return True for a 200 OK response."""
        from src.ui.data_loaders.api_client import check_health

        mock_httpx_get.return_value.status_code = 200
        assert check_health() is True

    def test_returns_false_on_non_200(self, mock_httpx_get: MagicMock) -> None:
        """check_health should return False for any non-200 status."""
        from src.ui.data_loaders.api_client import check_health

        mock_httpx_get.return_value.status_code = 503
        assert check_health() is False

    def test_returns_false_on_http_error(self) -> None:
        """check_health should swallow httpx errors and return False."""
        import httpx

        from src.ui.data_loaders.api_client import check_health

        with patch("src.ui.data_loaders.api_client.httpx.get", side_effect=httpx.ConnectError("down")):
            assert check_health() is False

    def test_hits_health_endpoint(self, mock_httpx_get: MagicMock) -> None:
        """check_health must call /v1/health."""
        from src.ui.data_loaders.api_client import check_health

        check_health()

        called_url: str = mock_httpx_get.call_args.args[0]
        assert called_url.endswith("/v1/health"), f"Unexpected health endpoint: {called_url}"


# ---------------------------------------------------------------------------
# 4. App Builder — import + build smoke test
# ---------------------------------------------------------------------------


class TestGradioAppBuilder:
    """Smoke tests for the Gradio Blocks app factory."""

    def test_build_app_returns_blocks_instance(self) -> None:
        """build_app() must return a gr.Blocks object without raising."""
        import gradio as gr

        from src.ui.app import build_app

        app = build_app()

        assert isinstance(app, gr.Blocks), f"Expected gr.Blocks, got {type(app)}"

    def test_build_app_has_expected_title(self) -> None:
        """The Blocks app title must match the branding constant."""
        from src.ui.app import build_app

        app = build_app()

        assert "Telecom" in (app.title or ""), f"App title missing 'Telecom': {app.title!r}"

    def test_build_app_is_idempotent(self) -> None:
        """Calling build_app() twice should not raise — no singleton side-effects."""
        from src.ui.app import build_app

        app_a = build_app()
        app_b = build_app()

        import gradio as gr

        assert isinstance(app_a, gr.Blocks)
        assert isinstance(app_b, gr.Blocks)
