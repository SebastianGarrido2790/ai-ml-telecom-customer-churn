"""
Unit tests for the Data Enrichment Generator.

This module focuses on the probabilistic-deterministic hybrid logic used to
generate synthetic customer ticket notes. It validates the primary LLM
(Google Gemini) call path and the fallback mechanisms (Ollama, Rule-based).

Tiered Fallback Coverage:
    - Priority 1: Cloud-based LLM (Gemini).
    - Priority 2: Local-hosted LLM (Ollama).
    - Priority 3: Deterministic rule-based fallback based on customer profile.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.components.data_enrichment.generator import (
    SyntheticNoteOutput,
    _deterministic_fallback,
    generate_ticket_note,
)
from src.components.data_enrichment.schemas import CustomerInputContext


@pytest.fixture
def mock_customer_context():
    return CustomerInputContext(
        customerID="1234-ABCD",
        tenure=12,
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        InternetService="Fiber optic",
        OnlineSecurity="No",
        OnlineBackup="No",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="Yes",
        StreamingMovies="Yes",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=85.50,
    )


@pytest.fixture
def mock_enrich_config():
    config = MagicMock()
    config.model_provider = "google"
    config.model_name = "gemini-test"
    config.base_url = None
    return config


@pytest.mark.asyncio
async def test_generate_ticket_note_google_success(mock_customer_context, mock_enrich_config):
    """Test successful primary LLM call (Google)."""
    expected_output = SyntheticNoteOutput(ticket_note="Success from Google", primary_sentiment_tag="Neutral")

    with patch("src.components.data_enrichment.generator.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_agent.run = AsyncMock()
        mock_agent.run.return_value.data = expected_output

        result = await generate_ticket_note(mock_customer_context, mock_enrich_config)

        assert result.ticket_note == "Success from Google"
        assert result.primary_sentiment_tag == "Neutral"
        mock_agent.run.assert_called_once()


@pytest.mark.asyncio
async def test_generate_ticket_note_google_fail_fallback_to_deterministic(mock_customer_context, mock_enrich_config):
    """Test that Google failure triggers deterministic fallback (non-hybrid)."""
    with patch("src.components.data_enrichment.generator.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_agent.run = AsyncMock(side_effect=Exception("API Error"))

        result = await generate_ticket_note(mock_customer_context, mock_enrich_config)

        # Fiber optic + No Tech Support + Month-to-month -> Frustrated note
        assert "Fiber optic" in result.ticket_note
        assert result.primary_sentiment_tag == "Frustrated"


@pytest.mark.asyncio
async def test_generate_ticket_note_hybrid_google_fail_ollama_success(mock_customer_context, mock_enrich_config):
    """Test hybrid mode: Google fails, Ollama succeeds."""
    mock_enrich_config.model_provider = "hybrid"
    mock_enrich_config.secondary_model_name = "ollama-test"
    mock_enrich_config.secondary_base_url = "http://ollama:11434"

    expected_ollama_output = {"response": '{"ticket_note": "Success from Ollama", "primary_sentiment_tag": "Neutral"}'}

    with patch("src.components.data_enrichment.generator.Agent") as mock_agent_class:
        mock_agent = mock_agent_class.return_value
        mock_agent.run = AsyncMock(side_effect=Exception("Google Down"))

        with patch("httpx.AsyncClient.post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json = MagicMock(return_value=expected_ollama_output)

            result = await generate_ticket_note(mock_customer_context, mock_enrich_config)

            assert result.ticket_note == "Success from Ollama"
            assert result.primary_sentiment_tag == "Neutral"


def test_deterministic_fallback_scenarios(mock_customer_context):
    """Test different deterministic fallback branches."""
    # Scenario: High-friction (already tested via fallback above, but let's be explicit)
    res = _deterministic_fallback(mock_customer_context)
    assert res.primary_sentiment_tag == "Frustrated"

    # Scenario: Loyal long-term (Two year contract)
    # Must also disable "Technical Issue" branch signals (OnlineSecurity/OnlineBackup)
    ctx_loyal = mock_customer_context.model_copy(update={"Contract": "Two year", "OnlineSecurity": "Yes"})
    res = _deterministic_fallback(ctx_loyal)
    assert res.primary_sentiment_tag == "Satisfied"

    # Scenario: New customer (Tenure <= 6)
    ctx_new = mock_customer_context.model_copy(update={"tenure": 3, "Contract": "One year", "OnlineSecurity": "Yes"})
    res = _deterministic_fallback(ctx_new)
    assert res.primary_sentiment_tag == "Neutral"
    assert "New customer" in res.ticket_note
