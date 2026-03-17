"""
Unit tests for the Data Enrichment schemas and data contracts.
Ensures that incoming customer context and outgoing synthetic notes are strictly validated.
"""

import pytest
from pydantic import ValidationError

from src.components.data_enrichment.schemas import CustomerInputContext, SyntheticNoteOutput


def test_customer_input_context_valid():
    """
    Tests the successful creation of a CustomerInputContext with valid data.

    Verifies that all attributes are correctly assigned and types are preserved.
    """
    data = {
        "customerID": "1234-ABCD",
        "tenure": 12,
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "MonthlyCharges": 85.50,
        "TechSupport": "No",
        "Churn": "Yes",
    }
    context = CustomerInputContext(**data)
    assert context.customerID == "1234-ABCD"
    assert context.tenure == 12


def test_customer_input_context_invalid_tenure():
    """
    Tests that a negative tenure value triggers a ValidationError.

    Enforces the business rule that tenure must be a non-negative integer.
    """
    data = {
        "customerID": "1234-ABCD",
        "tenure": -5,
        "InternetService": "Fiber optic",
        "Contract": "Month-to-month",
        "MonthlyCharges": 85.50,
        "TechSupport": "No",
        "Churn": "Yes",
    }
    with pytest.raises(ValidationError):
        CustomerInputContext(**data)


def test_customer_input_context_invalid_literals():
    """
    Tests that invalid literal values trigger a ValidationError.

    Verifies that 'InternetService' and 'Churn' are restricted to predefined values.
    """
    data = {
        "customerID": "1234-ABCD",
        "tenure": 12,
        "InternetService": "Dial-up",  # Invalid
        "Contract": "Month-to-month",
        "MonthlyCharges": 85.50,
        "TechSupport": "No",
        "Churn": "Maybe",  # Invalid
    }
    with pytest.raises(ValidationError):
        CustomerInputContext(**data)


def test_synthetic_note_output_valid():
    """
    Tests the successful creation of a SyntheticNoteOutput with valid data.

    Ensures that LLM outputs can be correctly mapped to the internal schema.
    """
    data = {
        "ticket_note": "Customer called experiencing outage.",
        "primary_sentiment_tag": "Frustrated",
    }
    output = SyntheticNoteOutput(**data)
    assert output.ticket_note == "Customer called experiencing outage."


def test_synthetic_note_output_invalid_tag():
    """
    Tests that an invalid sentiment tag triggers a ValidationError.

    Enforces consistency in the categorical sentiment tagging.
    """
    data = {
        "ticket_note": "Customer called experiencing outage.",
        "primary_sentiment_tag": "Angry",  # Not in Literal
    }
    with pytest.raises(ValidationError):
        SyntheticNoteOutput(**data)
