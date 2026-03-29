"""
Unit tests for the Data Enrichment schemas and data contracts.

Tests cover the I/O boundary of the Agentic enrichment pipeline:
    - CustomerInputContext: validates all CRM fields, enforces type and range
      constraints, and rejects invalid Literal values.
    - SyntheticNoteOutput: validates the structured LLM output contract and
      rejects any sentiment tag not in the allowed set.

Leakage Prevention (C1 Fix):
    The `Churn` field has been removed from `CustomerInputContext`. All test
    fixtures have been updated to reflect the new 17-field schema. The invalid
    Literal test now uses `InternetService: "Dial-up"` as the sole invalid
    field, replacing the previously dual-invalid `Churn: "Maybe"` assertion.
"""

import pytest
from pydantic import ValidationError

from src.components.data_enrichment.schemas import CustomerInputContext, SyntheticNoteOutput

# ---------------------------------------------------------------------------
# Shared fixture: minimal valid CustomerInputContext payload
# ---------------------------------------------------------------------------

VALID_CONTEXT_PAYLOAD: dict = {
    "customerID": "1234-ABCD",
    "tenure": 12,
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
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
}


# ---------------------------------------------------------------------------
# CustomerInputContext tests
# ---------------------------------------------------------------------------


def test_customer_input_context_valid() -> None:
    """A fully valid CRM payload must be accepted without errors.

    Verifies that all 17 fields are correctly assigned and types are
    preserved. Confirms Churn is no longer a required field.
    """
    context = CustomerInputContext(**VALID_CONTEXT_PAYLOAD)

    assert context.customerID == "1234-ABCD"
    assert context.tenure == 12
    assert context.InternetService == "Fiber optic"
    assert context.Contract == "Month-to-month"
    assert context.MonthlyCharges == pytest.approx(85.50)
    assert not hasattr(context, "Churn"), "Churn must not be present on CustomerInputContext after C1 leakage fix."


def test_customer_input_context_invalid_tenure() -> None:
    """A negative tenure value must raise a ValidationError.

    Enforces the business rule that tenure must be a non-negative integer.
    """
    payload = {**VALID_CONTEXT_PAYLOAD, "tenure": -5}

    with pytest.raises(ValidationError):
        CustomerInputContext(**payload)


def test_customer_input_context_invalid_senior_citizen() -> None:
    """A SeniorCitizen value outside [0, 1] must raise a ValidationError.

    Enforces the binary constraint on this integer flag field.
    """
    payload = {**VALID_CONTEXT_PAYLOAD, "SeniorCitizen": 5}

    with pytest.raises(ValidationError):
        CustomerInputContext(**payload)


def test_customer_input_context_invalid_internet_service() -> None:
    """An InternetService value outside the allowed Literal set must raise ValidationError.

    'Dial-up' is not a valid category — only 'DSL', 'Fiber optic', and 'No' are permitted.
    This replaces the previous dual-invalid test that relied on Churn='Maybe'.
    """
    payload = {**VALID_CONTEXT_PAYLOAD, "InternetService": "Dial-up"}

    with pytest.raises(ValidationError):
        CustomerInputContext(**payload)


def test_customer_input_context_invalid_contract() -> None:
    """An invalid Contract value must raise a ValidationError."""
    payload = {**VALID_CONTEXT_PAYLOAD, "Contract": "Weekly"}

    with pytest.raises(ValidationError):
        CustomerInputContext(**payload)


def test_customer_input_context_invalid_monthly_charges() -> None:
    """A negative MonthlyCharges value must raise a ValidationError."""
    payload = {**VALID_CONTEXT_PAYLOAD, "MonthlyCharges": -25.0}

    with pytest.raises(ValidationError):
        CustomerInputContext(**payload)


def test_customer_input_context_churn_field_absent() -> None:
    """Passing Churn as an extra field must be silently ignored or raise ValidationError.

    Since CustomerInputContext does not declare Churn, Pydantic v2 will either
    ignore the extra field (default) or reject it (if model_config forbids extras).
    Either behaviour is acceptable — what is not acceptable is the field being
    stored on the model and forwarded to the LLM.
    """
    payload = {**VALID_CONTEXT_PAYLOAD, "Churn": "Yes"}
    context = CustomerInputContext(**payload)

    assert not hasattr(context, "Churn"), (
        "Churn must not be stored on CustomerInputContext — it must never reach the LLM prompt."
    )


# ---------------------------------------------------------------------------
# SyntheticNoteOutput tests
# ---------------------------------------------------------------------------


def test_synthetic_note_output_valid() -> None:
    """A valid note and allowed sentiment tag must be accepted without errors."""
    data = {
        "ticket_note": "Cust contacted support regarding recurring connectivity issues.",
        "primary_sentiment_tag": "Frustrated",
    }
    output = SyntheticNoteOutput(**data)

    assert output.ticket_note == data["ticket_note"]
    assert output.primary_sentiment_tag == "Frustrated"


def test_synthetic_note_output_all_valid_tags() -> None:
    """Every allowed sentiment tag must be accepted by the output schema."""
    allowed_tags = [
        "Frustrated",
        "Dissatisfied",
        "Neutral",
        "Satisfied",
        "Billing Inquiry",
        "Technical Issue",
    ]
    for tag in allowed_tags:
        output = SyntheticNoteOutput(ticket_note="Sample interaction note.", primary_sentiment_tag=tag)
        assert output.primary_sentiment_tag == tag


def test_synthetic_note_output_invalid_tag() -> None:
    """A sentiment tag outside the allowed Literal set must raise a ValidationError.

    'Angry' is not a valid category. This test guards against LLM hallucination
    passing an unconstrained string through the output contract.
    """
    data = {
        "ticket_note": "Cust called experiencing outage.",
        "primary_sentiment_tag": "Angry",
    }
    with pytest.raises(ValidationError):
        SyntheticNoteOutput(**data)


def test_synthetic_note_output_empty_ticket_note() -> None:
    """An empty ticket_note string must raise a ValidationError.

    Enforces that every enriched row contains a non-empty CRM note
    before being written to the artifact.
    """
    data = {
        "ticket_note": "",
        "primary_sentiment_tag": "Neutral",
    }
    with pytest.raises(ValidationError):
        SyntheticNoteOutput(**data)
