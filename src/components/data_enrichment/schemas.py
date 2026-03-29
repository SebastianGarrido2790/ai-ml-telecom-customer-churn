"""
Pydantic data models for structured data exchange in the Agentic Data Enrichment phase.

Defines the input contract for customer CRM context passed to the LLM and the
output contract for the structured synthetic note returned by the agent.

Leakage Prevention (C1 Fix):
    The `Churn` field has been deliberately removed from `CustomerInputContext`.
    Passing the target label to the LLM caused it to generate notes that explicitly
    referenced churn intent ("planning to switch", "decision to churn"), creating
    a near-perfect proxy of the target variable in the NLP embeddings.

    The schema now exposes all observable CRM signals a real support agent would
    see during a live call — richer context than the original 6-field schema —
    without any knowledge of the customer's eventual churn decision.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CustomerInputContext(BaseModel):
    """Input contract for a single customer row passed to the enrichment LLM.

    Contains only signals observable in a real CRM at the time of a support
    interaction. The target variable (Churn) is intentionally excluded to
    prevent the LLM from encoding label information into the generated notes.

    Attributes:
        customerID: Unique customer identifier for row traceability.
        tenure: Months the customer has been with the company.
        gender: Customer gender.
        SeniorCitizen: Whether the customer is a senior citizen (0 or 1).
        Partner: Whether the customer has a partner.
        Dependents: Whether the customer has dependents.
        InternetService: Internet connection type.
        OnlineSecurity: Whether the customer has online security add-on.
        OnlineBackup: Whether the customer has online backup add-on.
        DeviceProtection: Whether the customer has device protection add-on.
        TechSupport: Whether the customer has tech support add-on.
        StreamingTV: Whether the customer has streaming TV add-on.
        StreamingMovies: Whether the customer has streaming movies add-on.
        Contract: Contract term type.
        PaperlessBilling: Whether the customer uses paperless billing.
        PaymentMethod: Customer's payment method.
        MonthlyCharges: Monthly charge amount billed to the customer.
    """

    customerID: str = Field(..., description="Unique customer identifier")
    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed with the company")
    gender: str = Field(..., description="Customer gender")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Whether the customer is a senior citizen (1) or not (0)")
    Partner: str = Field(..., description="Whether the customer has a partner")
    Dependents: str = Field(..., description="Whether the customer has dependents")
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer's internet service connection type"
    )
    OnlineSecurity: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has online security"
    )
    OnlineBackup: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has online backup"
    )
    DeviceProtection: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has device protection"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has tech support"
    )
    StreamingTV: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has streaming TV"
    )
    StreamingMovies: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has streaming movies"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="The contract term of the customer"
    )
    PaperlessBilling: str = Field(..., description="Whether the customer uses paperless billing")
    PaymentMethod: str = Field(..., description="Customer's payment method")
    MonthlyCharges: float = Field(..., ge=0, description="The amount charged to the customer monthly")


class SyntheticNoteOutput(BaseModel):
    """Output contract for the structured synthetic note returned by the LLM.

    Guarantees that all agent outputs are parseable and categorically valid
    before being written to the enriched dataset artifact.

    Attributes:
        ticket_note: Short realistic CRM note describing a customer interaction.
        primary_sentiment_tag: Validated categorical sentiment label.
    """

    ticket_note: str = Field(
        ...,
        min_length=1,
        description="A short, realistic synthetic ticket note describing a customer interaction.",
    )
    primary_sentiment_tag: Literal[
        "Frustrated",
        "Dissatisfied",
        "Neutral",
        "Satisfied",
        "Billing Inquiry",
        "Technical Issue",
    ] = Field(..., description="The main sentiment or topic category of the note.")
