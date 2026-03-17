"""
This module defines the Pydantic data models used for structured data exchange
in the Agentic Data Enrichment phase. It includes schemas for both the input
customer context and the synthesized LLM output.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CustomerInputContext(BaseModel):
    """
    Schema for validating the structure of a customer row passed
    into the LLM for ticket note synthesis.
    """

    customerID: str = Field(..., description="Unique customer identifier")
    tenure: int = Field(
        ..., ge=0, description="Number of months the customer has stayed with the company"
    )
    InternetService: Literal["DSL", "Fiber optic", "No"] = Field(
        ..., description="Customer's internet service connection type"
    )
    Contract: Literal["Month-to-month", "One year", "Two year"] = Field(
        ..., description="The contract term of the customer"
    )
    MonthlyCharges: float = Field(
        ..., ge=0, description="The amount charged to the customer monthly"
    )
    TechSupport: Literal["Yes", "No", "No internet service"] = Field(
        ..., description="Whether the customer has tech support or not"
    )
    Churn: Literal["Yes", "No"] = Field(..., description="Whether the customer churned or not")


class SyntheticNoteOutput(BaseModel):
    """
    Schema guaranteeing deterministic execution shape for the Pydantic-AI Agent.
    """

    ticket_note: str = Field(
        ...,
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
