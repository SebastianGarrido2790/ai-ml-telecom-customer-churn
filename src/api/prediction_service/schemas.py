"""
Pydantic request and response schemas for the Prediction API microservice.

Raw pre-preprocessed fields are accepted (exactly as they appear in the
enriched dataset). The preprocessing step happens inside InferenceService,
not in the request schema — the schema's job is input validation only.

All fields match the raw Telco dataset types. TotalCharges is str | None
because the raw CSV contains blank strings for customers with tenure=0,
which NumericCleaner converts to NaN before imputation.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CustomerFeatureRequest(BaseModel):
    """Raw feature input for a single customer churn prediction.

    Contains all 19 structured fields and the ticket_note text field.
    No preprocessing is applied at schema level — raw values are passed
    directly to the structured_preprocessor and embedding service.

    Attributes:
        customerID: Optional identifier for traceability in batch responses.
        gender: Customer gender as stored in the raw dataset.
        SeniorCitizen: Binary flag (0 or 1).
        Partner: Whether the customer has a partner ('Yes' / 'No').
        Dependents: Whether the customer has dependents ('Yes' / 'No').
        tenure: Months with the company (>= 0).
        PhoneService: Phone service subscription status.
        MultipleLines: Multiple lines status.
        InternetService: Internet connection type.
        OnlineSecurity: Online security add-on status.
        OnlineBackup: Online backup add-on status.
        DeviceProtection: Device protection add-on status.
        TechSupport: Tech support add-on status.
        StreamingTV: Streaming TV add-on status.
        StreamingMovies: Streaming movies add-on status.
        Contract: Contract term type.
        PaperlessBilling: Paperless billing preference.
        PaymentMethod: Payment method.
        MonthlyCharges: Monthly charge amount (>= 0).
        TotalCharges: Total charges as string (blank for tenure=0 customers).
        ticket_note: Raw support interaction note for NLP branch embedding.
    """

    customerID: str | None = Field(
        default=None,
        description="Optional customer identifier for response traceability.",
    )
    gender: str
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(..., ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: str | None = Field(
        default=None,
        description="Total charges as string. Blank strings are valid (tenure=0 customers).",
    )
    ticket_note: str = Field(
        ...,
        description="Raw support interaction note for NLP branch embedding.",
    )


class ChurnPredictionResponse(BaseModel):
    """Churn risk prediction for a single customer.

    Attributes:
        customerID: Echoed from the request for response traceability.
        churn_probability: Final meta-learner probability of churn, range [0, 1].
        churn_prediction: Binary churn label at threshold=0.5.
        p_structured: Branch 1 (structured features) churn probability.
        p_nlp: Branch 2 (NLP embeddings) churn probability. 0.0 when the embedding
        service is unreachable (circuit breaker). nlp_branch_available: False when
        the embedding service was unreachable and the zero-vector fallback was used
        for Branch 2. model_version: Identifies the Late Fusion model version serving
        this request.
    """

    customerID: str | None
    churn_probability: float = Field(..., ge=0.0, le=1.0)
    churn_prediction: bool
    p_structured: float = Field(..., ge=0.0, le=1.0)
    p_nlp: float = Field(..., ge=0.0, le=1.0)
    nlp_branch_available: bool
    model_version: str


class BatchPredictRequest(BaseModel):
    """Request body for POST /v1/predict/batch.

    Attributes:
        customers: List of customer feature records to score in a single call.
    """

    customers: list[CustomerFeatureRequest] = Field(
        ...,
        min_length=1,
        description="List of customer feature records for bulk scoring.",
    )


class BatchPredictResponse(BaseModel):
    """Response body for POST /v1/predict/batch.

    Attributes:
        predictions: Ordered list of churn predictions, one per input customer.
        total: Total number of customers scored in this batch.
        nlp_branch_available: False if the embedding service was unreachable
                              for any prediction in this batch.
    """

    predictions: list[ChurnPredictionResponse]
    total: int
    nlp_branch_available: bool


class PredictionHealthResponse(BaseModel):
    """Response body for GET /v1/health.

    Attributes:
        status: Always 'healthy' on a 200 response.
        model_version: Late Fusion model version currently loaded.
    """

    status: str = Field(default="healthy")
    model_version: str
