"""
Pydantic request and response schemas for the Embedding Microservice.

All I/O contracts are typed explicitly — no untyped dict payloads permitted
(Rule: Pydantic Request/Response Models).
"""

from pydantic import BaseModel, Field


class EmbedRequest(BaseModel):
    """Request body for POST /v1/embed.

    Accepts a batch of ticket note strings. Batch support allows the Prediction
    API to embed multiple notes in a single HTTP call during batch prediction,
    reducing inter-service latency.

    Attributes:
        ticket_notes: One or more raw ticket note strings to embed.
    """

    ticket_notes: list[str] = Field(
        ...,
        min_length=1,
        description="List of raw ticket note strings to convert to PCA-reduced embeddings.",
    )


class EmbedResponse(BaseModel):
    """Response body for POST /v1/embed.

    Attributes:
        embeddings: List of PCA-reduced embedding vectors, one per input note.
        Shape: (n_notes, pca_components).
        model_version: Identifier of the NLP model and PCA configuration used,
        e.g. 'all-MiniLM-L6-v2-pca20'.
        dim: Dimensionality of each embedding vector (= pca_components).
    """

    embeddings: list[list[float]] = Field(
        ...,
        description="PCA-reduced embedding vectors. Shape: (n_notes, dim).",
    )
    model_version: str = Field(
        ...,
        description="NLP model and PCA version identifier.",
    )
    dim: int = Field(
        ...,
        gt=0,
        description="Embedding vector dimensionality (= pca_components).",
    )


class HealthResponse(BaseModel):
    """Response body for GET /v1/health.

    Attributes:
        status: Service readiness status. Always 'healthy' on a 200 response.
        model_version: NLP model and PCA version currently loaded.
    """

    status: str = Field(default="healthy")
    model_version: str
