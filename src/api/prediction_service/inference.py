"""
InferenceService — Late Fusion prediction orchestrator.

This module is the single owner of all inference logic for the Prediction API.
It handles:
    1. DataFrame reconstruction from raw Pydantic request fields.
    2. Structured feature preprocessing via structured_preprocessor.pkl.
    3. NLP embedding retrieval from the Embedding Microservice (with circuit breaker).
    4. Base model probability predictions (Branch 1 structured, Branch 2 NLP).
    5. Meta-learner stacking and final churn score.

The router is a pure HTTP conductor — it calls InferenceService.predict() and
returns the result. No transformation logic lives in the router (Decision D2).

Circuit Breaker Contract:
    If the Embedding Microservice is unreachable (timeout, connection error, or
    non-200 response), InferenceService:
        - Logs a WARNING with the exact error.
        - Falls back to a zero-vector of shape (n, pca_components).
        - Sets nlp_branch_available=False in the response.
        - Continues with Branch 1 structured prediction uninterrupted.
    This ensures the Prediction API never returns a 5xx when only the
    embedding dependency is down.
"""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from src.api.prediction_service.schemas import (
    BatchPredictResponse,
    ChurnPredictionResponse,
    CustomerFeatureRequest,
)
from src.utils.array_utils import ensure_ndarray
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Raw input column order — must match the column order the
# structured_preprocessor ColumnTransformer was fitted on in Phase 4.
# ---------------------------------------------------------------------------
STRUCTURED_RAW_COLS: list[str] = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


class InferenceService:
    """Orchestrates the full Late Fusion inference pipeline for a batch of customers.

    Holds references to all model artifacts and the embedding service URL.
    Designed to be instantiated once at startup and reused across requests.

    Attributes:
        structured_preprocessor: Fitted ColumnTransformer for structured features.
        structured_model: XGBoost Branch 1 classifier.
        nlp_model: XGBoost Branch 2 classifier.
        meta_model: Logistic Regression meta-learner.
        embedding_service_url: Base URL of the Embedding Microservice.
        model_version: Version string echoed in all responses.
        pca_components: NLP branch output dimension for zero-vector fallback.
        _timeout: httpx timeout for embedding service calls.
    """

    def __init__(
        self,
        structured_preprocessor: object,
        structured_model: object,
        nlp_model: object,
        meta_model: object,
        embedding_service_url: str,
        model_version: str,
        pca_components: int,
        timeout_seconds: float = 5.0,
    ) -> None:
        """Initialises InferenceService with all model artifacts and config.

        Args:
            structured_preprocessor: Fitted sklearn ColumnTransformer.
            structured_model: Fitted XGBoost classifier (Branch 1).
            nlp_model: Fitted XGBoost classifier (Branch 2).
            meta_model: Fitted Logistic Regression meta-learner.
            embedding_service_url: Base URL, e.g. 'http://localhost:8001'.
            model_version: Version string for response metadata.
            pca_components: PCA output dimension; used for zero-vector fallback.
            timeout_seconds: HTTP timeout for embedding service calls.
        """
        self.structured_preprocessor = structured_preprocessor
        self.structured_model = structured_model
        self.nlp_model = nlp_model
        self.meta_model = meta_model
        self.embedding_service_url = embedding_service_url
        self.model_version = model_version
        self.pca_components = pca_components
        self._timeout = httpx.Timeout(timeout_seconds)

    def _build_structured_df(self, customers: list[CustomerFeatureRequest]) -> pd.DataFrame:
        """Reconstructs a DataFrame from raw Pydantic request fields.

        Column order matches the training-time ColumnTransformer exactly.
        TotalCharges blank strings are preserved — NumericCleaner handles
        coercion to NaN downstream.

        Args:
            customers: List of validated CustomerFeatureRequest instances.

        Returns:
            DataFrame with STRUCTURED_RAW_COLS columns in correct order.
        """
        records = []
        for c in customers:
            records.append(
                {
                    "tenure": c.tenure,
                    "MonthlyCharges": c.MonthlyCharges,
                    "TotalCharges": c.TotalCharges if c.TotalCharges is not None else "",
                    "gender": c.gender,
                    "SeniorCitizen": c.SeniorCitizen,
                    "Partner": c.Partner,
                    "Dependents": c.Dependents,
                    "PhoneService": c.PhoneService,
                    "MultipleLines": c.MultipleLines,
                    "InternetService": c.InternetService,
                    "OnlineSecurity": c.OnlineSecurity,
                    "OnlineBackup": c.OnlineBackup,
                    "DeviceProtection": c.DeviceProtection,
                    "TechSupport": c.TechSupport,
                    "StreamingTV": c.StreamingTV,
                    "StreamingMovies": c.StreamingMovies,
                    "Contract": c.Contract,
                    "PaperlessBilling": c.PaperlessBilling,
                    "PaymentMethod": c.PaymentMethod,
                }
            )
        return pd.DataFrame(records, columns=pd.Index(STRUCTURED_RAW_COLS))

    async def _get_embeddings(self, ticket_notes: list[str]) -> tuple[np.ndarray, bool]:
        """Fetches PCA-reduced embeddings from the Embedding Microservice.

        Implements the circuit breaker pattern: any HTTP error, timeout, or
        non-200 response triggers the zero-vector fallback. The Prediction API
        continues serving structured branch predictions uninterrupted.

        Args:
            ticket_notes: List of raw ticket note strings to embed.

        Returns:
            Tuple of:
                - embeddings array of shape (n, pca_components).
                - nlp_branch_available: False if fallback was used.
        """
        n = len(ticket_notes)
        url = f"{self.embedding_service_url}/v1/embed"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    url,
                    json={"ticket_notes": ticket_notes},
                )
                response.raise_for_status()
                data = response.json()
                embeddings = np.array(data["embeddings"], dtype=np.float32)
                return embeddings, True

        except httpx.TimeoutException as exc:
            logger.warning(
                f"Embedding service timed out after {self._timeout.read}s: {exc!s}. "
                f"Using zero-vector fallback for {n} customer(s)."
            )
        except httpx.HTTPStatusError as exc:
            logger.warning(
                f"Embedding service returned HTTP {exc.response.status_code}: "
                f"{exc!s}. Using zero-vector fallback for {n} customer(s)."
            )
        except httpx.RequestError as exc:
            logger.warning(
                f"Embedding service unreachable ({type(exc).__name__}): {exc!s}. "
                f"Using zero-vector fallback for {n} customer(s)."
            )

        # Circuit breaker: zero-vector fallback
        return np.zeros((n, self.pca_components), dtype=np.float32), False

    async def predict_batch(self, customers: list[CustomerFeatureRequest]) -> BatchPredictResponse:
        """Runs the full Late Fusion inference pipeline for a batch of customers.

        Protocol:
            1. Reconstruct structured DataFrame from raw request fields.
            2. Apply structured_preprocessor → 46-dim structured feature matrix.
            3. Fetch PCA-reduced embeddings from Embedding Microservice (or fallback).
            4. Branch 1: structured_model.predict_proba → P_struct per customer.
            5. Branch 2: nlp_model.predict_proba → P_nlp per customer.
            6. Stack [P_struct, P_nlp] → meta_model.predict_proba → final score.
            7. Build and return BatchPredictResponse.

        Args:
            customers: Validated list of CustomerFeatureRequest instances.

        Returns:
            BatchPredictResponse with one ChurnPredictionResponse per customer.
        """
        # --- Step 1 & 2: Structured preprocessing ---
        struct_df = self._build_structured_df(customers)

        struct_features: np.ndarray | DataFrame | Series = self.structured_preprocessor.transform(  # type: ignore[attr-defined]
            struct_df
        )

        # Ensure numpy array for model consumption
        struct_features_np = ensure_ndarray(struct_features)

        # --- Step 3: NLP embeddings (with circuit breaker) ---
        ticket_notes = [c.ticket_note for c in customers]
        nlp_features: np.ndarray
        nlp_available: bool
        nlp_features, nlp_available = await self._get_embeddings(ticket_notes)

        # --- Steps 4 & 5: Base model probabilities ---
        p_struct_arr: np.ndarray = self.structured_model.predict_proba(  # type: ignore[attr-defined]
            struct_features_np
        )[:, 1]
        p_nlp_arr: np.ndarray = self.nlp_model.predict_proba(  # type: ignore[attr-defined]
            nlp_features
        )[:, 1]

        # --- Step 6: Meta-learner stacking ---
        stack = np.column_stack([p_struct_arr, p_nlp_arr])
        churn_probs: np.ndarray = self.meta_model.predict_proba(stack)[:, 1]  # type: ignore[attr-defined]

        # --- Step 7: Build responses ---
        predictions: list[ChurnPredictionResponse] = []
        for i, customer in enumerate(customers):
            prob = float(churn_probs[i])
            predictions.append(
                ChurnPredictionResponse(
                    customerID=customer.customerID,
                    churn_probability=round(prob, 6),
                    churn_prediction=prob >= 0.5,
                    p_structured=round(float(p_struct_arr[i]), 6),
                    p_nlp=round(float(p_nlp_arr[i]), 6),
                    nlp_branch_available=nlp_available,
                    model_version=self.model_version,
                )
            )

        return BatchPredictResponse(
            predictions=predictions,
            total=len(predictions),
            nlp_branch_available=nlp_available,
        )
