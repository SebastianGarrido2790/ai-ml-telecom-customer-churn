"""
Data contract definitions for the Telecom Customer Churn system.

This module defines:
1. Frozen dataclass entities for pipeline stage configurations (immutable configs).
2. Pydantic models for strict row-level validation of the raw Telco dataset
   and its enriched variant.

These contracts enforce the strict typing standard: no untyped dictionaries
bridging the Agent and the pipelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

# ============================================================================
# Pipeline Configuration Entities (Immutable Dataclasses)
# ============================================================================


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for the data ingestion pipeline stage.

    Attributes:
        root_dir: Root directory for ingestion artifacts.
        source_URL: Path or URL to the source data file.
        local_data_file: Local destination path for the ingested data.
        unzip_dir: Directory for unzipped/extracted data.
    """

    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for the data validation pipeline stage.

    Attributes:
        root_dir: Root directory for validation artifacts.
        STATUS_FILE: Path to the validation status output file.
        unzip_data_dir: Path to the data file to validate.
        all_schema: Dictionary containing the expected column schema.
    """

    root_dir: Path
    STATUS_FILE: str
    unzip_data_dir: Path
    all_schema: dict


@dataclass(frozen=True)
class DataEnrichmentConfig:
    """Configuration for the synthetic ticket note enrichment stage.

    Attributes:
        root_dir: Root directory for enrichment artifacts.
        raw_data_path: Path to the input raw dataset.
        enriched_data_file: Output path for the enriched dataset.
        prompts_dir: Directory containing prompt templates.
        all_schema: Dictionary containing the expected enriched column schema.
        model_name: Name of the LLM to use for enrichment.
        limit: Max number of rows to process (None for all).
        batch_size: Number of concurrent API calls.
    """

    root_dir: Path
    raw_data_path: Path
    enriched_data_file: Path
    prompts_dir: Path
    all_schema: dict
    model_provider: str
    model_name: str
    base_url: str | None
    secondary_model_name: str | None
    secondary_base_url: str | None
    limit: int | None
    batch_size: int


@dataclass(frozen=True)
class FeatureEngineeringConfig:
    """Configuration for the feature engineering stage.

    Attributes:
        root_dir: Root directory for feature engineering artifacts.
        input_data_path: Path to the input enriched dataset.
        train_data_path: Path to the output training features.
        test_data_path: Path to the output testing features.
        val_data_path: Path to the output validation features.
        structured_preprocessor_path: Path to the serialized structured preprocessor.
        nlp_preprocessor_path: Path to the serialized NLP preprocessor.
        embedding_model_name: Name of the sentence-transformer model.
        pca_components: Number of components for PCA dimensionality reduction.
        test_size: Proportion of the dataset to include in the test split.
        val_size: Proportion of the dataset to include in the validation split.
        random_state: Seed for reproducibility.
        target_column: Name of the target variable.
    """

    root_dir: Path
    input_data_path: Path
    train_data_path: Path
    test_data_path: Path
    val_data_path: Path
    structured_preprocessor_path: Path
    nlp_preprocessor_path: Path
    embedding_model_name: str
    pca_components: int
    test_size: float
    val_size: float
    random_state: int
    target_column: str


@dataclass(frozen=True)
class ModelTrainingConfig:
    """Configuration for the model training stage (Late Fusion).

    Attributes:
        root_dir: Root directory for training artifacts.
        train_data_path: Path to the input training features CSV.
        val_data_path: Path to the input validation features CSV.
        test_data_path: Path to the input test features CSV.
        structured_preprocessor_path: Path to the structured preprocessor artifact.
        nlp_preprocessor_path: Path to the NLP preprocessor artifact.
        structured_model_path: Output path for the trained structured branch model.
        nlp_model_path: Output path for the trained NLP branch model.
        meta_model_path: Output path for the trained meta-learner model.
        evaluation_report_path: Path to the model performance report.
        target_column: Name of the dependent variable.
        random_state: Seed for reproducibility.
        cv_folds: Number of cross-validation folds.
        structured_n_trials: Hyperparameter tuning trials for the structured branch.
        nlp_n_trials: Hyperparameter tuning trials for the NLP branch.
        meta_C: Inverse of regularization strength for the meta-learner.
        meta_max_iter: Maximum iterations for the meta-learner trainer.
        mlflow_uri: MLflow tracking server URI.
        experiment_name: Name of the MLflow experiment.
    """

    root_dir: Path
    train_data_path: Path
    val_data_path: Path
    test_data_path: Path
    structured_preprocessor_path: Path
    nlp_preprocessor_path: Path
    structured_model_path: Path
    nlp_model_path: Path
    meta_model_path: Path
    evaluation_report_path: Path
    target_column: str
    random_state: int
    cv_folds: int
    structured_n_trials: int
    nlp_n_trials: int
    meta_C: float
    meta_max_iter: int
    mlflow_uri: str
    experiment_name: str


# ============================================================================
# Pydantic Data Contracts (Row-Level Validation)
# ============================================================================


class TelcoCustomerRow(BaseModel):
    """Strict schema for one row of WA_Fn-UseC_-Telco-Customer-Churn.csv.

    Enforces type and range constraints matching the raw Telco dataset.
    TotalCharges is typed as Optional[str] because the raw CSV contains
    blank strings for some rows (e.g., customers with tenure=0).
    """

    customerID: str
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
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
    MonthlyCharges: float = Field(ge=0)
    TotalCharges: str | None = None
    Churn: str


class EnrichedTelcoRow(TelcoCustomerRow):
    """Extended schema with synthetic ticket note (Phase 2 output).

    Inherits all fields from TelcoCustomerRow and adds the
    AI-generated ticket note and sentiment fields.
    """

    ticket_note: str | None = Field(None, description="AI-generated customer complaint or interaction note")
    primary_sentiment_tag: str | None = Field(None, description="AI-classified sentiment of the customer")


@dataclass(frozen=True)
class EmbeddingServiceConfig:
    """Configuration for the Embedding Microservice (Phase 6).

    Loaded at startup by the Embedding Microservice via lifespan.
    Also consumed by the Prediction API to construct the embedding service URL
    and populate the circuit-breaker zero-vector fallback dimension.

    Attributes:
        host: Hostname of the embedding service.
              "localhost" for local runs; "embedding-service" inside Docker Compose.
        port: Port the embedding service listens on (default: 8001).
        timeout_seconds: HTTP client timeout for calls from the Prediction API.
        nlp_preprocessor_path: Path to the fitted nlp_preprocessor.pkl artifact.
        model_version: Human-readable version string logged in EmbedResponse.
        pca_components: Number of PCA output dimensions — used to construct the
                        zero-vector fallback in the circuit breaker.
    """

    host: str
    port: int
    timeout_seconds: float
    nlp_preprocessor_path: Path
    model_version: str
    pca_components: int


@dataclass(frozen=True)
class PredictionAPIConfig:
    """Configuration for the Prediction API microservice (Phase 6).

    Loaded at startup by the Prediction API via lifespan. Contains all artifact
    paths required to serve the Late Fusion inference pipeline and the URL of
    the Embedding Microservice it depends on.

    Attributes:
        host: Bind address for the Prediction API server.
        port: Port the Prediction API listens on (default: 8000).
        structured_preprocessor_path: Path to structured_preprocessor.pkl.
        structured_model_path: Path to structured_model.pkl (XGBoost Branch 1).
        nlp_model_path: Path to nlp_model.pkl (XGBoost Branch 2).
        meta_model_path: Path to meta_model.pkl (Logistic Regression stacker).
        embedding_service_url: Full URL for embedding calls, e.g.
                               "http://localhost:8001".
        model_version: Human-readable version string logged in responses.
        pca_components: NLP branch output dimension; used for zero-vector fallback.
    """

    host: str
    port: int
    structured_preprocessor_path: Path
    structured_model_path: Path
    nlp_model_path: Path
    meta_model_path: Path
    embedding_service_url: str
    model_version: str
    pca_components: int
