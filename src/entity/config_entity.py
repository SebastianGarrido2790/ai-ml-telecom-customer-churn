"""
Data contract definitions for the Telecom Customer Churn system.

This module defines:
1. Frozen dataclass entities for pipeline stage configurations (immutable configs).
2. Pydantic models for strict row-level validation of the raw Telco dataset
   and its enriched variant.

These contracts enforce a strict typing standard: no untyped dictionaries
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
    all_schema: dict[str, str]


@dataclass(frozen=True)
class DataEnrichmentConfig:
    """Configuration for the synthetic ticket note enrichment stage.

    Attributes:
        root_dir: Root directory for enrichment artifacts.
        raw_data_path: Path to the input raw dataset.
        enriched_data_file: Output path for the enriched dataset.
        prompts_dir: Directory containing prompt templates.
        all_schema: Dictionary containing the expected enriched column schema.
        model_provider: Provider strategy (google, openai, hybrid).
        model_name: Name of the primary LLM for enrichment.
        base_url: Optional base URL override for the primary LLM provider.
        secondary_model_name: Name of the fallback LLM model.
        secondary_base_url: Optional base URL for the secondary LLM provider.
        limit: Max number of rows to process (None for all).
        batch_size: Number of concurrent API calls per batch.
    """

    root_dir: Path
    raw_data_path: Path
    enriched_data_file: Path
    prompts_dir: Path
    all_schema: dict[str, str]
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

    The unified preprocessor has been split into two independent serialized
    artifacts to support Late Fusion training (Phase 5) and the Embedding
    Microservice (Phase 6). Both are fitted exclusively on the training set
    per the Anti-Skew Mandate (Rule 2.9).

    Attributes:
        root_dir: Root directory for feature engineering artifacts.
        input_data_path: Path to the input enriched dataset.
        train_data_path: Path to the output training features CSV.
        test_data_path: Path to the output test features CSV.
        val_data_path: Path to the output validation features CSV.
        structured_preprocessor_path: Path to the serialized structured
            (numeric + categorical) preprocessor pipeline.
        nlp_preprocessor_path: Path to the serialized NLP
            (TextEmbedder + PCA) preprocessor pipeline.
        embedding_model_name: Name of the sentence-transformer model.
        pca_components: Number of PCA components for NLP dimensionality reduction.
        test_size: Proportion of the dataset to include in the test split.
        val_size: Proportion of the dataset to include in the validation split.
        random_state: Seed for reproducibility.
        target_column: Name of the target variable column.
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
    """Configuration for the Late Fusion model training stage.

    Orchestrates three MLflow-tracked training runs: structured baseline
    (Branch 1), NLP baseline (Branch 2), and the stacked meta-learner
    (Late Fusion). All model artifacts are serialized to root_dir.

    Attributes:
        root_dir: Root directory for model training artifacts.
        train_data_path: Path to the training feature CSV.
        val_data_path: Path to the validation feature CSV.
        test_data_path: Path to the test feature CSV.
        structured_preprocessor_path: Path to the fitted structured preprocessor.
        nlp_preprocessor_path: Path to the fitted NLP preprocessor.
        structured_model_path: Output path for the serialized structured branch model.
        nlp_model_path: Output path for the serialized NLP branch model.
        meta_model_path: Output path for the serialized meta-learner.
        evaluation_report_path: Output path for the JSON evaluation report.
        target_column: Name of the target variable column.
        random_state: Seed for reproducibility across all branches.
        cv_folds: Number of cross-validation folds for OOF stacking.
        structured_n_trials: Number of Optuna trials for the structured branch.
        nlp_n_trials: Number of Optuna trials for the NLP branch.
        meta_C: Regularization strength for the Logistic Regression meta-learner.
        meta_max_iter: Maximum iterations for the Logistic Regression solver.
        mlflow_uri: MLflow tracking server URI.
        experiment_name: MLflow experiment name for all training runs.
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
    TotalCharges is typed as str | None because the raw CSV contains
    blank strings for customers with tenure=0.
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

    ticket_note: str | None = Field(
        None, description="AI-generated customer complaint or interaction note"
    )
    primary_sentiment_tag: str | None = Field(
        None, description="AI-classified sentiment of the customer"
    )
