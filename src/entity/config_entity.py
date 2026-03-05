"""
Data contract definitions for the Telecom Customer Churn system.

This module defines:
1. Frozen dataclass entities for pipeline stage configurations (immutable configs).
2. Pydantic models for strict row-level validation of the raw Telco dataset
   and its enriched variant.

These contracts enforce the 'Antigravity' standard: no untyped dictionaries
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
        enriched_data_file: Output path for the enriched dataset.
        prompts_dir: Directory containing prompt templates.
    """

    root_dir: Path
    enriched_data_file: Path
    prompts_dir: Path


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
    AI-generated ticket note field used for NLP-based feature engineering.
    """

    ticket_notes: str | None = Field(
        None, description="AI-generated customer complaint or interaction note"
    )
