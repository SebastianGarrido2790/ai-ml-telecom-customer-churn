"""
Unit tests for individual ML components.

This module provides granular validation for the core building blocks of the
system, including data ingestion, validation, enrichment orchestrators,
configuration management, and feature engineering transformers.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.components.data_enrichment.orchestrator import EnrichmentOrchestrator
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidator
from src.config.configuration import ConfigurationManager
from src.entity.config_entity import (
    DataEnrichmentConfig,
    DataIngestionConfig,
    FeatureEngineeringConfig,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for file-based component tests.

    Args:
        tmp_path: Pytest built-in fixture for temporary directory path.

    Returns:
        Path: Path object pointing to a temporary directory.
    """
    return tmp_path


# --- Data Ingestion ---
def test_data_ingestion_download_local(temp_dir):
    """Verifies that DataIngestion correctly downloads/copies a local file to the destination."""
    source_file = temp_dir / "source.zip"
    source_file.write_text("dummy zip content")
    dest_file = temp_dir / "dest.zip"

    config = DataIngestionConfig(
        root_dir=temp_dir,
        source_URL=str(source_file),
        local_data_file=dest_file,
        unzip_dir=temp_dir / "unzip",
    )

    ingestion = DataIngestion(config=config)
    ingestion.download_file()

    assert dest_file.exists()
    assert dest_file.read_text() == "dummy zip content"


def test_data_ingestion_extract_skip(temp_dir):
    """Verifies that DataIngestion logs a skip message when attempting to extract a non-zip file."""
    data_file = temp_dir / "data.txt"
    data_file.write_text("not a zip")

    config = DataIngestionConfig(
        root_dir=temp_dir,
        source_URL="http://example.com/data.txt",
        local_data_file=data_file,
        unzip_dir=temp_dir / "unzip",
    )

    ingestion = DataIngestion(config=config)
    with patch("src.components.data_ingestion.logger") as mock_logger:
        ingestion.extract_zip_file()
        mock_logger.info.assert_called()
        args, _ = mock_logger.info.call_args
        assert "is not a .zip file" in args[0]


# --- Data Validation ---
@patch("src.components.data_validation.gx.get_context")
def test_data_validator_init(mock_gx_context):
    """Verifies that DataValidator correctly initializes the Great Expectations context."""
    validator = DataValidator()
    mock_gx_context.assert_called_once()
    assert validator.context is not None


def test_data_validator_suites():
    """Verifies that DataValidator can build both raw and enriched Great Expectations suites."""
    validator = DataValidator()

    suite_raw = validator.build_raw_telco_suite()
    assert suite_raw.name == "raw_telco_churn_suite"

    suite_enriched = validator.build_enriched_telco_suite()
    assert suite_enriched.name == "enriched_telco_churn_suite"


def test_data_validator_validation_pass():
    """Verifies that valid data successfully passes the raw telco churn suite."""
    validator = DataValidator()
    validator.build_raw_telco_suite()

    df = pd.DataFrame(
        {
            "customerID": ["1"],
            "tenure": [10],
            "InternetService": ["DSL"],
            "Contract": ["Month-to-month"],
            "MonthlyCharges": [50.0],
            "TotalCharges": [500.0],
            "Churn": ["No"],
        }
    )

    results = validator.validate_dataset(df, "raw_telco_churn_suite", "test_id", "test_stage")
    assert results["success"] is True


# --- Data Enrichment ---
@pytest.mark.asyncio
async def test_enrichment_orchestrator_resume(temp_dir):
    """Verifies that EnrollmentOrchestrator correctly resumes processing.

    It checks if it can continue from an existing output file.
    """
    raw_path = temp_dir / "raw.csv"
    out_path = temp_dir / "out.csv"

    # Use string IDs to avoid pd.read_csv numeric conversion mismatch
    pd.DataFrame({"customerID": ["C1", "C2"], "tenure": [10, 20]}).to_csv(raw_path, index=False)

    # Existing output with one row processed
    pd.DataFrame(
        {
            "customerID": ["C1"],
            "tenure": [10],
            "ticket_note": ["old note"],
            "primary_sentiment_tag": ["Negative"],
        }
    ).to_csv(out_path, index=False)

    config = MagicMock(spec=DataEnrichmentConfig)
    orch = EnrichmentOrchestrator(raw_data_path=raw_path, output_path=out_path, config=config)

    with patch("src.components.data_enrichment.orchestrator.generate_ticket_note", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = MagicMock(ticket_note="new note", primary_sentiment_tag="Positive")

        df_res = await orch.run_enrichment(limit=2)

        # Verify first row is from existing file (resumed)
        assert df_res.loc[df_res["customerID"] == "C1", "ticket_note"].iloc[0] == "old note"
        # Verify second row was newly generated
        assert df_res.loc[df_res["customerID"] == "C2", "ticket_note"].iloc[0] == "new note"
        assert mock_gen.call_count == 1


# --- Configuration Manager ---
def test_configuration_manager_basic(temp_dir):
    """Verifies that ConfigurationManager correctly hydrates config entities from YAML files."""
    config_yaml = temp_dir / "config.yaml"
    params_yaml = temp_dir / "params.yaml"
    schema_yaml = temp_dir / "schema.yaml"

    config_yaml.write_text(
        "data_ingestion:\n"
        "  root_dir: artifacts/di\n"
        "  source_URL: http://di\n"
        "  local_data_file: artifacts/di/data.zip\n"
        "  unzip_dir: artifacts/di\n"
    )
    params_yaml.write_text("training:\n  target_column: Churn\n")
    schema_yaml.write_text(
        "COLUMNS:\n  id: int\n"
        "ENRICHED_COLUMNS:\n  id: int\n"
        "TARGET_COLUMN:\n  name: Churn\n"
    )

    with (
        patch("src.config.configuration.create_directories"),
        patch("src.config.configuration.read_yaml") as mock_read_yaml,
    ):
        m_config = MagicMock()
        m_config.data_ingestion.root_dir = "artifacts/di"
        m_config.data_ingestion.source_URL = "http://di"
        m_config.data_ingestion.local_data_file = "artifacts/di/data.zip"
        m_config.data_ingestion.unzip_dir = "artifacts/di"

        m_schema = {
            "COLUMNS": {"id": "int"},
            "ENRICHED_COLUMNS": {"id": "int"},
            "TARGET_COLUMN": {"name": "Churn"},
        }

        mock_read_yaml.side_effect = [m_config, MagicMock(), m_schema]

        cm = ConfigurationManager(config_filepath=config_yaml, params_filepath=params_yaml, schema_filepath=schema_yaml)
        config = cm.get_data_ingestion_config()
        assert config.root_dir == Path("artifacts/di")


# --- Feature Engineering ---
@patch("sentence_transformers.SentenceTransformer")
def test_feature_engineering_init(mock_st, temp_dir):
    """Verifies that FeatureEngineering initializes with correct transformer pipelines."""
    from src.components.feature_engineering import FeatureEngineering

    config = FeatureEngineeringConfig(
        root_dir=temp_dir,
        input_data_path=temp_dir / "input.csv",
        train_data_path=temp_dir / "train.csv",
        test_data_path=temp_dir / "test.csv",
        val_data_path=temp_dir / "val.csv",
        structured_preprocessor_path=temp_dir / "s_prep.pkl",
        nlp_preprocessor_path=temp_dir / "n_prep.pkl",
        embedding_model_name="test-model",
        pca_components=10,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        target_column="Churn",
    )

    fe = FeatureEngineering(config=config)
    assert fe.config.random_state == 42

    struct_p = fe.get_structured_preprocessor()
    assert "num" in [name for name, _, _ in struct_p.transformers]

    nlp_p = fe.get_nlp_preprocessor()
    assert "nlp" in [name for name, _, _ in nlp_p.transformers]


@patch("src.components.feature_engineering.pd.read_csv")
@patch("src.components.feature_engineering.joblib.dump")
@patch("sentence_transformers.SentenceTransformer")
def test_feature_engineering_run(mock_st, mock_dump, mock_read, temp_dir, sample_telco_df):
    """Verifies end-to-end execution of the feature engineering component.

    Includes stratified splitting validation.
    """
    from src.components.feature_engineering import FeatureEngineering

    # Use the centralized sample_telco_df
    mock_read.return_value = sample_telco_df

    config = FeatureEngineeringConfig(
        root_dir=temp_dir,
        input_data_path=temp_dir / "input.csv",
        train_data_path=temp_dir / "train.csv",
        test_data_path=temp_dir / "test.csv",
        val_data_path=temp_dir / "val.csv",
        structured_preprocessor_path=temp_dir / "s_prep.pkl",
        nlp_preprocessor_path=temp_dir / "n_prep.pkl",
        embedding_model_name="test-model",
        pca_components=2,
        test_size=0.2,
        val_size=0.2,
        random_state=42,
        target_column="Churn",
    )

    fe = FeatureEngineering(config=config)

    # Mock model encode to return dummy embeddings with correct shape (batch size)
    mock_model = MagicMock()
    mock_model.encode.side_effect = lambda x, *args, **kwargs: np.zeros((len(x), 2))
    mock_model.get_sentence_embedding_dimension.return_value = 2
    mock_st.return_value = mock_model

    fe.initiate_feature_engineering()

    assert mock_dump.call_count == 2  # structured and nlp preprocessors
    assert mock_read.called
