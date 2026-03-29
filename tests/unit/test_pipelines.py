"""
Unit tests for the end-to-end FTI pipeline stages.

This module validates the orchestration logic for each of the 6 pipeline stages,
ensuring that high-level pipeline entry points correctly trigger their
respective components and handle cross-stage dependencies via mocked
ConfigurationManager and component objects.

FTI Cycle Coverage:
    - Stage 00 (Ingestion) to Stage 05 (Model Training).
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# Mock problematic libraries before they are imported by pipeline stages
sys.modules["logfire"] = MagicMock()
sys.modules["mlflow"] = MagicMock()
sys.modules["mlflow.sklearn"] = MagicMock()
sys.modules["mlflow.xgboost"] = MagicMock()


@pytest.fixture
def mock_config_mgr():
    """Provides a mocked ConfigurationManager with Path-like objects."""
    with patch("src.config.configuration.ConfigurationManager") as m:
        mock_instance = m.return_value

        def create_path_mock(path_str):
            # A mock that looks like a path and supports pathlib operations
            pm = MagicMock(spec=Path)
            pm.exists.return_value = True
            pm.__str__.return_value = path_str
            pm.__fspath__.return_value = path_str
            # Return fresh mocks for new paths to avoid shared state issues
            pm.__truediv__.side_effect = lambda x: create_path_mock(f"{path_str}/{x}")
            return pm

        mock_instance.get_data_ingestion_config.return_value.local_data_file = create_path_mock("data/raw.zip")
        mock_instance.get_data_ingestion_config.return_value.unzip_dir = create_path_mock("data/")

        mock_instance.get_data_validation_config.return_value.unzip_data_dir = create_path_mock("data/raw")
        mock_instance.get_data_validation_config.return_value.STATUS_FILE = "reports/status.txt"
        mock_instance.get_data_validation_config.return_value.root_dir = create_path_mock("artifacts/val")
        mock_instance.get_data_validation_config.return_value.all_schema = {}

        mock_instance.get_data_enrichment_config.return_value.raw_data_path = create_path_mock("data/raw/data.csv")
        mock_instance.get_data_enrichment_config.return_value.enriched_data_file = create_path_mock("data/enriched.csv")
        mock_instance.get_data_enrichment_config.return_value.limit = 10
        mock_instance.get_data_enrichment_config.return_value.batch_size = 5
        mock_instance.get_data_enrichment_config.return_value.model_name = "test-model"

        mock_instance.get_feature_engineering_config.return_value.raw_data_path = create_path_mock("data/enriched.csv")

        mock_instance.get_model_training_config.return_value.trained_model_path = create_path_mock("models/fusion")

        yield m


# --- Stage 00: Data Ingestion ---
def test_stage_00_data_ingestion(mock_config_mgr):
    """Verifies that Stage 00 (Data Ingestion) correctly triggers the component."""
    with patch("src.pipeline.stage_00_data_ingestion.DataIngestion") as mock_comp:
        from src.pipeline.stage_00_data_ingestion import main

        main()
        mock_comp.assert_called_once()


# --- Stage 01: Data Validation ---
def test_stage_01_data_validation(mock_config_mgr):
    """Verifies that Stage 01 (Data Validation) correctly triggers the component and returns status."""
    with (
        patch("src.pipeline.stage_01_data_validation.DataValidator") as mock_comp_cls,
        patch("pandas.read_csv") as mock_read_csv,
        patch("src.pipeline.stage_01_data_validation.Path"),
    ):
        mock_read_csv.return_value = pd.DataFrame({"a": [1]})
        mock_comp = mock_comp_cls.return_value
        mock_comp.validate_dataset.return_value = {"status": "SUCCESS", "results": []}

        from src.pipeline.stage_01_data_validation import main

        main()
        mock_comp.validate_dataset.assert_called_once()


# --- Stage 02: Data Enrichment ---
@pytest.mark.asyncio
async def test_stage_02_data_enrichment(mock_config_mgr):
    """Verifies that Stage 02 (Data Enrichment) triggers the orchestrator asynchronously."""
    with patch("src.components.data_enrichment.orchestrator.EnrichmentOrchestrator") as mock_orch_cls:
        mock_instance = mock_orch_cls.return_value
        mock_instance.run_enrichment = AsyncMock()

        from src.pipeline.stage_02_data_enrichment import main

        await main()
        mock_instance.run_enrichment.assert_called()


# --- Stage 03: Enriched Validation ---
def test_stage_03_enriched_validation(mock_config_mgr):
    """Verifies that Stage 03 (Enriched Validation) correctly triggers the validation component."""
    # Stage 03 uses Path from data_enrichment_config (already mocked)
    with (
        patch("src.pipeline.stage_03_enriched_validation.DataValidator") as mock_comp_cls,
        patch("pandas.read_csv") as mock_read_csv,
    ):
        mock_read_csv.return_value = pd.DataFrame({"a": [1]})
        mock_comp = mock_comp_cls.return_value
        mock_comp.validate_dataset.return_value = {"status": "SUCCESS", "results": []}

        from src.pipeline.stage_03_enriched_validation import main

        main()
        mock_comp.validate_dataset.assert_called_once()


# --- Stage 04: Feature Engineering ---
def test_stage_04_feature_engineering(mock_config_mgr):
    """Verifies that Stage 04 (Feature Engineering) triggers the training pipeline object."""
    with patch("src.pipeline.stage_04_feature_engineering.FeatureEngineering") as mock_comp:
        from src.pipeline.stage_04_feature_engineering import FeatureEngineeringTrainingPipeline

        obj = FeatureEngineeringTrainingPipeline()
        obj.main()
        mock_comp.assert_called_once()


# --- Stage 05: Model Training ---
def test_stage_05_model_training(mock_config_mgr):
    """Verifies that Stage 05 (Model Training) triggers both trainer and evaluator components."""
    # Ensure modules are loaded so they are available as attributes to their parent packages

    with (
        patch("src.components.model_training.trainer.LateFusionTrainer") as mock_trainer,
        patch("src.components.model_training.evaluator.LateFusionEvaluator") as mock_evaluator,
    ):
        mock_evaluator.return_value.evaluate.return_value = {
            "structured_baseline": {"metrics": {"recall": 0.5}},
            "late_fusion_stacked": {
                "metrics": {"recall": 0.6},
                "recall_lift": 0.1,
                "f1_lift": 0.05,
            },
        }
        from src.pipeline.stage_05_model_training import main

        main()
        mock_trainer.assert_called_once()
        mock_evaluator.assert_called_once()
