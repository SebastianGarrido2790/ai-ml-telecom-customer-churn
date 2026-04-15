"""
Unit test conftest.py for centralized configuration and manager fixtures.
"""

import pytest
import yaml

from src.config.configuration import ConfigurationManager


@pytest.fixture
def temp_config_files(tmp_path):
    """Provides temporary configuration, params, and schema files for testing."""
    config_data = {
        "data_ingestion": {
            "root_dir": "artifacts/data_ingestion",
            "source_URL": "http://test.com/data",
            "local_data_file": "artifacts/data_ingestion/data.csv",
            "unzip_dir": "artifacts/data_ingestion",
        },
        "data_validation": {
            "root_dir": "artifacts/data_validation",
            "STATUS_FILE": "artifacts/data_validation/status.txt",
            "unzip_data_dir": "artifacts/data_ingestion/unzipped_data",
        },
        "data_enrichment": {
            "root_dir": "artifacts/data_enrichment",
            "raw_data_path": "artifacts/data_ingestion/data.csv",
            "enriched_data_file": "artifacts/data_enrichment/enriched.csv",
            "prompts_dir": "src/enrichment",
        },
        "feature_engineering": {
            "root_dir": "artifacts/feature_engineering",
            "input_data_path": "artifacts/data_enrichment/enriched.csv",
            "train_data_path": "artifacts/feature_engineering/train.csv",
            "test_data_path": "artifacts/feature_engineering/test.csv",
            "val_data_path": "artifacts/feature_engineering/val.csv",
            "structured_preprocessor_path": "artifacts/feature_engineering/structured.pkl",
            "nlp_preprocessor_path": "artifacts/feature_engineering/nlp.pkl",
        },
        "model_training": {
            "root_dir": "artifacts/model_training",
            "structured_model_path": "artifacts/model_training/structured.pkl",
            "nlp_model_path": "artifacts/model_training/nlp.pkl",
            "meta_model_path": "artifacts/model_training/meta.pkl",
            "evaluation_report_path": "artifacts/model_training/report.json",
        },
    }

    params_data = {
        "training": {"test_size": 0.2, "random_state": 42, "target_column": "Churn"},
        "enrichment": {
            "model_name": "gemini",
            "batch_size": 1,
            "limit": 0,
            "model_provider": "google",
            "base_url": None,
        },
        "feature_engineering": {
            "embedding_model_name": "all-MiniLM-L6-v2",
            "pca_components": 20,
            "test_size": 0.15,
            "val_size": 0.15,
            "random_state": 42,
        },
        "model_training": {
            "random_state": 42,
            "cv_folds": 5,
            "structured_branch": {"algorithm": "xgboost", "n_trials": 10},
            "nlp_branch": {"algorithm": "xgboost", "n_trials": 10},
            "meta_learner": {"algorithm": "logistic_regression", "C": 1.0, "max_iter": 100},
        },
    }

    config_file = tmp_path / "config.yaml"
    params_file = tmp_path / "params.yaml"
    schema_file = tmp_path / "schema.yaml"

    with config_file.open("w") as f:
        yaml.dump(config_data, f)
    with params_file.open("w") as f:
        yaml.dump(params_data, f)
    with schema_file.open("w") as f:
        yaml.dump(
            {
                "COLUMNS": {"Churn": "int"},
                "ENRICHED_COLUMNS": {"customerID": "str", "ticket_note": "str"},
            },
            f,
        )

    return config_file, params_file, schema_file


@pytest.fixture
def mock_config_manager(temp_config_files):
    """Provides a ConfigurationManager instance pre-loaded with temporary test files."""
    config_path, params_path, schema_path = temp_config_files
    return ConfigurationManager(config_path, params_path, schema_path)
