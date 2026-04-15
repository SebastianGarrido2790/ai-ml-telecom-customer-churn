"""
Unit tests for the ConfigurationManager and environment hydration logic.

This module validates that the ConfigurationManager correctly parses YAML files
and hydrates the frozen Pydantic dataclasses (Config Entities) used across the FTI pipelines.
"""

from src.config.configuration import ConfigurationManager


def test_config_manager_init(temp_config_files):
    """Verifies that ConfigurationManager correctly loads the main configuration object."""
    config_path, params_path, schema_path = temp_config_files
    cm = ConfigurationManager(config_path, params_path, schema_path)
    assert cm.config.data_ingestion.root_dir == "artifacts/data_ingestion"


def test_get_data_ingestion_config(temp_config_files):
    """Verifies that the data ingestion configuration entity is correctly hydrated."""
    config_path, params_path, schema_path = temp_config_files
    cm = ConfigurationManager(config_path, params_path, schema_path)
    config = cm.get_data_ingestion_config()
    assert str(config.root_dir).replace("\\", "/") == "artifacts/data_ingestion"


def test_get_data_validation_config(temp_config_files):
    """Verifies that the data validation configuration entity is correctly hydrated."""
    config_path, params_path, schema_path = temp_config_files
    cm = ConfigurationManager(config_path, params_path, schema_path)
    config = cm.get_data_validation_config()
    assert str(config.unzip_data_dir).replace("\\", "/") == "artifacts/data_ingestion/unzipped_data"


def test_get_data_enrichment_config(temp_config_files):
    """Verifies that the data enrichment configuration entity includes params and schema."""
    config_path, params_path, schema_path = temp_config_files
    cm = ConfigurationManager(config_path, params_path, schema_path)
    config = cm.get_data_enrichment_config()
    assert config.model_name == "gemini"
    assert "ticket_note" in config.all_schema
