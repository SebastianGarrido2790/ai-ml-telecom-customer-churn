"""
Configuration Manager for the Telecom Customer Churn system.

Central orchestrator that reads YAML configuration files and hydrates
immutable dataclass entities. This is the single entry point for any
pipeline stage to obtain its configuration — no stage reads YAML directly.

Usage:
    from src.config.configuration import ConfigurationManager
    config_mgr = ConfigurationManager()
    ingestion_config = config_mgr.get_data_ingestion_config()
"""

from pathlib import Path

from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import (
    DataEnrichmentConfig,
    DataIngestionConfig,
    DataValidationConfig,
)
from src.utils.common import create_directories, read_yaml
from src.utils.logger import get_logger

logger = get_logger(__name__)


class ConfigurationManager:
    """Reads YAML configs and produces typed, immutable pipeline configs.

    Attributes:
        config: Dot-accessible config.yaml content.
        params: Dot-accessible params.yaml content.
        schema: Dot-accessible schema.yaml content.
    """

    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH,
        schema_filepath: Path = SCHEMA_FILE_PATH,
    ) -> None:
        """Initializes the ConfigurationManager.

        Args:
            config_filepath: Path to config.yaml (system paths).
            params_filepath: Path to params.yaml (hyperparameters).
            schema_filepath: Path to schema.yaml (data schema).
        """
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        self.schema = read_yaml(schema_filepath)

        # Ensure the top-level artifacts directory exists
        create_directories([Path("artifacts")])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Returns the configuration for the data ingestion stage.

        Creates the ingestion root directory if it does not exist.

        Returns:
            DataIngestionConfig: Immutable config for data ingestion.
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """Returns the configuration for the data validation stage.

        Creates the validation root directory if it does not exist.

        Returns:
            DataValidationConfig: Immutable config for data validation.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=Path(config.unzip_data_dir),
            all_schema=schema,
        )

        return data_validation_config

    def get_data_enrichment_config(self) -> DataEnrichmentConfig:
        """Returns the configuration for the data enrichment stage.

        Creates the enrichment root directory if it does not exist.

        Returns:
            DataEnrichmentConfig: Immutable config for data enrichment.
        """
        config = self.config.data_enrichment

        create_directories([config.root_dir])

        # Pull from params.yaml with fallback
        enrich_params = getattr(self.params, "enrichment", None)

        model_name = (
            enrich_params.model_name
            if enrich_params and "model_name" in enrich_params
            else "gemini-1.5-flash"
        )
        batch_size = (
            enrich_params.batch_size
            if enrich_params and "batch_size" in enrich_params
            else 20
        )
        limit = (
            enrich_params.limit
            if enrich_params and "limit" in enrich_params
            else None
        )

        # Convert limit 0 to None (full dataset)
        if limit == 0:
            limit = None

        data_enrichment_config = DataEnrichmentConfig(
            root_dir=Path(config.root_dir),
            raw_data_path=Path(config.raw_data_path),
            enriched_data_file=Path(config.enriched_data_file),
            prompts_dir=Path(config.prompts_dir),
            all_schema=self.schema.ENRICHED_COLUMNS,
            model_name=model_name,
            limit=limit,
            batch_size=batch_size,
        )

        return data_enrichment_config
