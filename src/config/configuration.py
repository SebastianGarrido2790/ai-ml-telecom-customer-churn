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
    FeatureEngineeringConfig,
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

        model_provider = (
            enrich_params.model_provider
            if enrich_params and getattr(enrich_params, "model_provider", None) is not None
            else "google"
        )
        model_name = (
            enrich_params.model_name
            if enrich_params and getattr(enrich_params, "model_name", None) is not None
            else "gemini-2.0-flash"
        )
        base_url = (
            enrich_params.base_url
            if enrich_params and getattr(enrich_params, "base_url", None) is not None
            else None
        )
        secondary_model_name = (
            enrich_params.secondary_model_name
            if enrich_params and getattr(enrich_params, "secondary_model_name", None) is not None
            else None
        )
        secondary_base_url = (
            enrich_params.secondary_base_url
            if enrich_params and getattr(enrich_params, "secondary_base_url", None) is not None
            else None
        )
        batch_size = (
            enrich_params.batch_size
            if enrich_params and getattr(enrich_params, "batch_size", None) is not None
            else 20
        )
        limit = (
            enrich_params.limit
            if enrich_params and getattr(enrich_params, "limit", None) is not None
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
            model_provider=model_provider,
            model_name=model_name,
            base_url=base_url,
            secondary_model_name=secondary_model_name,
            secondary_base_url=secondary_base_url,
            limit=limit,
            batch_size=batch_size,
        )

        return data_enrichment_config

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        """Returns the configuration for the feature engineering stage.

        Creates the feature engineering root directory if it does not exist.

        Returns:
            FeatureEngineeringConfig: Immutable config for feature engineering.
        """
        config = self.config.feature_engineering
        params = self.params.feature_engineering

        create_directories([config.root_dir])

        feature_engineering_config = FeatureEngineeringConfig(
            root_dir=Path(config.root_dir),
            input_data_path=Path(config.input_data_path),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            val_data_path=Path(config.val_data_path),
            preprocessor_path=Path(config.preprocessor_path),
            embedding_model_name=params.embedding_model_name,
            pca_components=params.pca_components,
            test_size=params.test_size,
            val_size=params.val_size,
            random_state=params.random_state,
            target_column=self.params.training.target_column,
        )

        return feature_engineering_config
