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

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from src.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH, SCHEMA_FILE_PATH
from src.entity.config_entity import (
    DataEnrichmentConfig,
    DataIngestionConfig,
    DataValidationConfig,
    EmbeddingServiceConfig,
    FeatureEngineeringConfig,
    ModelTrainingConfig,
    PredictionAPIConfig,
)
from src.utils.common import create_directories, read_yaml
from src.utils.exceptions import DataQualityContext, SchemaContractViolation
from src.utils.logger import get_logger


class _SchemaContract(BaseModel):
    """Pydantic contract for schema.yaml structural validation.

    Validates that the three mandatory top-level keys are present and
    that their column maps are non-empty dicts keyed by column name.

    Attributes:
        COLUMNS: Column-to-dtype map for the raw Telco dataset.
        ENRICHED_COLUMNS: Column-to-dtype map for the enriched dataset.
        TARGET_COLUMN: Mapping that must contain at least a ``name`` key.
    """

    COLUMNS: dict[str, Any]
    ENRICHED_COLUMNS: dict[str, Any]
    TARGET_COLUMN: dict[str, Any]

    @field_validator("COLUMNS", "ENRICHED_COLUMNS")
    @classmethod
    def must_be_non_empty(cls, v: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        """Ensures column maps contain at least one entry.

        Args:
            v: The value of the validated field.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If the column map is empty.
        """
        if not v:
            raise ValueError("Column map must contain at least one entry.")
        return v

    @field_validator("TARGET_COLUMN")
    @classmethod
    def must_have_name_key(cls, v: dict[str, Any]) -> dict[str, Any]:  # noqa: N805
        """Ensures TARGET_COLUMN contains the required 'name' key.

        Args:
            v: The value of the validated field.

        Returns:
            The validated value unchanged.

        Raises:
            ValueError: If 'name' key is absent.
        """
        if "name" not in v:
            raise ValueError("TARGET_COLUMN must contain a 'name' key.")
        return v


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

        # Validate schema.yaml structure at load time
        self._validate_schema(schema_filepath)

        # Ensure the top-level artifacts directory exists
        create_directories([Path("artifacts")])

    def _validate_schema(self, schema_filepath: Path) -> None:
        """Validates the loaded schema.yaml against _SchemaContract.

        Raises SchemaContractViolation immediately if the file is missing
        required top-level sections or contains empty column maps, preventing
        silent fallback to hardcoded column lists in downstream stages.

        Args:
            schema_filepath: Path to the schema.yaml file (used in error context).

        Raises:
            SchemaContractViolation: If schema.yaml fails structural validation.
        """
        raw: dict[str, Any] = dict(self.schema)  # ConfigBox → plain dict
        try:
            _SchemaContract(**raw)
            logger.info(f"schema.yaml structural validation passed: {schema_filepath}")
        except ValidationError as exc:
            # Flatten Pydantic error messages for the agent-readable context
            errors = [f"{e['loc']}: {e['msg']}" for e in exc.errors()]
            raise SchemaContractViolation(
                message=(f"schema.yaml failed structural validation: {schema_filepath}. Errors: {errors}"),
                context=DataQualityContext(
                    dataset_id="schema.yaml",
                    pipeline_stage="configuration",
                    column=None,
                    expectation="COLUMNS, ENRICHED_COLUMNS (non-empty), TARGET_COLUMN (with 'name' key) must all be present.",
                    actual_value=errors,
                    row_count_affected=0,
                    suggested_action="Fix schema.yaml to include all required top-level sections with valid column definitions.",
                ),
            ) from exc

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Returns the configuration for the data ingestion stage.

        Creates the ingestion root directory if it does not exist.

        Returns:
            DataIngestionConfig: Immutable config for data ingestion.
        """
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        """Returns the configuration for the data validation stage.

        Creates the validation root directory if it does not exist.

        Returns:
            DataValidationConfig: Immutable config for data validation.
        """
        config = self.config.data_validation
        schema = self.schema.COLUMNS

        create_directories([config.root_dir])

        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            STATUS_FILE=config.STATUS_FILE,
            unzip_data_dir=Path(config.unzip_data_dir),
            all_schema=schema,
        )

    def get_data_enrichment_config(self) -> DataEnrichmentConfig:
        """Returns the configuration for the data enrichment stage.

        Creates the enrichment root directory if it does not exist.

        Returns:
            DataEnrichmentConfig: Immutable config for data enrichment.
        """
        config = self.config.data_enrichment

        create_directories([config.root_dir])

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
            enrich_params.base_url if enrich_params and getattr(enrich_params, "base_url", None) is not None else None
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
            enrich_params.batch_size if enrich_params and getattr(enrich_params, "batch_size", None) is not None else 20
        )
        limit = enrich_params.limit if enrich_params and getattr(enrich_params, "limit", None) is not None else None

        # Convert limit=0 sentinel to None (process full dataset)
        if limit == 0:
            limit = None

        return DataEnrichmentConfig(
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

    def get_feature_engineering_config(self) -> FeatureEngineeringConfig:
        """Returns the configuration for the feature engineering stage.

        Creates the feature engineering root directory if it does not exist.
        The preprocessor is now split into two independent serialized artifacts:
        structured_preprocessor.pkl and nlp_preprocessor.pkl.

        Returns:
            FeatureEngineeringConfig: Immutable config for feature engineering.
        """
        config = self.config.feature_engineering
        params = self.params.feature_engineering

        create_directories([config.root_dir])

        return FeatureEngineeringConfig(
            root_dir=Path(config.root_dir),
            input_data_path=Path(config.input_data_path),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            val_data_path=Path(config.val_data_path),
            structured_preprocessor_path=Path(config.structured_preprocessor_path),
            nlp_preprocessor_path=Path(config.nlp_preprocessor_path),
            embedding_model_name=params.embedding_model_name,
            pca_components=params.pca_components,
            test_size=params.test_size,
            val_size=params.val_size,
            random_state=params.random_state,
            target_column=self.params.training.target_column,
        )

    def get_model_training_config(self) -> ModelTrainingConfig:
        """Returns the configuration for the Late Fusion model training stage.

        Creates the model training root directory if it does not exist.

        Returns:
            ModelTrainingConfig: Immutable config for model training.
        """
        feat_config = self.config.feature_engineering
        model_config = self.config.model_training
        params = self.params.model_training

        create_directories([model_config.root_dir])

        return ModelTrainingConfig(
            root_dir=Path(model_config.root_dir),
            train_data_path=Path(feat_config.train_data_path),
            val_data_path=Path(feat_config.val_data_path),
            test_data_path=Path(feat_config.test_data_path),
            structured_preprocessor_path=Path(feat_config.structured_preprocessor_path),
            nlp_preprocessor_path=Path(feat_config.nlp_preprocessor_path),
            structured_model_path=Path(model_config.structured_model_path),
            nlp_model_path=Path(model_config.nlp_model_path),
            meta_model_path=Path(model_config.meta_model_path),
            evaluation_report_path=Path(model_config.evaluation_report_path),
            target_column=self.params.training.target_column,
            random_state=params.random_state,
            cv_folds=params.cv_folds,
            structured_n_trials=params.structured_branch.n_trials,
            nlp_n_trials=params.nlp_branch.n_trials,
            meta_C=params.meta_learner.C,
            meta_max_iter=params.meta_learner.max_iter,
            mlflow_uri=self.params.mlflow.uri,
            experiment_name=self.params.mlflow.experiment_name,
        )

    def get_embedding_service_config(self) -> EmbeddingServiceConfig:
        """Returns the configuration for the Embedding Microservice.

        Reads artifact paths from the feature_engineering section (the preprocessor
        is a feature pipeline output) and service settings from the api section.

        Returns:
            EmbeddingServiceConfig: Immutable config for the embedding service.
        """
        api_cfg = self.config.api.embedding_service
        feat_cfg = self.config.feature_engineering
        fe_params = self.params.feature_engineering

        return EmbeddingServiceConfig(
            host=api_cfg.host,
            port=int(api_cfg.port),
            timeout_seconds=float(api_cfg.timeout_seconds),
            nlp_preprocessor_path=Path(feat_cfg.nlp_preprocessor_path),
            model_version=api_cfg.model_version,
            pca_components=int(fe_params.pca_components),
            api_key=os.environ.get("API_KEY", "dev-key-churn-2024"),
        )

    def get_prediction_api_config(self) -> PredictionAPIConfig:
        """Returns the configuration for the Prediction API microservice.

        Constructs the embedding service URL from the embedding service config
        section so the Prediction API can call it without hardcoding the address.

        Returns:
            PredictionAPIConfig: Immutable config for the prediction API.
        """
        api_cfg = self.config.api
        model_cfg = self.config.model_training
        fe_params = self.params.feature_engineering
        feat_cfg = self.config.feature_engineering

        # EMBEDDING_SERVICE_HOST overrides config.yaml when running inside
        # Docker Compose — container DNS name replaces 127.0.0.1.
        embed_host = os.environ.get(
            "EMBEDDING_SERVICE_HOST",
            api_cfg.embedding_service.host,
        )
        embed_port = int(api_cfg.embedding_service.port)
        embedding_service_url = f"http://{embed_host}:{embed_port}"

        return PredictionAPIConfig(
            host=api_cfg.prediction_api.host,
            port=int(api_cfg.prediction_api.port),
            structured_preprocessor_path=Path(feat_cfg.structured_preprocessor_path),
            structured_model_path=Path(model_cfg.structured_model_path),
            nlp_model_path=Path(model_cfg.nlp_model_path),
            meta_model_path=Path(model_cfg.meta_model_path),
            embedding_service_url=embedding_service_url,
            model_version=api_cfg.prediction_api.model_version,
            pca_components=int(fe_params.pca_components),
            api_key=os.environ.get("API_KEY", "dev-key-churn-2024"),
        )
