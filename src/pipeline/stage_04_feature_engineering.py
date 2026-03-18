"""
Phase 4: Feature Engineering Training Pipeline Stage.

This module orchestrates the execution of the Feature Engineering component,
handling data ingestion, transformation, and serialization of the preprocessor.
"""

from src.components.feature_engineering import FeatureEngineering
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

logger = get_logger(__name__)

STAGE_NAME = "Feature Engineering Stage"


class FeatureEngineeringTrainingPipeline:
    """Pipeline orchestrator for the Feature Engineering stage."""

    def __init__(self) -> None:
        """Initializes the preprocessing pipeline."""
        pass

    def main(self) -> None:
        """Executes the feature engineering pipeline logic."""
        config_manager = ConfigurationManager()
        feature_config = config_manager.get_feature_engineering_config()

        feature_engineering = FeatureEngineering(config=feature_config)
        feature_engineering.initiate_feature_engineering()


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = FeatureEngineeringTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
