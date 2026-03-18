"""
Pipeline stage 00: Data Ingestion

Orchestrates the data ingestion component to fetch data from its source (URL or local path)
and store it in the artifacts structure.
"""

from src.components.data_ingestion import DataIngestion
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="stage_00_data_ingestion")


def main() -> None:
    """Executes the Data Ingestion pipeline stage."""
    try:
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()

        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.download_file()
        data_ingestion.extract_zip_file()

        logger.info("Data Ingestion Stage Completed Successfully.")

    except Exception as e:
        logger.error(f"Error occurred during Data Ingestion: {e}")
        raise e


if __name__ == "__main__":
    main()
