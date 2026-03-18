"""
Data Ingestion Component.

Fetches raw data from external sources or local paths and stores them
in the project's artifact directory.
"""

import shutil
import urllib.request
import zipfile

from src.entity.config_entity import DataIngestionConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataIngestion:
    """Component responsible for fetching raw data from a source URL or local path.

    Attributes:
        config (DataIngestionConfig): Configuration for data ingestion.
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        """Initializes the DataIngestion component.

        Args:
            config (DataIngestionConfig): Ingestion configuration entity.
        """
        self.config = config

    def download_file(self) -> None:
        """Downloads the data file from source_URL into local_data_file.

        Supports both HTTP/HTTPS and local file paths. If a local path is
        provided, it copies the file instead of downloading.

        Raises:
            FileNotFoundError: If the local source file is not found.
            Exception: For other download or copy errors.
        """
        source = self.config.source_URL
        destination = self.config.local_data_file

        if not destination.exists():
            logger.info(f"Downloading data from {source} into {destination}...")
            if source.startswith("http://") or source.startswith("https://"):
                try:
                    urllib.request.urlretrieve(source, destination)
                    logger.info(f"Successfully downloaded file to: {destination}")
                except Exception as e:
                    logger.error(f"Failed to download file from URL {source}: {e}")
                    raise e
            else:
                # Assume it's a local file path
                try:
                    logger.info("Source is a local path, copying instead of making HTTP request...")
                    shutil.copy2(source, destination)
                    logger.info(f"Successfully copied file to: {destination}")
                except FileNotFoundError as e:
                    logger.error(f"Local file not found at {source}. Ensure the file exists.")
                    raise e
                except Exception as e:
                    logger.error(f"Failed to copy local file: {e}")
                    raise e
        else:
            logger.info(f"File already exists at {destination}. Skipping download.")

    def extract_zip_file(self) -> None:
        """Extracts the zip file if the downloaded file is a zip archive.

        Checks if the local_data_file has a '.zip' extension and extracts
        it into the configured unzip directory.
        """
        unzip_path = self.config.unzip_dir
        unzip_path.mkdir(parents=True, exist_ok=True)

        # Check if local_data_file is actually a zip file
        if str(self.config.local_data_file).endswith(".zip"):
            logger.info(f"Extracting zip file {self.config.local_data_file} into {unzip_path}...")
            with zipfile.ZipFile(self.config.local_data_file, "r") as zip_ref:
                zip_ref.extractall(unzip_path)
            logger.info("Extraction complete.")
        else:
            logger.info(
                f"Downloaded file {self.config.local_data_file} is not a .zip file. "
                "Skipping extraction."
            )
