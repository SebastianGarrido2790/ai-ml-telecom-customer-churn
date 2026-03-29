"""
Unit tests for the Data Ingestion component.

This module ensures that raw data is correctly retrieved from both local
filesystem paths and remote HTTP/HTTPS URLs, handled via shutil and urllib.
It also validates idempotent behavior (skipping existing files).

Key components:
    - download_file: URI-aware retrieval logic.
    - unzip_file: Compression management (if applicable).
"""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.components.data_ingestion import DataIngestion
from src.entity.config_entity import DataIngestionConfig


@pytest.fixture
def data_ingestion_config(tmp_path: Path) -> DataIngestionConfig:
    """Provides a mocked DataIngestionConfig for testing.

    Args:
        tmp_path (Path): Pytest temporary directory fixture.

    Returns:
        DataIngestionConfig: A configuration pointing to temporary test paths.
    """
    return DataIngestionConfig(
        root_dir=tmp_path / "artifacts" / "data_ingestion",
        source_URL="data/raw/data.csv",
        local_data_file=tmp_path / "artifacts" / "data_ingestion" / "data.csv",
        unzip_dir=tmp_path / "artifacts" / "data_ingestion",
    )


@patch("src.components.data_ingestion.shutil.copy2")
@patch("src.components.data_ingestion.urllib.request.urlretrieve")
def test_download_file_local_path(
    mock_urlretrieve: patch,
    mock_copy2: patch,
    data_ingestion_config: DataIngestionConfig,
    tmp_path: Path,
) -> None:
    """Verifies that download_file copies local files when given a local path."""
    # Ensure source file is treated as non-existent to trigger copy exception handling,
    # but since we mocked copy2, we don't need the file to actually exist.
    ingestion = DataIngestion(config=data_ingestion_config)
    ingestion.download_file()

    mock_copy2.assert_called_once_with(data_ingestion_config.source_URL, data_ingestion_config.local_data_file)
    mock_urlretrieve.assert_not_called()


@patch("src.components.data_ingestion.shutil.copy2")
@patch("src.components.data_ingestion.urllib.request.urlretrieve")
def test_download_file_http_url(
    mock_urlretrieve: patch, mock_copy2: patch, data_ingestion_config: DataIngestionConfig
) -> None:
    """Verifies that download_file uses urlretrieve when given an HTTP/HTTPS URL."""
    # Change config to HTTP URL
    http_config = DataIngestionConfig(
        root_dir=data_ingestion_config.root_dir,
        source_URL="https://example.com/data.csv",
        local_data_file=data_ingestion_config.local_data_file,
        unzip_dir=data_ingestion_config.unzip_dir,
    )

    ingestion = DataIngestion(config=http_config)
    ingestion.download_file()

    mock_urlretrieve.assert_called_once_with(http_config.source_URL, http_config.local_data_file)
    mock_copy2.assert_not_called()


def test_download_file_already_exists(data_ingestion_config: DataIngestionConfig) -> None:
    """Verifies that download_file skips execution if the file already exists."""
    # Create the destination file
    data_ingestion_config.local_data_file.parent.mkdir(parents=True, exist_ok=True)
    with data_ingestion_config.local_data_file.open("w") as f:
        f.write("test")

    ingestion = DataIngestion(config=data_ingestion_config)

    with patch("src.components.data_ingestion.shutil.copy2") as mock_copy2:
        ingestion.download_file()
        mock_copy2.assert_not_called()
