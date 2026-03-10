"""
Common utility functions for the Telecom Customer Churn system.

This module provides reusable helper functions for handling YAML configuration,
directory management, JSON serialization, and file metadata, among others.
These utilities ensure consistency across different pipeline stages.
"""

import json
from pathlib import Path
from typing import Any

import yaml
from box import ConfigBox
from box.exceptions import BoxValueError

from src.utils.logger import get_logger

logger = get_logger(__name__)


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """Reads a YAML file and returns its content as a dot-accessible ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file.

    Returns:
        ConfigBox: Dot-accessible configuration object.

    Raises:
        ValueError: If the YAML file is empty.
        FileNotFoundError: If the file does not exist.
    """
    try:
        with open(path_to_yaml, encoding="utf-8") as yaml_file:
            content = yaml.safe_load(yaml_file)
            if content is None:
                raise ValueError(f"YAML file is empty: {path_to_yaml}")
            logger.info(f"YAML file loaded successfully: {path_to_yaml}")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError(f"YAML file is empty: {path_to_yaml}") from None


def create_directories(path_to_directories: list[Any], verbose: bool = True) -> None:
    """Creates a list of directories if they do not already exist.

    Args:
        path_to_directories (list[Any]): List of directory paths to create.
        verbose (bool): If True, logs each directory creation. Defaults to True.
    """
    for path in path_to_directories:
        Path(path).mkdir(parents=True, exist_ok=True)
        if verbose:
            logger.info(f"Created directory: {path}")


def save_json(path: Path, data: dict[str, Any]) -> None:
    """Saves a dictionary as a JSON file.

    Args:
        path (Path): Destination file path.
        data (dict[str, Any]): Data to serialize.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON file saved at: {path}")


def load_json(path: Path) -> ConfigBox:
    """Loads a JSON file and returns its content as a dot-accessible ConfigBox.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        ConfigBox: Dot-accessible data object.
    """
    with open(path, encoding="utf-8") as f:
        content = json.load(f)
    logger.info(f"JSON file loaded successfully: {path}")
    return ConfigBox(content)


def get_size(path: Path) -> str:
    """Returns the file size in a human-readable format (KB).

    Args:
        path (Path): Path to the file.

    Returns:
        str: File size string, e.g. '~ 1234 KB'.
    """
    size_in_kb = round(path.stat().st_size / 1024)
    return f"~ {size_in_kb} KB"
