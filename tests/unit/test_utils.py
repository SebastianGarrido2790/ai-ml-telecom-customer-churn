"""
Unit tests for core system utilities.

This module provides exhaustive coverage for the project's utility layer,
including YAML/JSON file handlers, centralized logging, MLflow environment
configuration, custom Agentic Data Science exceptions, and sklearn-compatible
transformers.

Key component tests:
    - common.py: read_yaml, create_directories, save_json.
    - mlflow_config.py: Environment-aware MLflow URI routing.
    - feature_utils.py: TextEmbedder and NumericCleaner implementation.
"""

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from box import ConfigBox

from src.utils.common import create_directories, get_size, load_json, read_yaml, save_json
from src.utils.exceptions import (
    CustomException,
    DataQualityContext,
    DataQualityError,
)
from src.utils.feature_utils import NumericCleaner, TextEmbedder
from src.utils.mlflow_config import get_mlflow_uri


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for file-based tests.

    Args:
        tmp_path: Pytest built-in fixture for temporary directory path.

    Returns:
        Path: Path object pointing to a temporary directory.
    """
    return tmp_path


# --- Common Utils ---
def test_read_yaml(temp_dir):
    """Verifies that a valid YAML file can be read into a ConfigBox object."""
    yaml_path = temp_dir / "test.yaml"
    data = {"key": "value"}
    yaml_path.write_text(yaml.dump(data))

    config = read_yaml(yaml_path)
    assert isinstance(config, ConfigBox)
    assert config.key == "value"


def test_read_yaml_empty(temp_dir):
    """Verifies that reading an empty YAML file raises a ValueError."""
    yaml_path = temp_dir / "empty.yaml"
    yaml_path.write_text("")
    with pytest.raises(ValueError, match="YAML file is empty"):
        read_yaml(yaml_path)


def test_create_directories(temp_dir):
    """Verifies that multiple directory levels can be created successfully."""
    paths = [temp_dir / "dir1", temp_dir / "dir2/subdir"]
    create_directories(paths)
    for p in paths:
        assert Path(p).exists()


def test_save_load_json(temp_dir):
    """Verifies that data can be saved to and loaded from a JSON file."""
    json_path = temp_dir / "test.json"
    data = {"name": "test"}
    save_json(json_path, data)
    assert json_path.exists()

    loaded = load_json(json_path)
    assert loaded.name == "test"


def test_get_size(temp_dir):
    """Verifies that the file size utility returns correct human-readable strings."""
    file_path = temp_dir / "large.txt"
    file_path.write_text(" " * 1024)  # 1 KB
    size_str = get_size(file_path)
    assert "1 KB" in size_str


# --- Exceptions ---
def test_custom_exception():
    """Verifies that CustomException correctly captures traceback and error messages."""
    try:
        raise ValueError("test error")
    except ValueError as e:
        ce = CustomException(e, sys)
        assert "test error" in str(ce)
        assert "line number" in str(ce)


def test_data_quality_error():
    """Verifies that DataQualityError correctly formats context for the AI agent."""
    ctx = DataQualityContext(
        dataset_id="test_ds",
        pipeline_stage="ingestion",
        column="col1",
        expectation="not null",
        actual_value=None,
        row_count_affected=10,
        suggested_action="check source",
    )
    err = DataQualityError("Validation failed", ctx)
    agent_ctx = err.to_agent_context()
    assert "Stage: ingestion" in agent_ctx
    assert "Column: col1" in agent_ctx


# --- Feature Utils ---
def test_numeric_cleaner():
    """Verifies that NumericCleaner correctly coerces strings and handles missing values."""
    import numpy as np
    import pandas as pd

    cleaner = NumericCleaner()
    df = pd.DataFrame({"col": ["1.0", " ", "3.5"]})
    transformed = cleaner.transform(df)
    assert np.isnan(transformed["col"].iloc[1])
    assert transformed["col"].dtype == np.float64


@patch("sentence_transformers.SentenceTransformer")
def test_text_embedder(mock_st):
    """Verifies TextEmbedder initialization, transformation, and pickling safety."""
    import numpy as np

    mock_model = MagicMock()
    mock_model.encode.return_value = np.array([[0.1, 0.2]])
    mock_model.get_sentence_embedding_dimension.return_value = 2
    mock_st.return_value = mock_model

    embedder = TextEmbedder(model_name="test")
    X = ["hello world"]

    # Test fit_transform
    res = embedder.fit_transform(X)
    assert res.shape == (1, 2)
    assert res[0, 0] == 0.1

    # Test feature names
    names = embedder.get_feature_names_out(["text"])
    assert names[0] == "text_0"

    # Test pickling safety (__getstate__)
    state = embedder.__getstate__()
    assert state["_model"] is None


# --- MLflow Config ---
def test_get_mlflow_uri_env_var(temp_dir):
    """Verifies that MLFLOW_TRACKING_URI environment variable has the highest priority."""
    # Priority 1: Direct env var
    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "http://env-uri"}):
        uri = get_mlflow_uri(temp_dir / "params.yaml")
        assert uri == "http://env-uri"


def test_get_mlflow_uri_staging(temp_dir):
    """Verifies that 'staging' environment defaults correctly when no env var is set."""
    # Priority 2: Staging default
    with (
        patch("src.utils.mlflow_config.ENV", "staging"),
        patch.dict(os.environ, {}, clear=True),
    ):
        uri = get_mlflow_uri(temp_dir / "params.yaml")
        assert "staging" in uri


def test_get_mlflow_uri_yaml(temp_dir):
    """Verifies that YAML configuration is used as a fallback for local environment."""
    # Priority 3: YAML fallback
    with (
        patch("src.utils.mlflow_config.ENV", "local"),
        patch.dict(os.environ, {}, clear=True),
        patch("pathlib.Path.exists") as mock_exists,
        patch("src.utils.mlflow_config.yaml.safe_load") as mock_yaml,
        patch("pathlib.Path.open"),
    ):
        mock_exists.return_value = True
        mock_yaml.return_value = {"mlflow": {"uri": "http://yaml-uri"}}
        uri = get_mlflow_uri(temp_dir / "params.yaml")
        assert uri == "http://yaml-uri"


def test_get_mlflow_uri_fallback(temp_dir):
    """Verifies that the default mlruns directory is used as final fallback."""
    # Final fallback
    with (
        patch("src.utils.mlflow_config.ENV", "local"),
        patch.dict(os.environ, {}, clear=True),
        patch("pathlib.Path.exists") as mock_exists,
    ):
        mock_exists.return_value = False
        uri = get_mlflow_uri(temp_dir / "params.yaml")
        assert "mlruns" in uri
