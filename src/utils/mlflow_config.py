"""
Utility functions for MLflow configuration across modules.
Fully environment-aware, using ENV loaded from src.utils.paths.
"""

import os
import yaml
from pathlib import Path
from dotenv import load_dotenv
from src.utils.logger import get_logger
from src.constants import PARAMS_FILE_PATH

# Load environment variables from .env file
load_dotenv()

logger = get_logger(__name__)

ENV = os.getenv("ENV", "local").lower()  # e.g., "local", "staging", "production"


def get_mlflow_uri(params_path: Path = PARAMS_FILE_PATH) -> str:
    """
    Returns the MLflow Tracking URI with clear priority and automatic environment handling.
    Detects the appropriate MLflow URI based on the current environment (ENV),
    checks environment variables, and falls back to config/params.yaml.

    Priority:
        1. Environment variable MLFLOW_TRACKING_URI
        2. Environment-based defaults (production/staging/local)
        3. config/params.yaml (fallback for local mode)

    Args:
        params_path (Path): Path to params.yaml (default: config/params.yaml).

    Returns:
        str: MLflow Tracking URI.
    """

    # --- Priority 1: Environment variable ---
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
    if mlflow_uri:
        logger.info(f"[ENV={ENV}] Using MLflow URI from environment: {mlflow_uri}")
        return mlflow_uri

    # --- Priority 2: Environment-based defaults ---
    if ENV == "production":
        # In production, we MUST have a tracking URI
        raise RuntimeError("Production mode requires MLFLOW_TRACKING_URI to be set.")

    elif ENV == "staging":
        # Optional: Define a default staging server
        staging_uri = "http://staging-mlflow-server:5000"
        logger.info(f"[ENV={ENV}] Using staging MLflow URI: {staging_uri}")
        return staging_uri

    # --- Priority 3: YAML fallback (local mode) ---
    if params_file := Path(params_path):
        if params_file.exists():
            try:
                with open(params_file, "r") as f:
                    params = yaml.safe_load(f)
                    if params and "mlflow" in params and "uri" in params["mlflow"]:
                        uri = params["mlflow"]["uri"]
                        logger.info(
                            f"[ENV={ENV}] Using MLflow URI from {params_path}: {uri}"
                        )
                        return uri
            except Exception as e:
                logger.warning(f"Error reading {params_path}: {e}")

    # Fallback for local
    local_uri = "file:./mlruns"
    logger.info(f"[ENV={ENV}] Using fallback local MLflow URI: {local_uri}")
    return local_uri
