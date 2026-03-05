"""
Project-wide path constants.

Single source of truth for all configuration file paths, data directories,
and artifact locations. Every pipeline module should import paths from here
instead of hardcoding strings.
"""

from pathlib import Path

# --- Configuration Files ---
CONFIG_FILE_PATH = Path("config/config.yaml")
PARAMS_FILE_PATH = Path("config/params.yaml")
SCHEMA_FILE_PATH = Path("config/schema.yaml")

# --- Data Directories ---
RAW_DATA_DIR = Path("data/raw")
EXTERNAL_DATA_DIR = Path("data/external")

# --- Artifact Directories ---
ARTIFACTS_DIR = Path("artifacts")
MODELS_DIR = ARTIFACTS_DIR / "models"
GX_DIR = ARTIFACTS_DIR / "gx"
