"""
Project-wide path constants.

Single source of truth for all configuration file paths, data directories,
and artifact locations. Every pipeline module should import paths from here
instead of hardcoding strings.
"""

from pathlib import Path

# --- Project Root ---
# Automatically finds the top-level directory (the one containing 'src/')
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- Configuration & Schema ---
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE_PATH = CONFIG_DIR / "config.yaml"
PARAMS_FILE_PATH = CONFIG_DIR / "params.yaml"
SCHEMA_FILE_PATH = CONFIG_DIR / "schema.yaml"

# --- Model & Reports ---
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DOCS_DIR = REPORTS_DIR / "docs"

# --- Logs & MLflow ---
LOGS_DIR = PROJECT_ROOT / "logs"

# --- Data Directories ---
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
EXTERNAL_DATA_DIR = PROJECT_ROOT / "data" / "external"

# --- Artifact Directories ---
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# --- Ensure directories exist ---
directories_to_create = [
    REPORTS_DIR,
    FIGURES_DIR,
    DOCS_DIR,
    LOGS_DIR,
    RAW_DATA_DIR,
    EXTERNAL_DATA_DIR,
    ARTIFACTS_DIR,
]

for path in directories_to_create:
    path.mkdir(parents=True, exist_ok=True)
