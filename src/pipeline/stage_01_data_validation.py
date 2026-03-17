"""
Script to validate the raw Telco dataset using Great Expectations.

Consumes the system configuration to locate the raw data and executes
a predefined suite of statistical and schema contracts.
"""

import json
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.components.data_validation import DataValidator
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__, headline="Raw Data Validation")


def main() -> None:
    """
    Orchestrates the raw data validation process.

    1. Loads configuration using the ConfigurationManager.
    2. Reads the raw dataset from the configured path.
    3. Initializes the DataValidator and builds the 'raw_telco_churn_suite'.
    4. Executes validation and logs results, failing loudly on contract violations.
    """
    config_mgr = ConfigurationManager()
    val_config = config_mgr.get_data_validation_config()

    # 1. Define Paths
    raw_data_path = val_config.unzip_data_dir

    if not raw_data_path.exists():
        logger.error(f"Raw data file not found at {raw_data_path}")
        return

    # 2. Load Data
    logger.info(f"Loading raw data from {raw_data_path}...")
    df = pd.read_csv(raw_data_path)

    # 3. Initialize Validator
    validator = DataValidator()

    # 4. Build Suite (using schema.yaml from ConfigurationManager)
    logger.info("Building raw telco expectation suite...")
    validator.build_raw_telco_suite(schema=val_config.all_schema)

    # 5. Validate
    logger.info("Running validation...")
    status_file = Path(val_config.STATUS_FILE)
    report_file = val_config.root_dir / "validation_report.json"

    try:
        results = validator.validate_dataset(
            df=df,
            suite_name="raw_telco_churn_suite",
            dataset_id="telco_raw",
            pipeline_stage="ingestion",
        )
        logger.info("Raw Data Validation PASSED ✅")
        status_file.write_text("Validation Status: PASS")
        with report_file.open("w") as f:
            json.dump(results, f, indent=4)
    except Exception as e:
        logger.error("Raw Data Validation FAILED ❌")
        status_file.write_text("Validation Status: FAIL")
        if hasattr(e, "to_agent_context"):
            logger.error(e.to_agent_context())  # type: ignore
        else:
            logger.error(str(e))
        raise


if __name__ == "__main__":
    main()
