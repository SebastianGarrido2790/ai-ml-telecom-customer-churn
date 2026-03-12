"""
Script to validate the enriched Telco dataset (Phase 2 output).

Ensures that the LLM-generated synthetic ticket notes and sentiment tags
adhere to the defined data contracts before the features are consumed
by the training pipeline.
"""

import pandas as pd
from dotenv import load_dotenv

from src.components.data_validation import DataValidator
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

# Load environment variables
load_dotenv()

logger = get_logger(__name__, headline="Enriched Data Validation")


def main() -> None:
    """
    Orchestrates the enriched data validation process.

    1. Loads configuration using the ConfigurationManager.
    2. Reads the enriched dataset from the artifacts directory.
    3. Initializes the DataValidator and builds the 'enriched_telco_churn_suite'.
    4. Executes validation and logs results, ensuring LLM outputs are high-quality.
    """
    config_mgr = ConfigurationManager()
    enrich_config = config_mgr.get_data_enrichment_config()

    # 1. Define Paths
    enriched_data_path = enrich_config.enriched_data_file

    if not enriched_data_path.exists():
        logger.error(f"Enriched data file not found at {enriched_data_path}")
        return

    # 2. Load Data
    logger.info(f"Loading enriched data from {enriched_data_path}...")
    df = pd.read_csv(enriched_data_path)

    # 3. Initialize Validator
    validator = DataValidator()

    # 4. Build Suite (using enriched schema from schema.yaml)
    logger.info("Building enriched telco expectation suite...")
    validator.build_enriched_telco_suite(schema=enrich_config.all_schema)

    # 5. Validate
    logger.info("Running validation...")
    try:
        validator.validate_dataset(
            df=df,
            suite_name="enriched_telco_churn_suite",
            dataset_id="telco_enriched",
            pipeline_stage="enrichment",
        )
        logger.info("Enriched Data Validation PASSED ✅")
    except Exception as e:
        logger.error("Enriched Data Validation FAILED ❌")
        if hasattr(e, "to_agent_context"):
            logger.error(e.to_agent_context())  # type: ignore
        else:
            logger.error(str(e))


if __name__ == "__main__":
    main()
