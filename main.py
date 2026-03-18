"""
Main entry point for the Telecom Customer Churn pipeline.

This script allows for manual orchestration of the pipeline stages for debugging
and development purposes.

Usage:
    uv run python main.py
"""

import asyncio
import sys

from src.pipeline.stage_00_data_ingestion import main as stage_00_main
from src.pipeline.stage_01_data_validation import main as stage_01_main
from src.pipeline.stage_02_data_enrichment import main as stage_02_main
from src.pipeline.stage_03_enriched_validation import main as stage_03_main
from src.utils.exceptions import CustomException
from src.utils.logger import get_logger, log_spacer

logger = get_logger(__name__, headline="main.py")


async def run_pipeline() -> None:
    """
    Orchestrates the Telecom Customer Churn pipeline stages.

    Stages:
    0. Stage 00: Data Ingestion
    1. Stage 01: Raw Data Validation (Great Expectations)
    2. Stage 02: Data Enrichment (LLM Synthetic Notes)
    3. Stage 03: Enriched Data Validation (Great Expectations)
    """
    try:
        logger.info("Starting Pipeline Orchestration...")

        log_spacer()
        logger.info("STAGE 00: Data Ingestion")
        stage_00_main()

        log_spacer()
        logger.info("STAGE 01: Raw Data Validation")
        stage_01_main()

        log_spacer()
        logger.info("STAGE 02: Data Enrichment")
        # Stage 02 is async because it performs multiple I/O bound LLM calls
        await stage_02_main()

        log_spacer()
        logger.info("STAGE 03: Enriched Data Validation")
        stage_03_main()

        logger.info("Pipeline Orchestration Finished Successfully ✅")

    except Exception as e:
        logger.error(f"Pipeline Orchestration Failed: {str(e)} ❌")
        raise CustomException(e, sys) from e


if __name__ == "__main__":
    asyncio.run(run_pipeline())
