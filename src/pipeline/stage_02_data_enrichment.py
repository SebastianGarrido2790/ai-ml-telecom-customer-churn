"""
Entry point script for the Data Enrichment phase.
Executes the orchestration pipeline on the raw Telco dataset.
"""

import asyncio

from src.components.data_enrichment.orchestrator import EnrichmentOrchestrator
from src.config.configuration import ConfigurationManager
from src.utils.logger import get_logger

logger = get_logger(__name__, headline="Data Enrichment")


async def main() -> None:
    """
    Initializes paths and executes the enrichment process with batch limits.
    """
    config_mgr = ConfigurationManager()
    enrich_config = config_mgr.get_data_enrichment_config()

    orchestrator = EnrichmentOrchestrator(
        raw_data_path=enrich_config.raw_data_path,
        output_path=enrich_config.enriched_data_file,
        model_name=enrich_config.model_name,
    )

    limit = enrich_config.limit
    batch_size = enrich_config.batch_size

    if limit is None:
        logger.info("Running enrichment on the ENTIRE dataset (limit=None).")
    else:
        logger.info(f"Running enrichment with LIMIT: {limit}")

    logger.info(f"Model: {enrich_config.model_name} | Batch Size: {batch_size}")
    logger.info("Starting enrichment process...")
    await orchestrator.run_enrichment(batch_size=batch_size, limit=limit)
    logger.info("Main script execution finished.")


if __name__ == "__main__":
    asyncio.run(main())
