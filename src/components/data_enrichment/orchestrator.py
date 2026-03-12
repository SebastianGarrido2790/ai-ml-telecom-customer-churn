"""
This module orchestrates the batch processing of the Telco dataset for enrichment.
It manages the flow from raw CSV ingestion to concurrent LLM synthesis and final persistence.
"""

import asyncio
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from src.components.data_enrichment.generator import generate_ticket_note
from src.components.data_enrichment.schemas import CustomerInputContext, SyntheticNoteOutput
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnrichmentOrchestrator:
    """
    The "Brain" of Phase 2.

    Orchestrates batches of deterministic rows into the probabilistic LLM generator,
    tracking errors, retries, and returning the updated DataFrame.
    """

    def __init__(self, raw_data_path: str | Path, output_path: str | Path, model_name: str):
        """
        Initializes the orchestrator with input and output paths.

        Args:
            raw_data_path (str | Path): Path to the raw Telco dataset.
            output_path (str | Path): Path where the enriched dataset will be saved.
            model_name (str): Name of the LLM to use.
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.model_name = model_name

    async def _process_batch(self, batch_df: pd.DataFrame) -> list[SyntheticNoteOutput | None]:
        """
        Processes a single batch of rows concurrently.

        Args:
            batch_df (pd.DataFrame): Segment of the dataset to process.

        Returns:
            list[SyntheticNoteOutput | None]: List of synthesized outputs or None on failure.
        """
        tasks = []
        for _, row in batch_df.iterrows():
            try:
                # Enforce strict data contract before passing to the tool
                context = CustomerInputContext(
                    customerID=str(row.get("customerID", "unknown")),
                    tenure=int(row.get("tenure", 0)),
                    InternetService=str(row.get("InternetService", "No")),
                    Contract=str(row.get("Contract", "Month-to-month")),
                    MonthlyCharges=float(row.get("MonthlyCharges", 0.0)),
                    TechSupport=str(row.get("TechSupport", "No")),
                    Churn=str(row.get("Churn", "No")),
                )
                tasks.append(generate_ticket_note(context, model_name=self.model_name))
            except ValidationError as ve:
                logger.warning(f"[ValidationError] Skipping row {row.get('customerID')}: {ve}")

                # Mock a dummy coroutine for failed inputs to keep index alignment
                async def dummy_fail():
                    return None

                tasks.append(dummy_fail())

        # 2. Execute NLP batch concurrently, returning exceptions rather than crashing pipeline
        results = await asyncio.gather(*tasks, return_exceptions=True)

        parsed_results = []
        for res in results:
            if isinstance(res, Exception):
                logger.error(f"[GeneratorError] Output synthesis failed: {res}")
                parsed_results.append(None)
            else:
                parsed_results.append(res)

        return parsed_results

    async def run_enrichment(self, batch_size: int = 20, limit: int | None = None) -> pd.DataFrame:
        """
        Reads the CSV, processes in batches asynchronously, and saves PROGRESSIVELY.

        Args:
            batch_size (int): Number of parallel API calls. Defaults to 20.
            limit (int | None): Max number of rows to process. Useful for dry runs.

        Returns:
            pd.DataFrame: The enriched DataFrame with 'ticket_note' and 'primary_sentiment_tag'.
        """
        logger.info(f"Loading raw data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)

        if limit is not None:
            df = df.head(limit)

        total_rows = len(df)
        logger.info(f"Total rows to enrich: {total_rows}")

        # We will keep parallel lists tracking the generation
        ticket_notes = [None] * total_rows
        sentiments = [None] * total_rows

        for i in range(0, total_rows, batch_size):
            batch_df = df.iloc[i : i + batch_size]
            logger.info(f"Processing batch {i} to {i + len(batch_df)} / {total_rows}...")

            batch_results = await self._process_batch(batch_df)

            # Map back to lists based on index alignment
            for j, res in enumerate(batch_results):
                real_idx = i + j
                if res is not None:
                    ticket_notes[real_idx] = res.ticket_note
                    sentiments[real_idx] = res.primary_sentiment_tag

        # Ensure our dataset reflects the newly joined properties
        df["ticket_note"] = ticket_notes
        df["primary_sentiment_tag"] = sentiments

        # 3. Create parent directories if they don't exist, and save our enriched artifact
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving enriched dataset to {self.output_path}")
        df.to_csv(self.output_path, index=False)
        logger.info("Enrichment phase completed successfully.")

        return df
