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
from src.entity.config_entity import DataEnrichmentConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


class EnrichmentOrchestrator:
    """
    The "Brain" of Phase 2.

    Orchestrates batches of deterministic rows into the probabilistic LLM generator,
    tracking errors, retries, and returning the updated DataFrame.
    """

    def __init__(
        self, raw_data_path: str | Path, output_path: str | Path, config: DataEnrichmentConfig
    ):
        """
        Initializes the orchestrator with input and output paths.

        Args:
            raw_data_path (str | Path): Path to the raw Telco dataset.
            output_path (str | Path): Path where the enriched dataset will be saved.
            config (DataEnrichmentConfig): Comprehensive configuration for enrichment.
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.config = config

    async def _process_row(
        self, row: pd.Series, real_idx: int, semaphore: asyncio.Semaphore
    ) -> tuple[int, SyntheticNoteOutput | None]:
        """
        Processes a single row concurrently, bounded by a global semaphore.
        """
        async with semaphore:
            try:
                context = CustomerInputContext(
                    customerID=str(row.get("customerID", "unknown")),
                    tenure=int(row.get("tenure", 0)),
                    InternetService=str(row.get("InternetService", "No")),
                    Contract=str(row.get("Contract", "Month-to-month")),
                    MonthlyCharges=float(row.get("MonthlyCharges", 0.0)),
                    TechSupport=str(row.get("TechSupport", "No")),
                    Churn=str(row.get("Churn", "No")),
                )
                res = await generate_ticket_note(context, config=self.config)
                return real_idx, res
            except ValidationError as ve:
                logger.warning(f"[ValidationError] Skipping row {row.get('customerID')}: {ve}")
                return real_idx, None
            except Exception as e:
                logger.error(f"[GeneratorError] Output synthesis failed for row {real_idx}: {e}")
                return real_idx, None

    async def run_enrichment(self, batch_size: int = 50, limit: int | None = None) -> pd.DataFrame:
        """
        Reads the CSV, identifies rows already processed in the output file,
        and continues processing missing rows.
        """
        logger.info(f"Loading raw data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)

        if limit is not None and limit > 0:
            df = df.head(limit)

        # Initialize columns if they don't exist
        if "ticket_note" not in df.columns:
            df["ticket_note"] = None
        if "primary_sentiment_tag" not in df.columns:
            df["primary_sentiment_tag"] = None

        # Resume logic: Check if output already exists
        if self.output_path.exists():
            logger.info(
                f"Found existing enrichment file at {self.output_path}. Resuming progress..."
            )
            existing_df = pd.read_csv(self.output_path)

            # Map existing results back to raw df based on customerID (unique)
            # Use 'customerID' to join because indices might differ if limit changed
            existing_map = existing_df.set_index("customerID")[
                ["ticket_note", "primary_sentiment_tag"]
            ].to_dict("index")

            for idx, row in df.iterrows():
                cid = str(row["customerID"])
                if cid in existing_map:
                    df.at[idx, "ticket_note"] = existing_map[cid].get("ticket_note")
                    df.at[idx, "primary_sentiment_tag"] = existing_map[cid].get(
                        "primary_sentiment_tag"
                    )


        # Identify indices needing processing
        # We need work if either column is null (using pd.isna to handle None/NaN)
        mask = df["ticket_note"].isna() | df["primary_sentiment_tag"].isna()
        indices_to_process = df[mask].index.tolist()

        total_rows = len(df)
        processed_count = total_rows - len(indices_to_process)
        to_process_count = len(indices_to_process)


        logger.info(
            f"Total rows: {total_rows} | "
            f"Already processed: {processed_count} | "
            f"To process: {to_process_count}"
        )


        if not indices_to_process:
            logger.info("All rows already enriched. Skipping LLM processing.")
            return df

        # Parallel lists for the session (full length)
        # Note: We keep the dataframe logic consistent
        semaphore = asyncio.Semaphore(batch_size)
        tasks = []
        for idx in indices_to_process:
            row = df.loc[idx]
            tasks.append(asyncio.create_task(self._process_row(row, idx, semaphore)))

        save_interval = 10
        completed = 0
        total_to_process = len(tasks)

        for future in asyncio.as_completed(tasks):
            real_idx, res = await future
            completed += 1

            if res is not None:
                df.at[real_idx, "ticket_note"] = res.ticket_note
                df.at[real_idx, "primary_sentiment_tag"] = res.primary_sentiment_tag

            # LOGGING FREQUENCY (Rule 1.7 - Better Prompting/UX)
            if completed % 5 == 0 or completed == total_to_process:
                logger.info(
                    f"Progress: {completed} / {total_to_process} batches (Total: {total_rows})"
                )

            if completed % save_interval == 0:
                logger.info(f"Checkpoint at {completed}: Saving progress to {self.output_path}")
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.output_path, index=False)

        # Final dataset construction
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving final enriched dataset to {self.output_path}")
        df.to_csv(self.output_path, index=False)
        logger.info("Enrichment phase completed successfully.")

        return df
