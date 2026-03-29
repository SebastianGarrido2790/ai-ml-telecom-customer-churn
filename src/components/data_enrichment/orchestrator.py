"""
Batch processing orchestrator for the Agentic Data Enrichment phase.

Manages the flow from raw CSV ingestion through concurrent LLM synthesis
to final artifact persistence. Implements resume logic so interrupted runs
continue from the last checkpoint rather than restarting from row zero.

Leakage Prevention (C1 Fix):
    The `Churn` argument has been removed from the `CustomerInputContext`
    constructor call in `_process_row`. The expanded schema now maps all
    observable CRM fields for each row. No target-variable information
    is passed to the LLM at any point in the pipeline.
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
    """Orchestrates batch processing of the Telco dataset for LLM enrichment.

    Routes deterministic row data into the probabilistic LLM generator,
    tracking errors and retries, and persists the enriched DataFrame
    progressively to prevent data loss on interruption.

    Attributes:
        raw_data_path: Path to the raw Telco dataset CSV.
        output_path: Path where the enriched dataset will be saved.
        config: Comprehensive configuration for enrichment provider and limits.
    """

    def __init__(
        self,
        raw_data_path: str | Path,
        output_path: str | Path,
        config: DataEnrichmentConfig,
    ) -> None:
        """Initializes the orchestrator with input/output paths and config.

        Args:
            raw_data_path: Path to the raw Telco dataset.
            output_path: Path where the enriched dataset will be saved.
            config: DataEnrichmentConfig with all provider and limit settings.
        """
        self.raw_data_path = Path(raw_data_path)
        self.output_path = Path(output_path)
        self.config = config

    async def _process_row(
        self,
        row: pd.Series,
        real_idx: int,
        semaphore: asyncio.Semaphore,
    ) -> tuple[int, SyntheticNoteOutput | None]:
        """Processes a single row concurrently, bounded by a global semaphore.

        Constructs a leakage-free CustomerInputContext from observable CRM
        fields only. The Churn column is never passed to the LLM.

        Args:
            row: A single DataFrame row as a pandas Series.
            real_idx: The original DataFrame index for result placement.
            semaphore: Global concurrency limiter for API rate control.

        Returns:
            Tuple of (original_index, SyntheticNoteOutput or None on failure).
        """
        async with semaphore:
            try:
                context = CustomerInputContext(
                    customerID=str(row.get("customerID", "unknown")),
                    tenure=int(row.get("tenure", 0)),
                    gender=str(row.get("gender", "Female")),
                    SeniorCitizen=int(row.get("SeniorCitizen", 0)),
                    Partner=str(row.get("Partner", "No")),
                    Dependents=str(row.get("Dependents", "No")),
                    InternetService=str(row.get("InternetService", "No")),
                    OnlineSecurity=str(row.get("OnlineSecurity", "No")),
                    OnlineBackup=str(row.get("OnlineBackup", "No")),
                    DeviceProtection=str(row.get("DeviceProtection", "No")),
                    TechSupport=str(row.get("TechSupport", "No")),
                    StreamingTV=str(row.get("StreamingTV", "No")),
                    StreamingMovies=str(row.get("StreamingMovies", "No")),
                    Contract=str(row.get("Contract", "Month-to-month")),
                    PaperlessBilling=str(row.get("PaperlessBilling", "No")),
                    PaymentMethod=str(row.get("PaymentMethod", "Mailed check")),
                    MonthlyCharges=float(row.get("MonthlyCharges", 0.0)),
                )
                res = await generate_ticket_note(context, config=self.config)
                return real_idx, res
            except ValidationError as ve:
                logger.warning(f"[ValidationError] Skipping row {row.get('customerID')}: {ve}")
                return real_idx, None
            except Exception as e:
                logger.error(f"[GeneratorError] Output synthesis failed for row {real_idx}: {e}")
                return real_idx, None

    async def run_enrichment(
        self,
        batch_size: int = 50,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Reads the raw CSV, identifies unprocessed rows, and runs enrichment.

        Implements resume logic: if the output file already exists, previously
        processed rows are loaded back and only missing rows are sent to the LLM.
        Checkpoints are saved every 10 completed rows to prevent full data loss.

        Args:
            batch_size: Maximum number of concurrent LLM API calls.
            limit: Maximum number of rows to process (None for full dataset).

        Returns:
            Enriched DataFrame with ticket_note and primary_sentiment_tag columns.
        """
        logger.info(f"Loading raw data from {self.raw_data_path}")
        df = pd.read_csv(self.raw_data_path)

        if limit is not None and limit > 0:
            df = df.head(limit)

        if "ticket_note" not in df.columns:
            df["ticket_note"] = None
        if "primary_sentiment_tag" not in df.columns:
            df["primary_sentiment_tag"] = None

        # Resume logic: restore previously completed rows from output file
        if self.output_path.exists():
            logger.info(f"Found existing enrichment file at {self.output_path}. Resuming progress...")
            existing_df = pd.read_csv(self.output_path)
            existing_map = existing_df.set_index("customerID")[["ticket_note", "primary_sentiment_tag"]].to_dict(
                "index"
            )

            for idx, row in df.iterrows():
                cid = str(row["customerID"])
                if cid in existing_map:
                    df.at[idx, "ticket_note"] = existing_map[cid].get("ticket_note")
                    df.at[idx, "primary_sentiment_tag"] = existing_map[cid].get("primary_sentiment_tag")

        mask = df["ticket_note"].isna() | df["primary_sentiment_tag"].isna()
        indices_to_process = df[mask].index.tolist()

        total_rows = len(df)
        processed_count = total_rows - len(indices_to_process)
        to_process_count = len(indices_to_process)

        logger.info(f"Total rows: {total_rows} | Already processed: {processed_count} | To process: {to_process_count}")

        if not indices_to_process:
            logger.info("All rows already enriched. Skipping LLM processing.")
            return df

        semaphore = asyncio.Semaphore(batch_size)
        tasks = [asyncio.create_task(self._process_row(df.loc[idx], idx, semaphore)) for idx in indices_to_process]

        save_interval = 10
        total_to_process = len(tasks)
        for completed, future in enumerate(asyncio.as_completed(tasks), 1):
            real_idx, res = await future

            if res is not None:
                df.at[real_idx, "ticket_note"] = res.ticket_note
                df.at[real_idx, "primary_sentiment_tag"] = res.primary_sentiment_tag

            if completed % 5 == 0 or completed == total_to_process:
                logger.info(f"Progress: {completed} / {total_to_process} (Total dataset: {total_rows})")

            if completed % save_interval == 0:
                logger.info(f"Checkpoint at {completed}: saving to {self.output_path}")
                self.output_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(self.output_path, index=False)

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving final enriched dataset to {self.output_path}")
        df.to_csv(self.output_path, index=False)
        logger.info("Enrichment phase completed successfully.")

        return df
