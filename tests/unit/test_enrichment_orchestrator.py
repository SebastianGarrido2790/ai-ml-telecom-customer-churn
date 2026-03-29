"""
Unit tests for the Data Enrichment Orchestrator.

This module validates the batch processing and state persistence logic of the
enrichment pipeline. It ensures that large customer datasets are enriched
efficiently with periodic checkpointing and reliable resume capabilities.

Key validations:
    - run_enrichment: Async batch execution logic.
    - Resume Logic: Detection and skipping of pre-enriched records.
    - Checkpointing: Immutable persistence of state every N records.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from src.components.data_enrichment.orchestrator import EnrichmentOrchestrator
from src.components.data_enrichment.schemas import SyntheticNoteOutput


@pytest.fixture
def mock_enrich_config():
    config = MagicMock()
    config.model_provider = "google"
    config.model_name = "gemini-test"
    config.batch_size = 2
    return config


@pytest.fixture
def raw_data_csv(tmp_path):
    df = pd.DataFrame(
        {
            "customerID": ["C1", "C2", "C3"],
            "tenure": [1, 2, 3],
            "gender": ["Male", "Female", "Male"],
            "MonthlyCharges": [29.99, 39.99, 49.99],
        }
    )
    csv_path = tmp_path / "raw.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


@pytest.mark.asyncio
async def test_run_enrichment_full(raw_data_csv, tmp_path, mock_enrich_config):
    """Test full enrichment of a small dataset."""
    output_path = tmp_path / "enriched.csv"
    orchestrator = EnrichmentOrchestrator(raw_data_csv, output_path, mock_enrich_config)

    mock_note = SyntheticNoteOutput(ticket_note="Note", primary_sentiment_tag="Neutral")

    with patch(
        "src.components.data_enrichment.orchestrator.generate_ticket_note",
        new_callable=AsyncMock,
        return_value=mock_note,
    ) as mock_gen:
        result_df = await orchestrator.run_enrichment(batch_size=2)

        assert len(result_df) == 3
        assert not result_df["ticket_note"].isna().any()
        assert mock_gen.call_count == 3
        assert output_path.exists()


@pytest.mark.asyncio
async def test_run_enrichment_resume(raw_data_csv, tmp_path, mock_enrich_config):
    """Test resume logic: only missing rows are processed."""
    output_path = tmp_path / "enriched.csv"

    # Predominantly enriched file (only C2 is missing)
    existing_df = pd.DataFrame(
        {
            "customerID": ["C1", "C3"],
            "tenure": [1, 3],
            "ticket_note": ["Existing C1", "Existing C3"],
            "primary_sentiment_tag": ["Satisfied", "Satisfied"],
        }
    )
    existing_df.to_csv(output_path, index=False)

    orchestrator = EnrichmentOrchestrator(raw_data_csv, output_path, mock_enrich_config)

    mock_note = SyntheticNoteOutput(ticket_note="New C2", primary_sentiment_tag="Neutral")

    with patch(
        "src.components.data_enrichment.orchestrator.generate_ticket_note",
        new_callable=AsyncMock,
        return_value=mock_note,
    ) as mock_gen:
        result_df = await orchestrator.run_enrichment(batch_size=2)

        assert len(result_df) == 3
        assert result_df.loc[0, "ticket_note"] == "Existing C1"
        assert result_df.loc[1, "ticket_note"] == "New C2"
        assert result_df.loc[2, "ticket_note"] == "Existing C3"
        assert mock_gen.call_count == 1  # Only C2 was processed


@pytest.mark.asyncio
async def test_run_enrichment_checkpointing(raw_data_csv, tmp_path, mock_enrich_config):
    """Test that checkpoints are saved during processing."""
    # Create 15 rows to trigger a checkpoint (interval=10)
    df = pd.DataFrame(
        {
            "customerID": [f"C{i}" for i in range(15)],
            "tenure": [1] * 15,
        }
    )
    large_raw = tmp_path / "large_raw.csv"
    df.to_csv(large_raw, index=False)

    output_path = tmp_path / "large_enriched.csv"
    orchestrator = EnrichmentOrchestrator(large_raw, output_path, mock_enrich_config)

    mock_note = SyntheticNoteOutput(ticket_note="Note", primary_sentiment_tag="Neutral")

    with patch(
        "src.components.data_enrichment.orchestrator.generate_ticket_note",
        new_callable=AsyncMock,
        return_value=mock_note,
    ):
        # We'll use a small save_interval if possible, but the code has it hardcoded as 10.
        # So 15 rows will trigger 1 checkpoint at row 10.
        await orchestrator.run_enrichment(batch_size=5)

        assert output_path.exists()
        saved_df = pd.read_csv(output_path)
        assert len(saved_df) == 15
        assert not saved_df["ticket_note"].isna().any()
