"""
Custom exception definitions for the Data Quality and enrichment pipelines.
Provides structured error context for both developers and AI agents.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class DataQualityContext:
    """
    Structured context attached to every data quality exception.

    Provides specific details about where and why a data contract was violated,
    allowing agents to attempt self-correction.

    Attributes:
        dataset_id (str): Unique identifier for the dataset.
        pipeline_stage (str): The stage where the error occurred (e.g., 'ingestion').
        column (str | None): The specific column that failed validation, if applicable.
        expectation (str): A description of the expected state/value.
        actual_value (Any): The actual value or summary of values observed.
        row_count_affected (int): Number of rows impacted by the violation.
        suggested_action (str): A hint for resolution (e.g., 'Check for nulls').
    """

    dataset_id: str
    pipeline_stage: str
    column: str | None
    expectation: str
    actual_value: Any
    row_count_affected: int = 0
    suggested_action: str = ""


class DataQualityError(Exception):
    """
    Base exception for all Feature Pipeline data quality failures.

    Adheres to the FTI architecture by providing a standardized error interface.
    """

    def __init__(self, message: str, context: DataQualityContext) -> None:
        """
        Initializes the error with a message and structured context.

        Args:
            message (str): High-level error description.
            context (DataQualityContext): Detailed metadata about the failure.
        """
        super().__init__(message)
        self.context = context

    def to_agent_context(self) -> str:
        """
        Serialize exception context for agent consumption.

        Returns:
            str: A formatted string designed for an LLM to interpret the failure.
        """
        ctx = self.context
        return (
            f"[DATA QUALITY FAILURE]\n"
            f"Stage: {ctx.pipeline_stage}\n"
            f"Dataset: {ctx.dataset_id}\n"
            f"Column: {ctx.column or 'N/A'}\n"
            f"Expected: {ctx.expectation}\n"
            f"Observed: {ctx.actual_value}\n"
            f"Rows affected: {ctx.row_count_affected}\n"
            f"Suggested action: {ctx.suggested_action or 'Inspect raw data source'}"
        )


class StatisticalContractViolation(DataQualityError):
    """
    Raised when Great Expectations validation fails on a dataset.
    """

    pass


class SchemaContractViolation(DataQualityError):
    """
    Raised when incoming data fails Pydantic schema validation.
    """

    pass
