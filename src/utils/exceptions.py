"""
Custom exception definitions for the Data Quality and enrichment pipelines.
Provides structured error context for both developers and AI agents.
"""

from dataclasses import dataclass
from typing import Any
from types import ModuleType


def error_message_detail(error: Exception | str, error_detail: ModuleType) -> str:
    """
    Extracts the detailed error message including file name and line number.

    Args:
        error (Exception | str): The exception or error message.
        error_detail (ModuleType): The sys module to access execution info.

    Returns:
        str: A formatted error message string.
    """
    _, _, exc_tb = error_detail.exc_info()

    # Safety check to prevent crashes in edge cases where the traceback might be incomplete.
    if exc_tb is not None and exc_tb.tb_frame is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
    else:
        file_name = "unknown"
        line_number = 0

    error_message = (
        f"Error occurred in python script: [{file_name}] "
        f"line number: [{line_number}] "
        f"error message: [{str(error)}]"
    )

    return error_message


class CustomException(Exception):
    """
    Custom Exception class to provide detailed traceback information within the message.
    This is critical for MLOps pipelines to quickly debug failures in automated workflows and prevent silent failures.

    Implementation details:
    - Captures Context: Automatically extracts the file name and line number where the error occurred.
    - Formatting: Wraps the error into a standardized string format for logs.
    - Strict Typing: Uses ModuleType instead of untyped sys imports to satisfy modern linters.
    """

    def __init__(self, error_message: Exception | str, error_detail: ModuleType):
        """
        Initialize the CustomException.

        Args:
            error_message (Exception | str): The original error message or exception object.
            error_detail (ModuleType): The sys module to capture stack trace.
        """
        # Generate the detailed message
        self.detailed_message = error_message_detail(error=error_message, error_detail=error_detail)
        # Call the base class constructor with the detailed message
        super().__init__(self.detailed_message)

    def __str__(self) -> str:
        return self.detailed_message


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
