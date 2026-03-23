"""
This module implements the Data Validation phase using Great Expectations (GX).
It defines the expectation suites for raw and enriched datasets and provides
a runner to execute validation checkpoints.
"""

from pathlib import Path
from typing import Any

import great_expectations as gx
import pandas as pd
from great_expectations.core.expectation_suite import ExpectationSuite

from src.utils.exceptions import DataQualityContext, StatisticalContractViolation
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataValidator:
    """
    Orchestrates Data Quality gates using Great Expectations.
    Adheres to the FTI architecture by blocking bad data before training or inference.
    """

    def __init__(self, data_context_path: str | Path | None = None) -> None:
        """
        Initializes the GX Data Context.

        Args:
            data_context_path: Path to the GX project directory.
                              If None, uses ephemeral context (standard for microservices).
        """
        if data_context_path:
            self.context = gx.get_context(context_root_dir=str(data_context_path))
        else:
            self.context = gx.get_context()

    def build_raw_telco_suite(
        self, suite_name: str = "raw_telco_churn_suite", schema: dict[str, Any] | None = None
    ) -> ExpectationSuite:
        """
        Defines expectations for the raw Telco dataset based on EDA findings.

        Uses the GX 1.0+ API to create or retrieve the suite.

        Args:
            suite_name (str): Name of the expectation suite. Defaults to "raw_telco_churn_suite".
            schema (dict[str, Any] | None): Optional schema dictionary from schema.yaml.
                                           If provided, enforces column presence.

        Returns:
            ExpectationSuite: The populated suite.
        """
        from great_expectations.expectations import (
            ExpectColumnValuesToBeBetween,
            ExpectColumnValuesToBeInSet,
            ExpectTableColumnsToMatchSet,
        )

        try:
            suite = self.context.suites.get(name=suite_name)
            return suite
        except Exception:
            suite = gx.ExpectationSuite(name=suite_name)

        # 1. Column Presence (Driven by schema.yaml if available)
        if schema:
            required_cols = list(schema.keys())
            logger.info(f"Enforcing presence of {len(required_cols)} columns from schema.yaml")
        else:
            required_cols = [
                "customerID",
                "tenure",
                "InternetService",
                "Contract",
                "MonthlyCharges",
                "TotalCharges",
                "Churn",
            ]

        suite.add_expectation(
            ExpectTableColumnsToMatchSet(column_set=required_cols, exact_match=False)
        )

        # 2. Tenure Range
        suite.add_expectation(
            ExpectColumnValuesToBeBetween(column="tenure", min_value=0, max_value=120)
        )

        # 3. Internet Service Values
        suite.add_expectation(
            ExpectColumnValuesToBeInSet(
                column="InternetService",
                value_set=["DSL", "Fiber optic", "No"],
            )
        )

        # 4. Contract Values
        suite.add_expectation(
            ExpectColumnValuesToBeInSet(
                column="Contract",
                value_set=["Month-to-month", "One year", "Two year"],
            )
        )

        # 5. Churn Values
        suite.add_expectation(ExpectColumnValuesToBeInSet(column="Churn", value_set=["Yes", "No"]))

        self.context.suites.add(suite)
        return suite

    def build_enriched_telco_suite(
        self, suite_name: str = "enriched_telco_churn_suite", schema: dict[str, Any] | None = None
    ) -> ExpectationSuite:
        """
        Defines expectations for the AI-enriched dataset (Phase 2 output).

        Uses the GX 1.0+ API to create or retrieve the suite.

        Args:
            suite_name (str): Name of the expectation suite.
                             Defaults to "enriched_telco_churn_suite".
            schema (dict[str, Any] | None): Optional schema dictionary.

        Returns:
            ExpectationSuite: The populated suite.
        """
        from great_expectations.expectations import (
            ExpectColumnValueLengthsToBeBetween,
            ExpectColumnValuesToBeInSet,
            ExpectColumnValuesToNotBeNull,
            ExpectTableColumnsToMatchSet,
        )

        try:
            suite = self.context.suites.get(name=suite_name)
            return suite
        except Exception:
            suite = gx.ExpectationSuite(name=suite_name)

        # 0. Column Presence (Driven by schema.yaml if available)
        if schema:
            required_cols = list(schema.keys())
            logger.info(
                f"Enforcing presence of {len(required_cols)} columns (enriched) from schema.yaml"
            )
            suite.add_expectation(
                ExpectTableColumnsToMatchSet(column_set=required_cols, exact_match=False)
            )

        # 1. Ticket Note Presence and Quality
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column="ticket_note", mostly=0.95))
        suite.add_expectation(
            ExpectColumnValueLengthsToBeBetween(column="ticket_note", min_value=10, mostly=0.9)
        )

        # 2. Sentiment Tag Consistency
        # value_set must match SyntheticNoteOutput.primary_sentiment_tag Literal exactly.
        # "Dissatisfied" was absent from the original suite but is a valid schema tag
        # produced by the leakage-free prompt (C1 fix).
        suite.add_expectation(
            ExpectColumnValuesToBeInSet(
                column="primary_sentiment_tag",
                value_set=[
                    "Frustrated",
                    "Dissatisfied",
                    "Neutral",
                    "Satisfied",
                    "Billing Inquiry",
                    "Technical Issue",
                ],
                mostly=0.95,
            )
        )

        self.context.suites.add(suite)
        return suite

    def validate_dataset(
        self, df: pd.DataFrame, suite_name: str, dataset_id: str, pipeline_stage: str
    ) -> dict[str, Any]:
        """
        Runs validation and raises statistical contract violations on failure.

        Args:
            df (pd.DataFrame): The dataset to validate.
            suite_name (str): The name of the expectation suite to use.
            dataset_id (str): A unique ID for the dataset (used for logging).
            pipeline_stage (str): The current pipeline stage (e.g., 'ingestion', 'enrichment').

        Returns:
            dict[str, Any]: The validation results as a JSON-compatible dictionary.

        Raises:
            StatisticalContractViolation: If the validation fail threshold is met.
        """
        datasource_name = f"ds_{dataset_id}"
        try:
            self.context.get_datasource(datasource_name)
        except Exception:
            self.context.data_sources.add_pandas(name=datasource_name)

        datasource = self.context.get_datasource(datasource_name)
        data_asset_name = f"asset_{dataset_id}"
        # In GX 1.0, for pandas, we can use read_dataframe to get a batch directly
        batch = datasource.read_dataframe(dataframe=df, asset_name=data_asset_name)

        validator = self.context.get_validator(batch=batch, expectation_suite_name=suite_name)

        results = validator.validate()

        if not results["success"]:
            # Extract failed expectations for the exception context
            failed_results = [r for r in results["results"] if not r["success"]]
            failed_logs = [
                f"{r['expectation_config']['type']} on "
                f"'{r['expectation_config']['kwargs'].get('column', 'table')}'"
                for r in failed_results[:5]
            ]

            raise StatisticalContractViolation(
                message=f"Dataset {dataset_id} failed {len(failed_results)} expectations.",
                context=DataQualityContext(
                    dataset_id=dataset_id,
                    pipeline_stage=pipeline_stage,
                    column=None,
                    expectation=f"All expectations in {suite_name} must pass.",
                    actual_value=failed_logs,
                    row_count_affected=len(df),
                    suggested_action="Review data quality and fix issues or adjust expectations.",
                ),
            )

        logger.info(f"Validation successful for {dataset_id} using suite {suite_name}")
        return results.to_json_dict()
