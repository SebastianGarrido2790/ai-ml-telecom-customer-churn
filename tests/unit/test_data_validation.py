"""
Unit tests for the Data Validation component.

This module validates the data quality contract enforcement using the Great
Expectations framework. It ensures that both raw and enriched datasets adhere
to the structural and statistical rules defined in the system.

Key validations:
    - build_raw_telco_suite: Foundational Telco schema rules.
    - validate_dataset: Success and failure (StatisticalContractViolation) paths.
    - build_enriched_telco_suite: Validation of synthetic NLP features.
"""

import pandas as pd
import pytest

from src.components.data_validation import DataValidator
from src.utils.exceptions import StatisticalContractViolation


@pytest.fixture
def validator():
    """Provides a DataValidator with an ephemeral context."""
    return DataValidator()


@pytest.fixture
def sample_df():
    """A valid raw Telco dataset snippet."""
    return pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "tenure": [10, 20],
            "InternetService": ["DSL", "Fiber optic"],
            "Contract": ["Month-to-month", "Two year"],
            "MonthlyCharges": [50.0, 80.0],
            "TotalCharges": ["500", "1600"],
            "Churn": ["No", "Yes"],
        }
    )


def test_build_raw_telco_suite(validator):
    """Test that the raw Telco suite is built with foundational expectations."""
    suite = validator.build_raw_telco_suite()
    assert suite.name == "raw_telco_churn_suite"
    # Check that expectations were added (at least 5 based on code)
    assert len(suite.expectations) >= 5


def test_validate_dataset_success(validator, sample_df):
    """Test validation of a valid dataset."""
    validator.build_raw_telco_suite()
    results = validator.validate_dataset(
        df=sample_df,
        suite_name="raw_telco_churn_suite",
        dataset_id="test_success",
        pipeline_stage="ingestion",
    )
    assert results["success"] is True


def test_validate_dataset_failure(validator, sample_df):
    """Test that invalid data raises StatisticalContractViolation."""
    # Break the data: invalid tenure and invalid Churn
    sample_df.at[0, "tenure"] = 500  # Max is 120
    sample_df.at[1, "Churn"] = "Maybe"  # Valid are Yes/No

    validator.build_raw_telco_suite()

    with pytest.raises(StatisticalContractViolation) as excinfo:
        validator.validate_dataset(
            df=sample_df,
            suite_name="raw_telco_churn_suite",
            dataset_id="test_fail",
            pipeline_stage="ingestion",
        )

    assert "test_fail" in str(excinfo.value)
    assert "ingestion" in str(excinfo.value.context.pipeline_stage)


def test_build_enriched_telco_suite(validator):
    """Test that the enriched suite includes ticket_note expectations."""
    suite = validator.build_enriched_telco_suite()
    assert suite.name == "enriched_telco_churn_suite"

    # Enriched data snippet
    enriched_df = pd.DataFrame(
        {
            "ticket_note": ["Customer called to complain about billing."] * 10,
            "primary_sentiment_tag": ["Frustrated"] * 10,
        }
    )

    results = validator.validate_dataset(
        df=enriched_df,
        suite_name="enriched_telco_churn_suite",
        dataset_id="test_enriched",
        pipeline_stage="enrichment",
    )
    assert results["success"] is True
