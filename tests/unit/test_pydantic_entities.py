"""Phase 1 Verification: Pydantic data contract validation test."""

import pytest
from pydantic import ValidationError

from src.entity.config_entity import TelcoCustomerRow


def test_valid_row() -> None:
    """Test that a valid Telco row passes validation."""
    row = TelcoCustomerRow(
        customerID="7590-VHVEG",
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=1,
        PhoneService="No",
        MultipleLines="No phone service",
        InternetService="DSL",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=29.85,
        TotalCharges="29.85",
        Churn="No",
    )
    assert row.customerID == "7590-VHVEG"


def test_bad_row_rejected() -> None:
    """Test that a row with invalid values is rejected by Pydantic."""
    with pytest.raises(ValidationError):
        TelcoCustomerRow(
            customerID="X",
            gender="M",
            SeniorCitizen=5,  # Invalid: must be 0 or 1
            Partner="Y",
            Dependents="N",
            tenure=-1,  # Invalid: must be >= 0
            PhoneService="Y",
            MultipleLines="Y",
            InternetService="X",
            OnlineSecurity="X",
            OnlineBackup="X",
            DeviceProtection="X",
            TechSupport="X",
            StreamingTV="X",
            StreamingMovies="X",
            Contract="X",
            PaperlessBilling="X",
            PaymentMethod="X",
            MonthlyCharges=-10,  # Invalid: must be >= 0
            TotalCharges="0",
            Churn="X",
        )
