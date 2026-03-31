"""
SHAP Explanation Component for the Gradio UI.

This module provides a function to generate a SHAP waterfall plot for
individual customer churn predictions. It loads the necessary model
and preprocessing artifacts from the `artifacts/` directory to
explain the structured branch (Branch 1) of the Late Fusion model.
"""

import os
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.utils.logger import get_logger

# Configure logging
logger = get_logger(__name__, headline="shap_chart.py")

# Configuration paths for loading artifacts
FEATURE_ENG_DIR = Path(os.environ.get("FEATURE_ENG_DIR", "artifacts/feature_engineering"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "artifacts/model_training"))

_structured_preprocessor = None
_structured_model = None


def _load_artifacts() -> None:
    """Loads required model and preprocessing artifacts from the file store.

    This function lazily loads the structured preprocessor and the XGBoost
    structured model and caches them as module-level globals for reused
    shaping requests.
    """
    global _structured_preprocessor, _structured_model
    if _structured_preprocessor is None:
        try:
            _structured_preprocessor = joblib.load(FEATURE_ENG_DIR / "structured_preprocessor.pkl")
        except Exception as exc:
            logger.warning(f"Failed to load structured_preprocessor: {exc!s}")
            _structured_preprocessor = None

    if _structured_model is None:
        try:
            _structured_model = joblib.load(MODEL_DIR / "structured_model.pkl")
        except Exception as exc:
            logger.warning(f"Failed to load structured_model: {exc!s}")
            _structured_model = None


def get_shap_plot(customer_data: dict[str, Any]) -> plt.Figure | None:
    """Generates a SHAP waterfall chart for a single customer prediction.

    Reconstructs the raw feature DataFrame, applies the structured
    preprocessing pipeline, and computes SHAP values using the Branch 1
    XGBoost classifier. The output is a matplotlib figure object.

    Args:
        customer_data: A dictionary containing the raw features of a single customer
            as provided in the predict request.

    Returns:
        A matplotlib Figure object containing the SHAP waterfall chart,
        or None if the required artifacts cannot be loaded.
    """
    _load_artifacts()

    if _structured_preprocessor is None or _structured_model is None:
        # Artifacts might not be available
        return None

    # Reconstruct raw DataFrame matching the training layout exactly
    structured_raw_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ]

    # Extract only the needed fields
    row = {k: customer_data.get(k, "") for k in structured_raw_cols}
    if "TotalCharges" in customer_data and customer_data["TotalCharges"] is None:
        row["TotalCharges"] = ""

    df = pd.DataFrame([row], columns=structured_raw_cols)

    # Preprocess
    # structured_preprocessor from feature_engineering yields a pandas dataframe
    # if using set_output(transform="pandas"), which our system does.
    x_processed = _structured_preprocessor.transform(df)

    # Explainer
    explainer = shap.TreeExplainer(_structured_model)
    shap_values = explainer(x_processed)

    # Generate matplotlib figure
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    return fig
