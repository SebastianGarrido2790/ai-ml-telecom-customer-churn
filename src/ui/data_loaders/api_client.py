"""
API Client for the Gradio UI.

This module provides utility functions to communicate with the Prediction API
microservice. It handles both single and batch prediction requests, as well
as health checks. The API URL is configurable via the `PREDICTION_API_URL`
environment variable, defaulting to localhost for development.
"""

import logging
import os
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Fallback to localhost if run outside docker
API_URL = os.environ.get("PREDICTION_API_URL", "http://localhost:8000")


def predict_single(customer_data: dict[str, Any]) -> dict[str, Any]:
    """Sends a single customer feature record to the prediction API.

    Args:
        customer_data: A dictionary containing raw customer features and the
            ticket_note string required by the Late Fusion model.

    Returns:
        A dictionary containing the prediction results (churn probability,
        prediction flag, branch specifics) or an error message.
    """
    try:
        response = httpx.post(f"{API_URL}/v1/predict", json=customer_data, timeout=10.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"API Error in predict_single: {e}")
        return {"error": str(e)}


def predict_batch(customers_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Sends a list of customer records for bulk scoring at the prediction API.

    Args:
        customers_list: A list of customer feature dictionaries to be scored.

    Returns:
        A dictionary containing a list of predictions for the entire batch
        along with metadata about the total count and NLP branch availability.
    """
    try:
        payload = {"customers": customers_list}
        response = httpx.post(f"{API_URL}/v1/predict/batch", json=payload, timeout=30.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPError as e:
        logger.error(f"API Error in predict_batch: {e}")
        return {"error": str(e)}


def check_health() -> bool:
    """Checks the health and connectivity of the Prediction API microservice.

    Returns:
        True if the API returns a 200 OK status, False otherwise.
    """
    try:
        response = httpx.get(f"{API_URL}/v1/health", timeout=5.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False
