"""
Batch Prediction Page for the Gradio UI.

This module provides the logic and layout for uploading a CSV file of
customer feature data for bulk scoring. It interfaces with the multi-predict
endpoint of the Prediction API and presents results in a tabular format.
"""

from typing import Any

import gradio as gr
import pandas as pd

from src.ui.data_loaders.api_client import predict_batch


def process_batch(file_path: str | None) -> tuple[pd.DataFrame, str]:
    """Reads a CSV file and sends customer records to the prediction API for batch scoring.

    Args:
        file_path: The local path to the uploaded CSV file.

    Returns:
        A tuple containing a pandas DataFrame with the batch results and
        a summary status string indicating success or failure.
    """
    if not file_path:
        return pd.DataFrame(), "Please upload a CSV file."

    try:
        df = pd.read_csv(file_path)
        if "ticket_note" not in df.columns:
            df["ticket_note"] = "No note provided."
    except Exception as e:
        return pd.DataFrame(), f"Failed to read CSV: {e}"

    # Convert dataframe to records
    records: list[dict[str, Any]] = df.to_dict(orient="records")

    # Send to the API
    response = predict_batch(records)

    if "error" in response:
        return pd.DataFrame(), f"API Error: {response['error']}"

    nlp_available = response.get("nlp_branch_available", False)
    nlp_status = "✅ NLP Active" if nlp_available else "⚠️ NLP Fallback Used"

    predictions = response.get("predictions", [])

    # Build dataframe for UI
    result_records = []
    for pred in predictions:
        result_records.append(
            {
                "Customer ID": pred.get("customerID", "N/A"),
                "Churn Prediction": "Yes" if pred.get("churn_prediction") else "No",
                "Probability": round(pred.get("churn_probability", 0), 4),
                "Structured Branch": round(pred.get("p_structured", 0), 4),
                "NLP Branch": round(pred.get("p_nlp", 0), 4),
            }
        )

    result_df = pd.DataFrame(result_records)
    return result_df, f"Scored {response.get('total', 0)} customers. Status: {nlp_status}"


def create_batch_predict_tab() -> None:
    """Creates the 'Batch Prediction' tab in the Gradio interface.

    Defines the file upload component, the action button, and the
    output dataframe table and status indicator.
    """
    with gr.Tab("Batch Prediction"):
        gr.Markdown("### Upload a CSV for bulk processing")

        with gr.Row():
            file_upload = gr.File(label="Upload Customer Data (CSV)", file_types=[".csv"])
            status_text = gr.Textbox(label="Batch Status")

        with gr.Row():
            btn = gr.Button("Process Batch", variant="primary")

        gr.Markdown("---")
        results_table = gr.Dataframe(label="Results", interactive=False)

        btn.click(fn=process_batch, inputs=[file_upload], outputs=[results_table, status_text])
