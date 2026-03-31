"""
Experiment Comparison Page for the Gradio UI.

This module provides the logic and layout for comparing MLflow experiment
metrics across different model architectures (Structured, NLP, and Late Fusion).
It reads results directly from the `evaluation_report.json` generated during
the Training Pipeline.
"""

import json
import logging
import os
from pathlib import Path

import gradio as gr
import pandas as pd

logger = logging.getLogger(__name__)

EVAL_REPORT_PATH = Path(os.environ.get("EVAL_REPORT_PATH", "artifacts/model_training/evaluation_report.json"))


def _load_evaluation_report() -> pd.DataFrame:
    """Loads and parses the evaluation report JSON into a comparison DataFrame.

    Returns:
        A pandas DataFrame where each row compares a different iteration
        of the churn model (baselines vs Late Fusion) across metrics
        like Recall, Precision, and F1.
    """
    if not EVAL_REPORT_PATH.exists():
        logger.warning(f"Evaluation report not found at {EVAL_REPORT_PATH}")
        return pd.DataFrame([{"Error": "evaluation_report.json not found"}])

    try:
        with EVAL_REPORT_PATH.open() as f:
            report = json.load(f)

        # Parse comparison
        data = []
        for run_name in ["structured_baseline", "nlp_baseline", "late_fusion_stacked"]:
            if run_name in report:
                metrics = report[run_name].get("metrics", {})
                row = {
                    "Model Architecture": run_name.replace("_", " ").title(),
                    "Recall": round(metrics.get("recall", 0), 4),
                    "Precision": round(metrics.get("precision", 0), 4),
                    "F1 Score": round(metrics.get("f1", 0), 4),
                    "ROC AUC": round(metrics.get("roc_auc", 0), 4),
                }

                # Add lifts if applicable
                if run_name == "late_fusion_stacked":
                    row["Recall Lift vs Baseline"] = round(report[run_name].get("recall_lift", 0), 4)
                    row["F1 Lift vs Baseline"] = round(report[run_name].get("f1_lift", 0), 4)
                else:
                    row["Recall Lift vs Baseline"] = "—"
                    row["F1 Lift vs Baseline"] = "—"

                data.append(row)

        return pd.DataFrame(data)
    except Exception as e:
        logger.error(f"Failed to parse evaluation report: {e}")
        return pd.DataFrame([{"Error": f"Failed to parse report: {e}"}])


def create_run_comparison_tab() -> None:
    """Creates the 'Experiment Tracking & Runs' tab in the Gradio interface.

    Loads the evaluation comparison table and provides a reference link
    to the main MLflow tracking server for detailed audit logs.
    """
    with gr.Tab("Experiment Tracking & Runs"):
        gr.Markdown("### MLflow Reference Architecture: Champion vs Challenger")
        gr.Markdown(
            "The table below reads directly from `evaluation_report.json` generated "
            "during the DVC training pipeline. It tracks the uplift of Late Fusion "
            "against the standard baselines."
        )

        df = _load_evaluation_report()
        gr.Dataframe(value=df, interactive=False)

        gr.Markdown("---")
        gr.Markdown(
            "Access the full detailed MLflow Dashboard over at the `mlflow-server` container "
            "running on port 5000: [http://localhost:5000](http://localhost:5000/)."
        )
