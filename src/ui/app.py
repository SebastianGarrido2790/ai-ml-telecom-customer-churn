"""
Main Entry point for the Gradio Dashboad.

This script aggregates the different UI tabs (Single Predict, Batch Predict,
and Experiment Comparison) into a single Gradio Blocks application. It
applies a consistent theme and serves the dashboard on a configurable port.
"""

import gradio as gr

from src.ui.pages.batch_predict import create_batch_predict_tab
from src.ui.pages.run_comparison import create_run_comparison_tab
from src.ui.pages.single_predict import create_single_predict_tab
from src.utils.logger import get_logger

logger = get_logger(__name__)


def build_app() -> gr.Blocks:
    """Builds and orchestrates the main Gradio Blocks application.

    Initializes the UI theme, sets up the header documentation, and
    assembles all modular tabs (pages) into the main dashboard.

    Returns:
        A compiled Gradio Blocks application object.
    """
    theme = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="blue",
        neutral_hue="slate",
    ).set(
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
    )

    with gr.Blocks(title="AI/ML Telecom Customer Churn Dashboard", theme=theme) as app:
        gr.Markdown("# 📊 AI/ML Telecom Customer Churn Dashboard")
        gr.Markdown(
            "An interactive Agentic Data Science workflow testing the Late Fusion model "
            "with circuit-breaker fallback to NLP. Use the tabs below to predict single customers, "
            "process bulk datasets, and view experiment metrics."
        )

        create_single_predict_tab()
        create_batch_predict_tab()
        create_run_comparison_tab()

    return app


if __name__ == "__main__":
    logger.info("Initializing Gradio UI...")
    app = build_app()
    # Port 7860 is exposed in the Dockerfile
    app.launch(server_name="0.0.0.0", server_port=7860)
