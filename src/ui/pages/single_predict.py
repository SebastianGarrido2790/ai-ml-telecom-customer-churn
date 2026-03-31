"""
Single Prediction Page for the Gradio UI.

This module defines the UI layout and logic for the 'Single Prediction' tab.
It allows users to manually enter customer feature data and support ticket
notes to toggle the Late Fusion model's dual-branch inference. It also
integrates local SHAP explanations for transparency.
"""

from typing import Any

import gradio as gr
import matplotlib.pyplot as plt

from src.ui.components.shap_chart import get_shap_plot
from src.ui.data_loaders.api_client import predict_single


def create_single_predict_tab() -> None:
    """Creates the 'Single Prediction' tab in the Gradio interface.

    This function defines the layout, including demographic, service,
    and billing input fields, along with the action button and result
    display (probabilities, status badges, and SHAP plots).
    """
    with gr.Tab("Single Prediction"):
        gr.Markdown("### Enter Customer Details for Late Fusion Churn Prediction")

        with gr.Row():
            # Column 1: Demographics & Account info
            with gr.Column():
                customer_id_val = gr.Textbox(label="Customer ID (Optional)")
                gender_val = gr.Radio(["Female", "Male"], label="Gender", value="Female")
                senior_val = gr.Radio(["0", "1"], label="Senior Citizen", value="0")
                partner_val = gr.Radio(["No", "Yes"], label="Partner", value="No")
                dependents_val = gr.Radio(["No", "Yes"], label="Dependents", value="No")
                tenure_val = gr.Number(label="Tenure (Months)", value=0, precision=0)

            # Column 2: Service Info
            with gr.Column():
                phone_val = gr.Radio(["No", "Yes"], label="Phone Service", value="Yes")
                multiple_val = gr.Dropdown(["No phone service", "No", "Yes"], label="Multiple Lines", value="No")
                internet_val = gr.Dropdown(["DSL", "Fiber optic", "No"], label="Internet Service", value="DSL")
                security_val = gr.Dropdown(["No internet service", "No", "Yes"], label="Online Security", value="No")
                backup_val = gr.Dropdown(["No internet service", "No", "Yes"], label="Online Backup", value="No")
                protection_val = gr.Dropdown(
                    ["No internet service", "No", "Yes"], label="Device Protection", value="No"
                )
                support_val = gr.Dropdown(["No internet service", "No", "Yes"], label="Tech Support", value="No")
                tv_val = gr.Dropdown(["No internet service", "No", "Yes"], label="Streaming TV", value="No")
                movies_val = gr.Dropdown(["No internet service", "No", "Yes"], label="Streaming Movies", value="No")

            # Column 3: Billing & NLP Note
            with gr.Column():
                contract_val = gr.Dropdown(
                    ["Month-to-month", "One year", "Two year"], label="Contract", value="Month-to-month"
                )
                paperless_val = gr.Radio(["No", "Yes"], label="Paperless Billing", value="Yes")
                payment_val = gr.Dropdown(
                    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                    label="Payment Method",
                    value="Electronic check",
                )
                monthly_val = gr.Number(label="Monthly Charges ($)", value=0.0)
                total_val = gr.Textbox(label="Total Charges ($)", value="0.0")
                ticket_note_val = gr.Textbox(
                    label="Customer Support Ticket Note (NLP Branch)",
                    lines=3,
                    placeholder="E.g., Customer is very frustrated with recent price hikes and slow speeds.",
                )

        btn_predict = gr.Button("Predict Churn Risk", variant="primary")

        # Results Section
        gr.Markdown("---")
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Prediction Results")
                out_prob = gr.Number(label="Churn Probability")
                out_pred = gr.Textbox(label="Will Churn?", lines=1)

                gr.Markdown("#### Branch Details")
                out_p_struct = gr.Number(label="Structured Branch Prob")
                out_p_nlp = gr.Number(label="NLP Branch Prob")
                out_nlp_badge = gr.Textbox(label="NLP Branch Status", lines=1)

            with gr.Column(scale=2):
                gr.Markdown("### SHAP Feature Importance")
                shap_plot = gr.Plot()

        def _handle_predict(
            cid: str,
            gen: str,
            sen: str,
            pat: str,
            dep: str,
            ten: float,
            pho: str,
            mul: str,
            int_srv: str,
            sec: str,
            bac: str,
            pro: str,
            sup: str,
            tv: str,
            mov: str,
            con: str,
            pap: str,
            pay: str,
            mon: float,
            tot: str,
            t_note: str,
        ) -> tuple[float | None, str, float | None, float | None, str, plt.Figure | None]:
            """Local handler for the prediction button click event.

            Orchestrates input validation, the API call to the prediction
            microservice, and the generation of local SHAP plots.

            Args:
                cid-t_note: Raw input values from the Gradio components.

            Returns:
                A tuple containing values for each Gradio output component
                (probability, prediction string, branch probabilities,
                NLP status, and SHAP figure).
            """
            senior_int = int(sen) if str(sen) in ["0", "1"] else 0

            payload: dict[str, Any] = {
                "customerID": cid if cid else None,
                "gender": gen,
                "SeniorCitizen": senior_int,
                "Partner": pat,
                "Dependents": dep,
                "tenure": int(ten),
                "PhoneService": pho,
                "MultipleLines": mul,
                "InternetService": int_srv,
                "OnlineSecurity": sec,
                "OnlineBackup": bac,
                "DeviceProtection": pro,
                "TechSupport": sup,
                "StreamingTV": tv,
                "StreamingMovies": mov,
                "Contract": con,
                "PaperlessBilling": pap,
                "PaymentMethod": pay,
                "MonthlyCharges": mon,
                "TotalCharges": tot if tot else None,
                "ticket_note": t_note if t_note else "No note provided",
            }

            # API call
            resp = predict_single(payload)
            if "error" in resp:
                return (None, f"Error: {resp['error']}", None, None, "Error", None)

            # SHAP locally
            fig = get_shap_plot(payload)

            nlp_available = resp.get("nlp_branch_available", False)
            nlp_status = "✅ Active" if nlp_available else "⚠️ Fallback (Service Unavailable)"

            return (
                resp.get("churn_probability"),
                "Yes" if resp.get("churn_prediction") else "No",
                resp.get("p_structured"),
                resp.get("p_nlp"),
                nlp_status,
                fig,
            )

        btn_predict.click(
            fn=_handle_predict,
            inputs=[
                customer_id_val,
                gender_val,
                senior_val,
                partner_val,
                dependents_val,
                tenure_val,
                phone_val,
                multiple_val,
                internet_val,
                security_val,
                backup_val,
                protection_val,
                support_val,
                tv_val,
                movies_val,
                contract_val,
                paperless_val,
                payment_val,
                monthly_val,
                total_val,
                ticket_note_val,
            ],
            outputs=[out_prob, out_pred, out_p_struct, out_p_nlp, out_nlp_badge, shap_plot],
        )
