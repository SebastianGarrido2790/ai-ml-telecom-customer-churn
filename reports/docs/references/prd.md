# Product Requirements Document (PRD): Telecom Customer Churn MLOps

## 1. Executive Summary
The **Telecom Customer Churn MLOps** system is a production-grade predictive platform designed to identify at-risk telecommunications customers. Unlike standard ML models that rely solely on usage statistics, this platform integrates **Agentic AI** to synthesize qualitative customer interactions, providing a deeper understanding of churn drivers through sentiment analysis and automated feature enrichment.

## 2. Goals & Objectives
*   **High-Accuracy Prediction:** Implement a robust classification model (XGBoost/LightGBM) to predict churn propensity.
*   **Agentic Data Enrichment:** Use LLM agents to bridge data gaps by generating high-fidelity synthetic ticket notes.
*   **Operational Excellence:** Deploy a fully automated MLOps pipeline using the FTI (Feature, Training, Inference) pattern.
*   **Business Insight:** provide a user-friendly dashboard for churn risk assessment and sentiment interpretability.

## 3. Scope & Targeted Users
*   **Customer Retention Teams:** primary users searching for individual customer risk scores and sentiment context.
*   **Data Science / MLOps Teams:** secondary users monitoring pipeline health, data drift, and model performance.

## 4. Functional Requirements

### 4.1 Data Pipeline (Feature & Validation)
*   **Automated Ingestion:** The system must ingest raw structured Telco data automatically.
*   **Data Contracts:** Enforce input/output schemas using **Great Expectations (GX)** to prevent pipeline corruption.
*   **AI Enrichment:** An AI agent must generate a "ticket_note" and sentiment category for every customer based on their demographic and usage features.
*   **NLP Feature Engineering:** The system must generate vector embeddings from these ticket notes for use in the ML model.

### 4.2 Model Training
*   **Experiment Tracking:** Record all runs, hyperparameters (tuned via **Optuna**), and metrics (Recall, Precision, F1-Score) in **MLflow**.
*   **Data Versioning:** Use **DVC** to track the exact dataset state for every trained model version.
*   **Model Registry:** Tag versions as `staging`, `production`, or `archived`.

### 4.3 Inference & Serving
*   **REST API:** Expose a **FastAPI** endpoint for real-time risk scoring.
*   **Batch Prediction:** Capability to process scheduled batch jobs on new customer data.
*   **Dashboard (UI):** A **Gradio** application allowing users to input customer IDs or manually input features to see a risk report.

## 5. Non-Functional Requirements
*   **Reproducibility:** The entire stack must be deployable with a single command via Docker.
*   **Scalability:** Decoupled architecture (FTI) must allow scaling the Inference microservice independently of Training.
*   **Observability:** Implement **Tracing** (OpenTelemetry) to visualize the AI Agent's thought process and tool usage.
*   **Security:** API keys (Google AI, AWS) must be handled via environment variables, not committed to code.

## 6. Technical Architecture (The Agentic Stack)
| Component | Technology |
|---|---|
| **Programming Language** | Python 3.11+ |
| **Dependency Management** | `uv` |
| **"Brain" (Agent)** | LangGraph + Google Gemini 2.0 Flash |
| **"Brawn" (Model)** | XGBoost / LightGBM |
| **MLOps Pipeline** | MLflow, DVC, Great Expectations (GX) |
| **API Framework** | FastAPI |
| **UI Framework** | Gradio |
| **Containerization** | Docker |
| **Deployment** | AWS ECS Fargate + GitHub Actions |

## 7. Success Metrics
*   **Technical:** F1-Score > 0.80 on the test set.
*   **Operational:** 100% CI/CD pass rate; < 500ms API latency.
*   **Business:** Clearly demonstrable patterns of sentiment-to-churn correlation in synthetic data.

## 8. Risks & Constraints
*   **Cost Management:** Monitor token usage for batch LLM enrichment.
*   **Model Fairness:** Review synthetic data for potential bias in generated customer profiles.
*   **Deployment Complexity:** Ensure cloud IAM roles and ECS configurations are well-documented.
