# Project Charter: Telecom Customer Churn MLOps (Agentic Stack)

This document outlines the high-level roadmap and architectural philosophy for building a production-ready, AI-driven churn prediction system.

---

### 1. What exactly am I going to build?
I am building a **fully automated, end-to-end MLOps ecosystem** designed to predict customer churn for a telecommunications company. 

It is a hybrid system that follows the **"Brain vs. Brawn"** architecture:
- **The Brain (Agentic AI):** Uses an LLM agent to enrich structured customer data with synthetic qualitative insights (e.g., realistic ticket notes and sentiment analysis).
- **The Brawn (Deterministic ML):** Uses traditional Gradient Boosted Models (XGBoost/LightGBM) to handle quantitative prediction at scale.
- **The Infrastructure:** A production-grade pipeline including data validation, experiment tracking, containerized serving, and cloud deployment.

### 2. Who’s it intended for?
- **Corporate Retention Teams:** Stakeholders who need to identify at-risk customers and understand the "why" behind the risk.
- **MLOps Engineers/Tech Leads:** As a portfolio showcase (flagship project) demonstrating the ability to bridge AI Agents with traditional ML in a modern CI/CD environment.

### 3. What problem does the system/app solve?
Traditional churn models often rely solely on structured data (usage, billing, tenure). While accurate, they miss the **nuanced sentiment** found in customer complaints or interaction logs—data that is often sparse or missing in public datasets. 
This system solves:
1. **Data Sparsity:** Using AI to generate high-fidelity synthetic notes to augment features.
2. **Production Gap:** Moving beyond Jupyter notebooks into a modular, containerized, and cloud-deployed microservice.
3. **Data Quality:** Preventing "garbage in, garbage out" via automated data contracts (GX).

### 4. How is it going to work?
The project follows the **FTI (Feature, Training, Inference)** design pattern:
1. **Feature Pipeline (Data Engineering):** Ingest raw CSV -> AI Agent generates synthetic notes -> NLP tools generate embeddings/summaries -> DVC versions the artifacts.
2. **Training Pipeline (Model Development):** Load versioned data -> Optuna hyperparameter tuning -> MLflow logs experiments, metrics, and models.
3. **Inference Pipeline (Model Serving):** The best model is served via a FastAPI microservice. A Gradio UI provides a dashboard for real-time predictions and customer risk assessment.

### 5. What is the expected result?
- A **live, cloud-deployed dashboard** (Gradio) on AWS ECS Fargate.
- A **99% reproducible pipeline** where a single command can retrain and redeploy the system.
- A **Feature Store** mindset using DVC to track data lineage.
- **Recall-First Evaluation Strategy**: Prioritizing the detection of every potential churner to minimize expensive False Negatives.
- **Observable AI Agent workflows** using Tracing.

### 6. What steps do I need to take to achieve this result?
1. **Phase 1: Project Scaffolding** - Environment, path constants, and core utilities.
2. **Phase 2: Data Enrichment Agent** - Build the AI Agent that generates synthetic ticket notes.
3. **Phase 3: Data Validation (GX) & EDA** - Formulate data contracts and analyze correlations.
4. **Phase 4: ML Engineering** - Feature engineering (NLP + Structured) and DVC versioning.
5. **Phase 5: Automated Training** - Optuna tuning + MLflow tracking.
6. **Phase 6: Model Serving** - Build the FastAPI backend ("Inference Microservice").
7. **Phase 7: Frontend & Containerization** - Develop the Gradio UI and Dockerize the stack.
8. **Phase 8: CI/CD & Deployment** - GitHub Actions to AWS ECS Fargate.

### 7. What could go wrong along the way?
- **API Costs:** High token usage for enriching 7,000+ rows with LLM notes.
- **Data Leakage:** AI-generated notes might accidentally contain "clues" about the target variable (Churn) that wouldn't exist in reality.
- **Environment Drift:** Complexity of local vs. cloud environments (Docker mitigates this).
- **Latency:** Real-time NLP embedding generation might slow down the inference endpoint.

### 8. What tools should I use to develop this project?
- **Dependency Management:** `uv` (Lightning-fast Python package installer).
- **Agentic Orchestration:** LangGraph + Google Gemini 2.0 Flash.
- **ML & Data:** Pandas, Scikit-learn, XGBoost, Optuna.
- **MLOps Foundations:** MLflow (Tracking), DVC (Versioning), Great Expectations (Quality).
- **Deployment:** FastAPI (API), Gradio (UI), Docker (Containers), GitHub Actions (CI/CD), AWS (Cloud).

### 9. What are the main concepts that are involved and how are they related?
1. **Separation of Concerns:** Keep Agent logic (probabilistic) separate from Tools/ML models (deterministic).
2. **Data Lineage:** Using DVC to ensure every model version is tied to a specific data version.
3. **Agentic Healing:** Using the AI Agent to validate its own outputs and handle edge cases in the data.
4. **Modular MLOps:** Each stage (Feature, Training, Inference) is an independent operational pipeline, allowing teams to scale components separately.
