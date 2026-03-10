# Executive Summary & Technical Roadmap: Telecom Customer Churn MLOps

## 1. Project Overview
This project builds an industrial-grade **Agentic MLOps** platform to predict and prevent customer churn in telecommunications. By merging **quantitative behavioral data** with **qualitative AI-generated sentiment**, the system provides a holistic risk profile for every customer. 

The architecture follows the **"Brain vs. Brawn"** model:
- **Brain (Agent):** Orchestrates data enrichment and qualitative interpretation.
- **Brawn (ML):** Executes deterministic, high-scale churn propensity scoring.

---

## 2. Technical Roadmap: Phase-by-Phase

### Phase 1: Project Scaffolding & Environment (DONE ✅)
- [x] Configure `pyproject.toml` with `uv` dependency management.
- [x] Set up modular `src/` structure (`api`, `components`, `pipeline`, `utils`).
- [x] Implement `ConfigurationManager` for YAML-based artifact/path orchestration.
- [x] Define **Pydantic Data Contracts** for the raw Telco dataset.
- [x] Verify environment with automated import and validation smoke tests.

### Phase 2: Data Enrichment — Synthetic Ticket Notes (AI Agent)
- [ ] Implement `src/enrichment/prompts/` with versioned system prompts.
- [ ] Build the **Enrichment Tool**: A Gemini-powered generator that creates unique customer interaction logs based on row-level features (e.g., tenure, service type).
- [ ] Orchestrate the **Agentic Row Iterator**: An agent that processes batches, validates LLM output against a schema, and handles retries/failures.
- [ ] Implement **Sentiment Classifier**: A secondary NLP pass to categorize generated notes.

### Phase 3: Data Validation (GX) — Phase 1 & 2
- [ ] Setup **Great Expectations (GX)** v1.0 data context.
- [ ] Create **Expectation Suites** for both raw and enriched feature sets.
- [ ] Implement `data_validation.py` component to generate data quality reports (HTML checkpoint).
- [ ] Block the pipeline if schema drift or data quality thresholds are not met.

### Phase 4: NLP Engineering & Feature Store (DVC)
- [ ] Implement **Vector Embedding Generator**: Convert ticket notes into 384-dim (or 768-dim) vectors using `sentence-transformers`.
- [ ] Execute **Dimensionality Reduction** (PCA/UMAP) or Feature Selection on NLP vectors.
- [ ] Merge NLP features with structured usage features.
- [ ] **Data Versioning (DVC)**: Commit the enriched, vectorized dataset to the DVC registry.

### Phase 5: Model Development & Experiment Tracking (MLflow)
- [ ] Implement **Data Transformation Component**: Sklearn pipeline for scaling, encoding, and handling class imbalance (SMOTE).
- [ ] Conduct **Hyperparameter Optimization** using **Optuna** (searching XGBoost, LightGBM, and Random Forest).
- [ ] **MLflow Tracking**: Log all runs, Recall (Primary), F1 metrics, ROC-AUC, confusion matrices, and feature importance artifacts. Evaluate the recall-precision trade-off based on retention campaign costs.
- [ ] Serialize the "Best Model" and log it to the **Model Registry**.

### Phase 6: Inference Pipeline (FastAPI)
- [ ] Build `src/api/` backbone using **FastAPI**.
- [ ] Implement **Prediction Endpoint**: Handles single customer JSON input, generates real-time note embeddings, and returns a churn risk score + sentiment class.
- [ ] Implement **Batch Prediction Endpoint**: Processes bulk CSV files.
- [ ] Add **Inference Guardrails**: Validates request data before model feeding.

### Phase 7: UI Development & Containerization
- [ ] Develop **Gradio Dashboard**: Interactive risk calculator with SHAP/Feature Importance visualizations for end-user interpretability.
- [ ] Write a production-ready **Dockerfile** (multi-stage build for minimal size).
- [ ] Implement `docker-compose.yaml` for local orchestration of the API, UI, and MLflow server.

### Phase 8: CI/CD & Cloud Deployment (AWS)
- [ ] Configure **GitHub Actions** for:
    - Code linting (`ruff`).
    - Unit testing (`pytest`).
    - Docker image build and push to **AWS ECR**.
- [ ] Define **AWS ECS Fargate** Task Definitions and Service.
- [ ] Deploy the Gradio UI and API backend to the cloud.

### Phase 9: Monitoring & Tracing
- [ ] Integrate **OpenTelemetry (OTel)** tracing for Agentic workflows.
- [ ] Monitor LLM token usage and latency metrics.

---

## 3. Core Value Drivers
1. **Explainable AI:** Qualitative notes tell the "story" that raw numbers miss.
2. **Industrial Scalability:** Decoupled FTI (Feature-Training-Inference) architecture.
3. **Agentic Integration:** Directing LLMs to perform specialized, validated tasks within a larger deterministic pipeline.
