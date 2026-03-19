# Executive Summary & Technical Roadmap: Telecom Customer Churn MLOps

## 1. Project Overview
This project builds an industrial-grade **Agentic MLOps** platform to predict and prevent customer
churn in telecommunications. By merging **quantitative behavioral data** with
**qualitative AI-generated sentiment**, the system provides a holistic risk profile for every customer.

The architecture follows the **"Brain vs. Brawn"** model:
- **Brain (Agent):** Orchestrates data enrichment and qualitative interpretation.
- **Brawn (ML):** Executes deterministic, high-scale churn propensity scoring.

---

## 2. Technical Roadmap: Phase-by-Phase

### Phase 1: Project Scaffolding, Environment & Data Ingestion (DONE ✅)
- [x] Configure `pyproject.toml` with `uv` dependency management.
- [x] Set up modular `src/` structure (`api`, `components`, `pipeline`, `utils`).
- [x] Implement `ConfigurationManager` for centralized YAML-based artifact/path orchestration.
- [x] Define **Pydantic Data Contracts** (`TelcoCustomerRow`) for the raw Telco dataset.
- [x] Verify environment with automated import and validation smoke tests (`tests/test_pydantic_entities.py`).
- [x] Implement **Data Ingestion** component and pipeline stage to sync external URL/local data safely to DVC-tracked `artifacts/data_ingestion`.

### Phase 2: Data Enrichment — Synthetic Ticket Notes (AI Agent) (DONE ✅)
- [x] Define versioned system prompts in `src/components/data_enrichment/prompts.py`.
- [x] Build the **Enrichment Generator**: A Gemini 2.0 Flash-powered tool using `pydantic-ai`
      that creates unique customer interaction logs based on row-level features.
- [x] Enforce I/O contracts with `CustomerInputContext` and `SyntheticNoteOutput` Pydantic schemas.
- [x] Orchestrate the **Async Batch Processor** `EnrichmentOrchestrator` with retries,
      deterministic fallback, and progressive CSV writes.
- [x] Implement **Sentiment Classifier**: Validated through structured JSON outputs
      yielding a `primary_sentiment_tag`.
- [x] Parameterize enrichment (`model_name`, `limit`) via `params.yaml` tracked by DVC.
- [x] Write unit tests for all enrichment schemas (`tests/test_enrichment.py`).
- [x] Refactor to **Components / Pipeline** separation (`src/components/` vs `src/pipeline/`).

### Phase 3: Data Validation (GX) — Raw & Enriched (DONE ✅)
- [x] Implement `DataValidator` component using **Great Expectations v1.0+** ephemeral context.
- [x] Create `raw_telco_churn_suite`: Column presence, tenure range, categorical value sets.
- [x] Create `enriched_telco_churn_suite`: Ticket note presence/length, sentiment tag consistency.
- [x] Integrate `schema.yaml` as the authoritative column definition for both suites.
- [x] Implement `StatisticalContractViolation` custom exception with full `DataQualityContext`.
- [x] Register all three stages in `dvc.yaml` with proper dependency tracking.
- [x] Generate hard validation artifacts (`status.txt`, `validation_report.json`) tracked in `dvc.yaml` to serve as quality gates.

### Phase 4: NLP Engineering & Feature Store (DVC) (DONE ✅)
- [x] Implement **Vector Embedding Generator**: Convert ticket notes into 384-dim vectors
      using `sentence-transformers`.
- [x] Execute **Dimensionality Reduction** (PCA) on NLP vectors for efficient storage and training.
- [x] Merge NLP features with structured usage features into a unified feature matrix.
- [x] Register the `feature_engineering` stage in `dvc.yaml` with automated artifact tracking.
- [x] Implement **Anti-Skew Alignment**: Ensured identical transformation and index alignment for Training, Validation, and Test sets.
- [x] Unit Test Suite for Cleaners, Embedders, and Split Logic (`tests/test_feature_engineering.py`).

### Phase 5: Model Development & Experiment Tracking (MLflow)
- [ ] Train a baseline model using just the structured data and the sentiment tags (ignoring the full text embeddings for now). This will serve as a benchmark for comparison with the final model.
- [ ] Implement **Data Transformation Component**: Sklearn pipeline for scaling, encoding, and handling class imbalance (SMOTE).
- [ ] Conduct **Hyperparameter Optimization** using **Optuna** (XGBoost, LightGBM, Random Forest).
- [ ] **MLflow Tracking**: Log all runs, Recall (Primary), F1 metrics, ROC-AUC, confusion matrices, and feature importance artifacts.
- [ ] Serialize the "Best Model" and log it to the **Model Registry**.

### Phase 6: Inference Pipeline (FastAPI)
- [ ] Build `src/api/` backbone using **FastAPI**.
- [ ] Implement **Prediction Endpoint**: Handles single customer JSON input, generates
      real-time note embeddings, and returns a churn risk score + sentiment class.
- [ ] Implement **Batch Prediction Endpoint**: Processes bulk CSV files.
- [ ] Add **Inference Guardrails**: Validates request data before model feeding.

### Phase 7: UI Development & Containerization
- [ ] Develop **Gradio Dashboard**: Interactive risk calculator with SHAP/Feature Importance
      visualizations for end-user interpretability.
- [ ] Write a production-ready **Dockerfile** (multi-stage build for minimal size).
- [ ] Implement `docker-compose.yaml` for local orchestration of the API, UI, and MLflow server.

### Phase 8: CI/CD & Cloud Deployment (AWS)
- [ ] Configure **GitHub Actions** for:
    - Code linting (`ruff`).
    - Unit testing (`pytest`) with coverage gate.
    - Docker image build and push to **AWS ECR**.
- [ ] Define **AWS ECS Fargate** Task Definitions and Service.
- [ ] Deploy the Gradio UI and API backend to the cloud.

### Phase 9: Monitoring & Tracing
- [ ] Integrate **OpenTelemetry (OTel)** tracing for Agentic workflows.
- [ ] Monitor LLM token usage and latency metrics.
- [ ] Implement Plan Success Rate (PSR) and Tool Call Accuracy (TCA) dashboards.

---

## 3. Architecture Deep Dives & Decisions

| Topic | Document |
|---|---|
| Overall System & FTI Pattern | [architecture.md](../architecture/architecture.md) |
| Phase 0: Data Ingestion | [data_ingestion.md](../architecture/data_ingestion.md) |
| Phase 2: Agentic Data Enrichment | [data_enrichment.md](../architecture/data_enrichment.md) |
| Phase 3: Great Expectations Validation | [data_validation_gx.md](../architecture/data_validation_gx.md) |
| Phase 4: NLP & Feature Engineering | [feature_engineering.md](../architecture/feature_engineering.md) |
| **Decision: Data Quality Checker** | [data_quality_checker.md](../decisions/data_quality_checker.md) |
| DVC Pipeline DAG | [dvc_pipeline.md](../architecture/dvc_pipeline.md) |
| Test Suite Coverage | [test_suite.md](../runbooks/test_suite.md) |

---

## 4. Core Value Drivers
1. **Explainable AI:** Qualitative notes tell the "story" that raw numbers miss.
2. **Industrial Scalability:** Decoupled FTI (Feature-Training-Inference) architecture.
3. **Agentic Integration:** Directing LLMs to perform specialized, validated tasks within a larger deterministic pipeline.
4. **Reproducibility:** Every pipeline run is fully parameterized and versioned via DVC + params.yaml.
