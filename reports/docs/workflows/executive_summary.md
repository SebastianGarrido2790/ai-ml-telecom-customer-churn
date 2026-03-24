# Executive Summary & Technical Roadmap: Telecom Customer Churn MLOps

| **Version** | v1.1 |

## 1. Project Overview

This project builds an industrial-grade **Agentic MLOps** platform to predict and prevent customer churn in telecommunications. By merging **quantitative behavioral data** with **qualitative AI-generated sentiment**, the system provides a holistic risk profile for every customer.

The architecture follows the **"Brain vs. Brawn"** model:
- **Brain (Agent):** Orchestrates data enrichment, qualitative interpretation, and business reasoning.
- **Brawn (ML):** Executes deterministic, high-scale churn propensity scoring via a Late Fusion ensemble.

### 1.1 Architectural Enhancement (Post-Phase 4)

Following the completion of the Feature Pipeline, the project adopted a **dual-enhancement strategy** that elevates the system from a standard MLOps pipeline to a fully Agentic, microservice-native platform.

**Enhancement 1 — Late Fusion Ensemble (Option A):** Rather than training a single model on merged features, the system trains two independent base models on separate feature branches and combines them via a stacking meta-learner. This produces a quantifiable business narrative: the performance delta between the structured-only baseline and the fusion model directly proves the ROI of the entire Phase 2 AI enrichment investment.

**Enhancement 2 — Embedding Microservice (Option B):** The `TextEmbedder` + `PCA` transformer is extracted from the monolithic `preprocessor.pkl` and deployed as an independent FastAPI microservice. This enforces Rule 1.3 (Tools as Microservices), enables independent versioning and scaling of the NLP component, and allows the embedding model to be swapped without retraining the churn model.

Both enhancements share a single foundational prerequisite: the unified `preprocessor.pkl` produced in Phase 4 is split into two independently serialized artifacts, `structured_preprocessor.pkl` and `nlp_preprocessor.pkl`, each fitted exclusively on the training set per the Anti-Skew Mandate.

---

## 2. Technical Roadmap: Phase-by-Phase

### Phase 1: Project Scaffolding, Environment & Data Ingestion (DONE ✅)

- [x] Configure `pyproject.toml` with `uv` dependency management.
- [x] Set up modular `src/` structure (`api`, `components`, `pipeline`, `utils`).
- [x] Implement `ConfigurationManager` for centralized YAML-based artifact/path orchestration.
- [x] Define **Pydantic Data Contracts** (`TelcoCustomerRow`) for the raw Telco dataset.
- [x] Verify environment with automated import and validation smoke tests (`tests/unit/test_pydantic_entities.py`).
- [x] Implement **Data Ingestion** component and pipeline stage to sync external URL/local data safely to DVC-tracked `artifacts/data_ingestion`.

### Phase 2: Data Enrichment — Synthetic Ticket Notes (AI Agent) (DONE ✅)

- [x] Define versioned system prompts in `src/components/data_enrichment/prompts.py`.
- [x] Build the **Enrichment Generator**: A Gemini 2.0 Flash-powered tool using `pydantic-ai` that creates unique customer interaction logs based on row-level features.
- [x] Enforce I/O contracts with `CustomerInputContext` and `SyntheticNoteOutput` Pydantic schemas.
- [x] Orchestrate the **Async Batch Processor** `EnrichmentOrchestrator` with retries, deterministic fallback, and progressive CSV writes.
- [x] Implement **Sentiment Classifier**: Validated through structured JSON outputs yielding a `primary_sentiment_tag`.
- [x] Parameterize enrichment (`model_name`, `limit`) via `params.yaml` tracked by DVC.
- [x] Write unit tests for all enrichment schemas (`tests/unit/test_enrichment.py`).
- [x] Refactor to **Components / Pipeline** separation (`src/components/` vs `src/pipeline/`).

**Production Results:** 7,043 customers enriched. 0 schema violations. Sentiment distribution: Satisfied (68.6%), Frustrated (26.3%), Neutral (3.5%), Billing Inquiry (1.3%), Dissatisfied (0.3%), Technical Issue (0.1%).

### Phase 3: Data Validation (GX) — Raw & Enriched (DONE ✅)

- [x] Implement `DataValidator` component using **Great Expectations v1.0+** ephemeral context.
- [x] Create `raw_telco_churn_suite`: Column presence, tenure range, categorical value sets.
- [x] Create `enriched_telco_churn_suite`: Ticket note presence/length, sentiment tag consistency.
- [x] Integrate `schema.yaml` as the authoritative column definition for both suites.
- [x] Implement `StatisticalContractViolation` custom exception with full `DataQualityContext`.
- [x] Register all validation stages in `dvc.yaml` with proper dependency tracking.
- [x] Generate hard validation artifacts (`status.txt`, `validation_report.json`) tracked in `dvc.yaml` to serve as CI/CD quality gates.

### Phase 4: NLP Engineering & Feature Store (DVC) (DONE ✅ — Modified)

- [x] Implement **Vector Embedding Generator**: Convert ticket notes into 384-dim vectors using `sentence-transformers/all-MiniLM-L6-v2`.
- [x] Execute **Dimensionality Reduction** (PCA, 20 components) on NLP vectors.
- [x] Merge NLP features with structured usage features into a unified feature matrix.
- [x] Register the `feature_engineering` stage in `dvc.yaml` with automated artifact tracking.
- [x] Implement **Anti-Skew Alignment**: Identical transformations and index alignment across Training, Validation, and Test sets; preprocessor fitted on Train only.
- [x] Unit Test Suite for Cleaners, Embedders, and Split Logic (`tests/test_feature_engineering.py`).
- [x] **[MODIFICATION REQUIRED]** Split unified `preprocessor.pkl` into two independently serialized artifacts to enable Late Fusion training and the Embedding Microservice:
    - `structured_preprocessor.pkl` — Numeric + Categorical pipelines.
    - `nlp_preprocessor.pkl` — `TextEmbedder` + `PCA` pipeline.
    - Update `dvc.yaml` Stage 4 outputs accordingly.
    - Add `ModelTrainingConfig`, `EmbeddingServiceConfig`, and `PredictionAPIConfig` entities to `src/entity/config_entity.py`.
    - Extend `ConfigurationManager` with corresponding `get_*_config()` methods.
    - Add `model_training` and `api` sections to `config/config.yaml` and `params.yaml`.

### Phase 5: Late Fusion Model Development & Experiment Tracking (MLflow) (⏳)

The training pipeline implements a **Late Fusion stacking architecture** with three tracked experiment runs per cycle, producing an explicit and quantifiable ROI narrative for the AI enrichment work completed in Phase 2.

**Branch 1 — Structured Baseline:**
- [x] Load `structured_preprocessor.pkl` and apply to train/val/test structured feature columns.
- [x] Handle class imbalance with **SMOTE** applied exclusively on the training set.
- [x] Tune XGBoost hyperparameters via **Optuna** (30 trials).
- [x] Log run to MLflow: `branch=structured`, metrics: Recall, F1, ROC-AUC, confusion matrix, feature importance artifact.

**Branch 2 — NLP Baseline:**
- [x] Load `nlp_preprocessor.pkl` and apply to train/val/test NLP (PCA-reduced) columns.
- [x] Tune XGBoost/LightGBM hyperparameters via **Optuna** (20 trials).
- [x] Log run to MLflow: `branch=nlp`, metrics: Recall, F1, ROC-AUC.

**Late Fusion Meta-Learner:**
- [x] Generate **Out-of-Fold (OOF)** probability predictions from both base models using `cross_val_predict(method='predict_proba')` on the training set to prevent meta-learner leakage.
- [x] Train **Logistic Regression** meta-learner on the stacked OOF arrays.
- [x] Retrain both base models on the full training set.
- [x] Evaluate the stacked ensemble on the held-out test set.
- [x] Log run to MLflow: `branch=fusion`, custom metrics: `recall_lift` and `f1_lift` over structured baseline.
- [x] Serialize all three model artifacts: `structured_model.pkl`, `nlp_model.pkl`, `meta_model.pkl` to `artifacts/model_training/`.
- [x] Register the champion model in the **MLflow Model Registry** with tag `production`.

**DVC Integration:**
- [x] Register `stage_05_train_model` in `dvc.yaml` with both preprocessors and train/val/test CSVs.

### Phase 6: Inference Pipeline — Microservice Architecture (FastAPI) (PLANNED)

The inference layer deploys as **two decoupled FastAPI microservices**, enforcing Rule 1.3 (Tools as Microservices) and enabling independent scaling and versioning of each component.

**Embedding Microservice (`src/api/embedding_service/`):**
- [ ] Implement FastAPI app with `lifespan` startup loading of `nlp_preprocessor.pkl`.
- [ ] Expose `POST /v1/embed` accepting `EmbedRequest` (`ticket_notes: list[str]`) and returning `EmbedResponse` (`embeddings: list[list[float]]`, `model_version`, `dim`).
- [ ] Expose `GET /v1/health` for container readiness probes.
- [ ] Port: `8001`.

**Prediction API (`src/api/prediction_service/`):**
- [ ] Implement FastAPI app with `lifespan` startup loading of `structured_preprocessor.pkl`, `structured_model.pkl`, `nlp_model.pkl`, and `meta_model.pkl`.
- [ ] Expose `POST /v1/predict` for real-time single-customer churn scoring. Inference flow: (1) apply `structured_preprocessor` → structured vector; (2) call Embedding Microservice `POST /v1/embed` → PCA vector; (3) base model predictions → `[P1, P2]`; (4) meta-learner → final churn score + `nlp_branch_available` flag.
- [ ] Expose `POST /v1/predict/batch` for bulk CSV processing.
- [ ] Expose `GET /v1/health` returning `{"status": "healthy", "model_version": "..."}`.
- [ ] Implement **circuit breaker**: if Embedding Microservice is unreachable, fall back to zero-vector (dim=20), log warning, and set `nlp_branch_available: false` in response. Branch 1 structured prediction continues uninterrupted.
- [ ] All endpoints use explicit Pydantic request/response models — no untyped `dict` I/O.
- [ ] Port: `8000`.

**DVC Integration:**
- [ ] Register `stage_06_evaluate_model` in `dvc.yaml` with all as dependencies; all three model pkl files and `evaluation_report.json` as outputs.

### Phase 7: UI Development & Containerization (PLANNED)

- [ ] Develop **Gradio Dashboard** (`src/app/`): Interactive churn risk calculator with SHAP feature importance visualizations, sentiment context display, and `nlp_branch_available` status indicator. No monolithic `app.py` — split into pages, styles, and data loaders.
- [ ] Write production-ready **multi-stage Dockerfiles** (Alpine/Distroless base) for both the Embedding Microservice and the Prediction API.
- [ ] Implement `docker-compose.yaml` orchestrating four services: `embedding-service` (8001), `prediction-api` (8000), `mlflow-server` (5000), `gradio-ui` (7860). No deprecated `version:` key.

### Phase 8: CI/CD & Cloud Deployment (AWS) (PLANNED)

- [ ] Configure **GitHub Actions** modular workflows:
    - Code linting (`ruff check` + `ruff format --check`).
    - Static type checking (`pyright`).
    - Unit tests (`pytest --cov=src --cov-fail-under=65`).
    - Docker image build and vulnerability scan (Docker Scout / Trivy).
    - Push to **AWS ECR** on merge to `main`.
- [ ] Execute **Multi-Point Validation Gate** (`validate_system.sh`) as final pre-deploy check:
      Pyright + Ruff → Pytest coverage → DVC status → Service health probe.
- [ ] Define **AWS ECS Fargate** Task Definitions for both API services and the Gradio UI.
- [ ] Configure OIDC-based GitHub → AWS authentication (no long-lived secrets).

### Phase 9: Monitoring, Tracing & AgentOps (PLANNED)

- [ ] Integrate **OpenTelemetry (OTel)** spans across the Agentic enrichment workflow (Phase 2 LangGraph agent) and both API services: Chain of Thought, tool latency, token usage per request.
- [ ] Implement **AgentOps Metrics**: Plan Success Rate (PSR), Tool Call Accuracy (TCA), retry latency, and `nlp_branch_available` ratio in production.
- [ ] Monitor LLM token usage and cost per enrichment batch.
- [ ] Configure data drift detection on incoming inference payloads vs. training distribution.

---

## 3. Architecture Deep Dives & Decisions

| Topic | Document |
|---|---|
| Overall System & FTI Pattern | [architecture.md](../architecture/architecture.md) |
| Phase 0: Data Ingestion | [data_ingestion.md](../architecture/data_ingestion.md) |
| Phase 2: Agentic Data Enrichment | [data_enrichment.md](../architecture/data_enrichment.md) |
| Phase 3: Great Expectations Validation | [data_validation_gx.md](../architecture/data_validation_gx.md) |
| Phase 4: NLP & Feature Engineering | [feature_engineering.md](../architecture/feature_engineering.md) |
| Phase 5: Late Fusion Model Architecture | [model_training.md](../architecture/model_training.md) |
| Phase 6: Microservice Inference Architecture | [inference_architecture.md](../architecture/inference_architecture.md) |
| **Decision: Data Quality Checker** | [data_quality_checker.md](../decisions/data_quality_checker.md) |
| **Decision: Late Fusion vs. Unified Model** | [model_architecture_decision.md](../decisions/model_architecture_decision.md) |
| **Decision: Embedding Microservice Extraction** | [embedding_service_decision.md](../decisions/embedding_service_decision.md) |
| DVC Pipeline DAG | [dvc_pipeline.md](../architecture/dvc_pipeline.md) |
| Test Suite Coverage | [test_suite.md](../runbooks/test_suite.md) |

---

## 4. Artifact Inventory

| Artifact | Stage | Path | Type |
|---|---|---|---|
| Raw Telco CSV | Phase 1 | `artifacts/data_ingestion/` | DVC-tracked data |
| Validation reports (raw) | Phase 3 | `artifacts/data_validation/` | CI/CD gate |
| Enriched CSV | Phase 2 | `artifacts/data_enrichment/` | DVC-tracked data |
| Validation reports (enriched) | Phase 3 | `artifacts/data_enrichment/` | CI/CD gate |
| Train / Val / Test CSVs | Phase 4 | `artifacts/feature_engineering/` | DVC-tracked features |
| `structured_preprocessor.pkl` | Phase 4 | `artifacts/feature_engineering/` | Serialized transformer |
| `nlp_preprocessor.pkl` | Phase 4 | `artifacts/feature_engineering/` | Serialized transformer |
| `structured_model.pkl` | Phase 5 | `artifacts/model_training/` | Serialized model |
| `nlp_model.pkl` | Phase 5 | `artifacts/model_training/` | Serialized model |
| `meta_model.pkl` | Phase 5 | `artifacts/model_training/` | Serialized model |
| `evaluation_report.json` | Phase 5 | `artifacts/model_training/` | MLflow-linked report |

---

## 5. Core Value Drivers

1. **Proven AI ROI:** The Late Fusion architecture quantifies the exact performance contribution of the AI-generated ticket notes via `recall_lift` and `f1_lift` over the structured baseline — a direct and auditable business case for the Agentic enrichment investment.
2. **Microservice-Native Inference:** The Embedding Microservice enforces Rule 1.3, enabling independent versioning and scaling of the NLP component decoupled from the churn model.
3. **Explainable AI:** Sentiment tags and SHAP feature importances expose the "why" behind every churn risk score, bridging the gap between statistical prediction and stakeholder trust.
4. **Industrial Scalability:** Decoupled FTI pattern with two independent microservices allows horizontal scaling of the NLP inference path without touching the prediction service.
5. **Reproducibility:** Every pipeline stage is fully parameterized and DVC-versioned. A single `dvc repro` command reconstructs the entire system from raw data to deployed artifacts.
6. **Production Readiness:** Anti-Skew Mandate enforced at the artifact level (two separate pkl files), circuit-breaker fallback in the Prediction API, and a Multi-Point Validation Gate before every deployment.