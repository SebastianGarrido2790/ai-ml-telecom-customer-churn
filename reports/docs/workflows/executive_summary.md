# Executive Summary & Technical Roadmap: Telecom Customer Churn MLOps

| **Version** | v1.3 |

## 1. Project Overview

This project builds an industrial-grade **Agentic MLOps** platform to predict and
prevent customer churn in telecommunications. By merging **quantitative behavioral data**
with **qualitative AI-generated sentiment**, the system provides a holistic risk profile
for every customer.

The architecture follows the **"Brain vs. Brawn"** model:
- **Brain (Agent):** Orchestrates data enrichment, qualitative interpretation, and
  business reasoning.
- **Brawn (ML):** Executes deterministic, high-scale churn propensity scoring via a
  Late Fusion ensemble served through two decoupled FastAPI microservices.

### 1.1 Architectural Enhancement (Post-Phase 4)

Following the completion of the Feature Pipeline, the project adopted a
**dual-enhancement strategy** that elevates the system from a standard MLOps pipeline
to a fully Agentic, microservice-native platform.

**Enhancement 1 — Late Fusion Ensemble (Option A):** Rather than training a single model
on merged features, the system trains two independent base models on separate feature
branches and combines them via a stacking meta-learner. This produces a quantifiable
business narrative: the performance delta between the structured-only baseline and the
fusion model directly proves the ROI of the entire Phase 2 AI enrichment investment.

**Enhancement 2 — Embedding Microservice (Option B):** The `TextEmbedder` + `PCA`
transformer is extracted from the monolithic `preprocessor.pkl` and deployed as an
independent FastAPI microservice. This enforces Rule 1.3 (Tools as Microservices),
enables independent versioning and scaling of the NLP component, and allows the
embedding model to be swapped without retraining the churn model.

Both enhancements share a single foundational prerequisite: the unified `preprocessor.pkl`
produced in Phase 4 is split into two independently serialized artifacts —
`structured_preprocessor.pkl` and `nlp_preprocessor.pkl` — each fitted exclusively on
the training set per the Anti-Skew Mandate.

---

## 2. Technical Roadmap: Phase-by-Phase

### Phase 1: Project Scaffolding, Environment & Data Ingestion (DONE ✅)

- [x] Configure `pyproject.toml` with `uv` dependency management.
- [x] Set up modular `src/` structure (`api`, `components`, `pipeline`, `utils`).
- [x] Implement `ConfigurationManager` for centralized YAML-based artifact/path orchestration.
- [x] Define **Pydantic Data Contracts** (`TelcoCustomerRow`) for the raw Telco dataset.
- [x] Verify environment with automated import and validation smoke tests (`tests/unit/test_pydantic_entities.py`).
- [x] Implement **Data Ingestion** component and pipeline stage to sync external URL/local
      data safely to DVC-tracked `artifacts/data_ingestion`.

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
- [x] Write unit tests for all enrichment schemas (`tests/unit/test_enrichment.py`).
- [x] Refactor to **Components / Pipeline** separation.

**Production Results (v2 — leakage-free):** 7,043 customers enriched. 0 schema violations.
Sentiment distribution: Billing Inquiry (58.1%), Dissatisfied (19.8%), Frustrated (10.8%),
Satisfied (6.2%), Neutral (5.0%). Churn rates per tag form a credible ordinal relationship
without any tag being a near-deterministic proxy of the target.

### Phase 3: Data Validation (GX) — Raw & Enriched (DONE ✅)

- [x] Implement `DataValidator` component using **Great Expectations v1.0+** ephemeral context.
- [x] Create `raw_telco_churn_suite`: Column presence, tenure range, categorical value sets.
- [x] Create `enriched_telco_churn_suite`: Ticket note presence/length, sentiment tag consistency.
      Updated to include `"Dissatisfied"` following the C1 leakage fix.
- [x] Integrate `schema.yaml` as the authoritative column definition for both suites.
- [x] Implement `StatisticalContractViolation` custom exception with full `DataQualityContext`.
- [x] Register all validation stages in `dvc.yaml` with proper dependency tracking.
- [x] Generate hard validation artifacts (`status.txt`, `validation_report.json`) tracked in
      `dvc.yaml` to serve as CI/CD quality gates.

### Phase 4: NLP Engineering & Feature Store (DVC) (DONE ✅)

- [x] Implement **Vector Embedding Generator**: Convert ticket notes into 384-dim vectors
      using `sentence-transformers/all-MiniLM-L6-v2`.
- [x] Execute **Dimensionality Reduction** (PCA, 20 components) on NLP vectors.
- [x] Merge NLP features with structured usage features into a unified feature matrix.
- [x] Register the `feature_engineering` stage in `dvc.yaml` with automated artifact tracking.
- [x] Implement **Anti-Skew Alignment**: Preprocessors fitted on Train only; identical
      transformations across Train, Validation, and Test sets.
- [x] Unit Test Suite for Cleaners, Embedders, and Split Logic.
- [x] Split unified `preprocessor.pkl` into two independently serialized artifacts:
    - `structured_preprocessor.pkl` — Numeric + Categorical pipelines.
    - `nlp_preprocessor.pkl` — `TextEmbedder` + `PCA` pipeline.
- [x] Add `ModelTrainingConfig`, `EmbeddingServiceConfig`, and `PredictionAPIConfig`
      entities to `src/entity/config_entity.py`.
- [x] Extend `ConfigurationManager` with `get_model_training_config()`,
      `get_embedding_service_config()`, `get_prediction_api_config()`.
- [x] Add `model_training` and `api` sections to `config/config.yaml` and `params.yaml`.

### Phase 5: Late Fusion Model Development & Experiment Tracking (MLflow) (DONE ✅)

The training pipeline implements a **Late Fusion stacking architecture** with three
tracked experiment runs per cycle.

**Branch 1 — Structured Baseline:**
- [x] Load `structured_preprocessor.pkl`; apply SMOTE (training set only); tune XGBoost
      via Optuna (30 trials); log to MLflow.

**Branch 2 — NLP Baseline:**
- [x] Load `nlp_preprocessor.pkl`; apply SMOTE independently; tune XGBoost via Optuna
      (20 trials); log to MLflow.

**Late Fusion Meta-Learner:**
- [x] Generate OOF probability predictions via `cross_val_predict` (5-fold StratifiedKFold).
- [x] Train Logistic Regression meta-learner on stacked OOF arrays.
- [x] Evaluate on held-out test set; log `recall_lift` and `f1_lift` to MLflow.
- [x] Serialize `structured_model.pkl`, `nlp_model.pkl`, `meta_model.pkl`.
- [x] Register champion model in MLflow Model Registry as `telco-churn-late-fusion v2`.

**C1 Leakage Fix (applied during Phase 5):** The original Phase 2 enrichment passed
the `Churn` label to the LLM, producing NLP Recall=1.000 — invalid for production.
The enrichment schema, system prompt, and deterministic fallback were all redesigned
to exclude the target variable. Pipeline re-executed from Stage 2.

**Final Results (leakage-free):**

| Branch | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Structured Baseline | 0.771 | 0.555 | 0.646 | 0.850 |
| NLP Baseline | 0.711 | 0.378 | 0.493 | 0.681 |
| Late Fusion | 0.654 | **0.594** | 0.622 | 0.848 |

The Late Fusion model reduces False Positives by 27% (125 vs. 173) compared to the
structured baseline — the operationally correct metric for a retention budget use case.

### Phase 6: Inference Pipeline — Microservice Architecture (FastAPI) (DONE ✅)

The inference layer is deployed as **two decoupled FastAPI microservices** and
operationally verified with real predictions.

**Embedding Microservice (`src/api/embedding_service/`) — port 8001:**
- [x] FastAPI app with `lifespan` startup loading of `nlp_preprocessor.pkl`.
- [x] **SentenceTransformer warmup protocol**: dummy `transform()` call at startup
      eliminates cold-start latency; keeps `httpx` timeout at 5s.
- [x] `POST /v1/embed` — batch ticket note embedding; `GET /v1/health`.
- [x] Verified: `{"status":"healthy","model_version":"all-MiniLM-L6-v2-pca20"}`.

**Prediction API (`src/api/prediction_service/`) — port 8000:**
- [x] FastAPI app with `lifespan` loading all four artifacts.
- [x] `InferenceService` (Decision D2 — separated from router) owns all computation:
      DataFrame reconstruction → structured preprocessing → embedding call → base model
      scoring → meta-learner stacking.
- [x] `POST /v1/predict` (real-time) and `POST /v1/predict/batch` (bulk).
- [x] `GET /v1/health` → `{"status":"healthy","model_version":"late-fusion-v2"}`.
- [x] **Circuit breaker**: embedding service timeout/error → zero-vector fallback →
      `nlp_branch_available: false`; structured prediction continues uninterrupted.
- [x] All endpoints use explicit Pydantic models — no untyped `dict` I/O.

**Operational verification:**
```json
POST /v1/predict  →  churn_probability: 0.700559
                     p_structured: 0.889536  |  p_nlp: 0.407729
                     nlp_branch_available: true  |  model_version: "late-fusion-v2"
```

**Unit tests:** 24/24 passing (`test_api_schemas.py`) — schemas, circuit breaker,
DataFrame reconstruction, batch contracts.

**Windows note documented:** `localhost` resolves to IPv6 (`::1`) on Windows;
embedding service binds IPv4 only. Config uses `host: "127.0.0.1"` explicitly.
Docker Compose (Phase 7) uses container DNS names — environment-specific.

### Phase 7: UI Development & Containerization (DONE ✅)

- [x] Develop **Gradio Dashboard** (`src/app/`): Interactive churn risk calculator with
      SHAP feature importance visualizations and `nlp_branch_available` status indicator.
      No monolithic `app.py` — split into pages, styles, and data loaders.
- [x] Write production-ready **multi-stage Dockerfiles** for all services.
- [x] Implement `docker-compose.yaml` orchestrating five services:
      `embedding-service` (8001), `prediction-api` (8000), `mlflow-server` (5000),
      `gradio-ui` (7860), and `postgres` (MLflow backend).
- [x] Configure `depends_on: condition: service_healthy` so the prediction API waits
      for the embedding service warmup to complete before receiving traffic.
      `start_period: 30s` on the embedding service health check accommodates the
      SentenceTransformer load time.

**Production Results (v1.0):**
The system is fully containerized and orchestrated. The Gradio UI provides a premium,
module-based user experience with real-time SHAP explanations and branch-level probability
breakdowns.
- **Health Status Banner**: Dynamic API connectivity monitoring.
- **Dual-Branch Visualization**: Split probability bars for Structured vs. NLP branches.
- **Observability**: Integrated deep links to the MLflow dashboard for full lineage.

### Phase 8: CI/CD & Cloud Deployment (AWS) (PLANNED)

- [ ] Configure **GitHub Actions** modular workflows:
    - Code linting (`ruff check` + `ruff format --check`).
    - Static type checking (`pyright`).
    - Unit tests (`pytest --cov=src --cov-fail-under=65`).
    - Docker image build and vulnerability scan (Docker Scout / Trivy).
    - Push to **AWS ECR** on merge to `main`.
- [ ] Execute **Multi-Point Validation Gate** (`validate_system.sh`) as final pre-deploy
      check: Pyright + Ruff → Pytest coverage → DVC status → Service health probes.
- [ ] Define **AWS ECS Fargate** Task Definitions for both API services and the Gradio UI.
- [ ] Configure OIDC-based GitHub → AWS authentication (no long-lived secrets).

### Phase 9: Monitoring, Tracing & AgentOps (PLANNED)

- [ ] Integrate **OpenTelemetry (OTel)** spans across the Agentic enrichment workflow
      and both API services: Chain of Thought, tool latency, token usage per request.
- [ ] Implement **AgentOps Metrics**: Plan Success Rate (PSR), Tool Call Accuracy (TCA),
      retry latency, and `nlp_branch_available` ratio in production.
- [ ] Monitor LLM token usage and cost per enrichment batch.
- [ ] Configure data drift detection on inference payloads vs. training distribution.

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
| Phase 7: Gradio Dashboard Architecture | [gradio_ui.md](../architecture/gradio_ui.md) |
| **Decision: Data Quality Checker** | [data_quality_checker.md](../decisions/data_quality_checker.md) |
| **Decision: Late Fusion vs. Unified Model** | [model_architecture_decision.md](../decisions/model_architecture_decision.md) |
| **Decision: Embedding Microservice Extraction** | [embedding_service_decision.md](../decisions/embedding_service_decision.md) |
| **Decision: InferenceService Separation (D2)** | [inference_service_decision.md](../decisions/inference_service_decision.md) |
| **Decision: Gradio Container Implementation** | [gradio.md](../decisions/gradio.md) |
| DVC Pipeline DAG | [dvc_pipeline.md](../architecture/dvc_pipeline.md) |
| Test Suite Coverage | [test_suite.md](../runbooks/test_suite.md) |

---

## 4. Artifact Inventory

| Artifact | Stage | Path | Type |
|---|---|---|---|
| Raw Telco CSV | Phase 1 | `artifacts/data_ingestion/` | DVC-tracked data |
| Validation reports (raw) | Phase 3 | `artifacts/data_validation/` | CI/CD gate |
| Enriched CSV (v2 — leakage-free) | Phase 2 | `artifacts/data_enrichment/` | DVC-tracked data |
| Validation reports (enriched) | Phase 3 | `artifacts/data_enrichment/` | CI/CD gate |
| Train / Val / Test CSVs | Phase 4 | `artifacts/feature_engineering/` | DVC-tracked features |
| `structured_preprocessor.pkl` | Phase 4 | `artifacts/feature_engineering/` | Serialized transformer |
| `nlp_preprocessor.pkl` | Phase 4 | `artifacts/feature_engineering/` | Serialized transformer |
| `structured_model.pkl` | Phase 5 | `artifacts/model_training/` | Serialized model |
| `nlp_model.pkl` | Phase 5 | `artifacts/model_training/` | Serialized model |
| `meta_model.pkl` | Phase 5 | `artifacts/model_training/` | Serialized model |
| `evaluation_report.json` | Phase 5 | `artifacts/model_training/` | MLflow-linked / DVC gate |

---

## 5. Core Value Drivers

1. **Proven AI ROI:** The Late Fusion architecture quantifies the exact performance
   contribution of AI-generated ticket notes via `recall_lift` and `f1_lift`. The fusion
   model reduces False Positives by 27% vs. the structured baseline — a directly
   auditable business case for the Agentic enrichment investment.
2. **Microservice-Native Inference:** Two decoupled FastAPI services — operational and
   verified — enforce Rule 1.3, enabling independent versioning and horizontal scaling
   of the NLP inference path without touching the prediction service.
3. **Production Resilience:** The circuit breaker ensures the prediction API never
   returns 5xx when the embedding service is unavailable. The warmup protocol eliminates
   cold-start latency. The `depends_on: condition: service_healthy` contract (Phase 7)
   prevents premature traffic before the NLP model is loaded.
4. **Explainable AI:** Sentiment tags and `p_structured` / `p_nlp` branch probabilities
   in every response expose the "why" behind every churn score.
5. **Industrial Scalability:** Decoupled FTI pattern + two-microservice inference layer
   allows each component to scale independently in ECS Fargate (Phase 8).
6. **Reproducibility:** Every pipeline stage is parameterized and DVC-versioned.
   `dvc repro` reconstructs the entire system from raw data to deployed artifacts.
7. **Integrity by Design:** The C1 leakage fix is permanently enforced by a unit test
   (`test_customer_input_context_churn_field_absent`) and the DVC dependency graph —
   any future change to the enrichment schema re-invalidates the pipeline.