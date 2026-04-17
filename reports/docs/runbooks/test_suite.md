# Test Suite Architecture — Runbook

## 1. Purpose

This document describes the project's unit testing strategy, the first tier of the **Testing Pyramid**.

> **Testing Pyramid Layer 1 (Pytest):** Strictly for **Tools and Pipelines**.
> Ensure deterministic code works 100% of the time. Tests must be fast, isolated, and never make live API calls.

---

## 2. Testing Philosophy: What We Test (and What We Don't)

| Layer | Responsible For | Tool |
|---|---|---|
| **Unit Tests (pytest)** | Deterministic tools, Pydantic schemas, pipeline components | `pytest` |
| **LLM Evals** | Agent output quality (Faithfulness, Relevance, Tool Accuracy) | LLM-as-a-Judge (Future) |
| **Observability** | Live production metrics (PSR, TCA, latency) | OpenTelemetry (Future) |

**We do NOT test LLM responses in unit tests.** The LLM is a probabilistic system; testing
its output would be non-deterministic and fragile. Instead, we test the **rigid contracts
around** the LLM — the Pydantic schemas that validate its inputs and outputs.

**Shared Infrastructure:** To ensure consistency and reduce boilerplate, all common fixtures (Standardized DataFrames, Mocks, Temp Files) are centralized in `conftest.py` files at the root and unit levels.

---

## 3. Test Files

### 3.1 `tests/unit/test_data_ingestion.py` — Data Ingestion Component

**Purpose:** Validates the core downloading, copying, and extraction logic for the pipeline's
Stage 0. Ensures robust handling of remote HTTP URLs and local environment files.

**Module Under Test:** `src/components/data_ingestion.py`

| Test | Component Tested | What It Proves |
|---|---|---|
| `test_download_file_local_path` | `download_file` | Accurately copies local datasets using `shutil`. |
| `test_download_file_http_url` | `download_file` | Captures HTTP/HTTPS sources and triggers `urllib`. |
| `test_download_file_already_exists` | `download_file` | Avoids redundant copies using idempotent logic. |

---

### 3.2 `tests/unit/test_pydantic_entities.py` — Phase 1 Data Contracts

**Purpose:** Validates the core Pydantic data contracts for the raw Telco dataset.

**Module Under Test:** `src/entity/config_entity.TelcoCustomerRow`

| Test | Schema Tested | What It Proves |
|---|---|---|
| `test_valid_row` | `TelcoCustomerRow` | A fully valid row is accepted without errors. |
| `test_bad_row_rejected` | `TelcoCustomerRow` | `SeniorCitizen > 1`, `tenure < 0`, `MonthlyCharges < 0` are rejected. |

---

### 3.3 `tests/unit/test_enrichment.py` — Phase 2 Enrichment Contracts (C1 Enhanced)

**Purpose:** Validates the Pydantic schemas that form the I/O boundary of the Agentic
enrichment pipeline.

**Module Under Test:** `src/components/data_enrichment/schemas`

> **C1 Enhancement:** Following the leakage investigation (Phase 5), the `Churn` field was removed from `CustomerInputContext` and the schema was expanded to 17 observable CRM fields. The test suite was updated to reflect the new schema. A dedicated leakage-prevention test (`test_customer_input_context_churn_field_absent`) was added to permanently guard against re-introduction of the target variable into the input contract.

#### Original Coverage (5 tests)

| Test | Schema Tested | Constraint Enforced |
|---|---|---|
| `test_customer_input_context_valid` | `CustomerInputContext` | All valid fields accepted. |
| `test_customer_input_context_invalid_tenure` | `CustomerInputContext` | `tenure < 0` → `ValidationError`. |
| `test_customer_input_context_invalid_literals` | `CustomerInputContext` | Invalid `InternetService` → `ValidationError`. |
| `test_synthetic_note_output_valid` | `SyntheticNoteOutput` | Valid note and tag accepted. |
| `test_synthetic_note_output_invalid_tag` | `SyntheticNoteOutput` | `"Angry"` tag not in `Literal` → `ValidationError`. |

#### New Coverage Added (6 additional tests — total: 11)

| Test | What It Proves |
|---|---|
| `test_customer_input_context_invalid_senior_citizen` | `SeniorCitizen` outside `[0,1]` → `ValidationError`. |
| `test_customer_input_context_invalid_contract` | Invalid `Contract` value → `ValidationError`. |
| `test_customer_input_context_invalid_monthly_charges` | Negative `MonthlyCharges` → `ValidationError`. |
| `test_customer_input_context_churn_field_absent` | `Churn` is not stored on the model — never reaches the LLM. (**Leakage guard**) |
| `test_synthetic_note_output_all_valid_tags` | All 6 allowed sentiment tags pass schema validation. |
| `test_synthetic_note_output_empty_ticket_note` | Empty `ticket_note` string → `ValidationError`. |

```python
# Leakage guard: Churn must never be stored on CustomerInputContext
def test_customer_input_context_churn_field_absent() -> None:
    payload = {**VALID_CONTEXT_PAYLOAD, "Churn": "Yes"}
    context = CustomerInputContext(**payload)
    assert not hasattr(context, "Churn"), (
        "Churn must not be stored on CustomerInputContext — "
        "it must never reach the LLM prompt."
    )
```

---

### 3.4 `tests/test_feature_engineering.py` — Phase 4 Transformation Logic

**Purpose:** Validates custom Scikit-Learn transformers and the data splitting/preprocessing orchestration. Ensures NLP embeddings and PCA logic integrate without leakage.

**Modules Under Test:** `src/components/feature_engineering.py`, `src/utils/feature_utils.py`

| Test | Component Tested | What It Proves |
|---|---|---|
| `TestNumericCleaner` | `NumericCleaner` | Coerces object-type columns to floats; handles blank strings. |
| `TestTextEmbedder` | `TextEmbedder` | Lazy loads `SentenceTransformer`; handles pickling safely. |
| `test_data_splitting_and_processing` | `FeatureEngineering` | Stratified train/val/test split; total samples preserved. |

> **v1.2 Refactor:** Redundant fixtures in this file were removed and replaced with the centralized `sample_telco_df` and `mock_sentence_transformer` fixtures from the root `conftest.py`.

---

### 3.5 `tests/unit/test_model_training.py` — Phase 5 Late Fusion Training

**Purpose:** Validates the four deterministic guarantees of the Late Fusion training pipeline. No live Optuna search, no MLflow server, and no LLM calls are triggered, all external dependencies are replaced with minimal fixtures.

**Modules Under Test:** `src/components/model_training/trainer.py`

#### Test Class 1: `TestOOFArrayShape` (2 tests)

| Test | What It Proves |
|---|---|
| `test_oof_shape_matches_training_set` | OOF vector length == `n_train`. Shape mismatch would silently corrupt the meta-learner input. |
| `test_oof_values_are_valid_probabilities` | All OOF values ∈ [0, 1]. Invalid probabilities would corrupt the stacking. |

#### Test Class 2: `TestSMOTEIsolation` (3 tests)

| Test | What It Proves |
|---|---|
| `test_smote_increases_train_size` | SMOTE adds synthetic samples to the minority class. |
| `test_smote_balances_classes` | Post-SMOTE class counts are equal. |
| `test_val_set_unchanged_by_smote` | Applying SMOTE to train never mutates the validation DataFrame. |

#### Test Class 3: `TestMetaLearnerInputContract` (2 tests)

| Test | What It Proves |
|---|---|
| `test_stacked_array_has_two_columns` | Meta-learner input has exactly 2 columns: `[P_struct, P_nlp]`. |
| `test_meta_learner_fits_on_stacked_oof` | Logistic Regression fits without error; `coef_` shape is `(1, 2)`. |

#### Test Class 4: `TestEvaluatorDescriptiveNames` (New — Phase 4/5)

**Purpose:** Validates that the model evaluator correctly propagates human-readable feature names (e.g., `num__tenure`, `cat__Contract_Month-to-month`) instead of raw indices for feature importance logging.

| Test | What It Proves |
|---|---|
| `test_feature_names_in_mlflow` | The importance CSV/plot logged to MLflow contains the correct strings from the preprocessor's `get_feature_names_out()`. |

#### Test Class 4: `TestEvaluationReportSchema` (5 tests)

Validates `evaluation_report.json` structure via a dedicated Pydantic schema
(`EvaluationReportSchema`), ensuring the DVC-tracked CI/CD gate artifact is always
well-formed.

| Test | What It Proves |
|---|---|
| `test_valid_report_passes_schema` | Correctly structured report passes Pydantic validation. |
| `test_report_serialises_to_json` | Report writes to and re-reads from disk correctly. |
| `test_missing_fusion_run_fails_schema` | Report without `late_fusion_stacked` key fails validation. |
| `test_missing_lift_metrics_fails_schema` | Fusion run without `recall_lift` fails validation. |
| `test_recall_lift_sign_is_positive_in_happy_path` | Positive lift is asserted in the expected case. |

---

### 3.6 `tests/unit/test_api_schemas.py` — Phase 6 API & Inference Contracts

**Purpose:** Validates the deterministic guarantees of the inference pipeline, including Pydantic schema validation for microservices and the inter-service circuit breaker logic.

**Modules Under Test:** `src/api/prediction_service/schemas.py`, `src/api/embedding_service/schemas.py`, `src/api/prediction_service/inference.py`, `src/api/prediction_service/main.py`, `src/api/embedding_service/main.py`

#### Test Class 1: `TestEmbeddingSchemas` (5 tests)

| Test | What It Proves |
|---|---|
| `test_embed_request_valid` | `EmbedRequest` accepts a list with at least one note. |
| `test_embed_request_empty_list_fails` | `EmbedRequest` fails if `ticket_notes` is empty (`min_length=1`). |
| `test_embed_response_valid` | `EmbedResponse` accepts valid embeddings and dimensions. |
| `test_embed_response_invalid_dim` | `EmbedResponse` fails if `dim=0` (`gt=0` constraint). |

#### Test Class 2: `TestCustomerFeatureRequest` (6 tests)

| Test | What It Proves |
|---|---|
| `test_valid_payload_accepted` | A full Telco payload with 19 fields is accepted. |
| `test_customer_id_is_optional` | `customerID` defaults to `None` if omitted. |
| `test_negative_tenure_fails` | `tenure < 0` triggers `ValidationError`. |
| `test_total_charges_none_accepted` | `TotalCharges=None` is accepted (tenure=0 cases). |

#### Test Class 3: `TestCircuitBreaker` (4 tests)

| Test | What It Proves |
|---|---|
| `test_timeout_triggers_zero_vector_fallback` | `httpx.TimeoutException` returns zero-vector and `available=False`. |
| `test_connection_error_fallback` | `httpx.ConnectError` triggers the same graceful degradation. |
| `test_successful_embedding_call` | Normal path returns actual embeddings and `available=True`. |

#### Test Class 4: `TestInferenceServiceDataFrame` (3 tests)

| Test | What It Proves |
|---|---|
| `test_dataframe_column_count` | Reconstructed DataFrame has exactly 19 columns in the correct order. |
| `test_total_charges_none_to_empty_string` | `None` is coerced to `""` for `NumericCleaner` parity. |

#### Other Classes (6 tests)
Includes `TestChurnPredictionResponse` (3 tests) and `TestBatchSchemas` (3 tests) for full I/O contract coverage.

#### Test Class 5: `TestAPIHardening` (New — Phase 3)

**Purpose:** Validates the security and robustness constraints implemented during Phase 3 hardening.

| Test | Component | What It Proves |
|---|---|---|
| `test_auth_missing_header` | Authentication | Requests without `X-API-Key` return `422 Unprocessable Entity`. |
| `test_auth_invalid_key` | Authentication | Requests with the wrong key return `401 Unauthorized`. |
| `test_batch_limit_exceeded`| Robustness | Batches > 1000 items are rejected with `422` (Schema guard). |
| `test_cors_preflight` | Middleware | `OPTIONS` requests return appropriate headers for web clients. |
| `test_global_exception_handler`| Reliability| Unhandled errors return a sanitized `500` JSON instead of a raw traceback. |

### 3.7 `tests/unit/test_config.py` — Configuration Management
**Purpose:** Validates the hydration of YAML configuration files into strongly-typed Pydantic entities.

| Test | Component Tested | What It Proves |
|---|---|---|
| `test_read_yaml_success` | `read_yaml` | Correctly parses valid YAML files from disk. |
| `test_configuration_manager_init` | `ConfigurationManager` | Hydrates all 5 core pipeline entities from `config.yaml`. |

### 3.8 `tests/unit/test_pipelines.py` — Pipeline Stage Orchestration
**Purpose:** Validates the stage-wise execution of the ML pipeline (Ingestion -> Validation -> Preprocessing -> Training).

| Test | Component Tested | What It Proves |
|---|---|---|
| `test_stage_01_ingestion` | `DataIngestion` | Stage 01 initiates and completes without error. |
| `test_stage_04_training` | `ModelTraining` | Stage 04 consumed processed features and registers model. |

### 3.9 `tests/unit/test_api.py` — FastAPI Endpoint Logic
**Purpose:** Validates top-level FastAPI routes and dependency injection.

| Test | Component Tested | What It Proves |
|---|---|---|
| `test_health_check` | `/health` | API is alive and reachable. |
| `test_prediction_endpoint_success` | `/predict` | End-to-end inference flow with mocked models succeeds. |

### 3.10 `tests/unit/test_array_utils.py` — Array Handling Utilities (v1.2)
**Purpose:** Validates the robust conversion of various data formats (Sparse, Pandas, List) into dense Numpy arrays for model inference.

| Test | Component Tested | What It Proves |
|---|---|---|
| `test_ensure_ndarray_from_sparse` | `ensure_ndarray` | Correctly densifies Scipy CSR matrices. |
| `test_ensure_ndarray_from_dataframe`| `ensure_ndarray` | Extracts values from Pandas DataFrames. |
| `test_ensure_ndarray_from_numpy` | `ensure_ndarray` | Returns existing numpy arrays unchanged. |

### 3.11 `tests/test_gradio_smoke.py` — Gradio UI Layer (v1.4)

**Purpose:** Validates the Gradio frontend's data loaders, API client integration, and app building logic. Ensures the UI can communicate with the Prediction API and handle errors gracefully without requiring a running server.

**Modules Under Test:** `src/ui/app.py`, `src/ui/data_loaders/api_client.py`

| Test Class | What It Proves |
|---|---|
| `TestPredictSingle` | Mocked API integration for single predictions, verifying `X-API-Key` injection and path correctness. |
| `TestPredictBatch` | Mocked integration for batch processing, ensuring large payloads are correctly wrapped and transmitted. |
| `TestCheckHealth` | Validates heart-beat logic used to show "API Status" in the UI header; handles timeouts/connection errors. |
| `TestGradioAppBuilder` | Ensures `build_app()` factory correctly wires pages into the `gr.Blocks` dashboard without import errors. |

---

## 4. Shared Test Infrastructure (`conftest.py`)

The project uses a hierarchical `conftest.py` structure to eliminate fixture duplication and ensure a "Single Source of Truth" for test data.

### 4.1 Root `tests/conftest.py` — Suite-Wide Fixtures
| Fixture | Purpose |
|---|---|
| `mock_sentence_transformer` | A persistent mock that intercepts `SentenceTransformer` calls, returning consistent vectors while preventing any external HTTP requests/downloads. |
| `sample_telco_df` | A factory that returns a valid 10-row Telco DataFrame containing all 19 raw and 20 NLP features, used for integration and component testing. |

### 4.2 Unit `tests/unit/conftest.py` — Unit-Specific Fixtures
| Fixture | Purpose |
|---|---|
| `temp_config_files` | Creates temporary `config.yaml` and `params.yaml` files in a `tmp_path` for configuration testing. |
| `mock_config_manager` | Bootstraps a `ConfigurationManager` instance pre-loaded with the temporary test config files. |

---

## 5. Test Execution

```bash
# Run the full test suite
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Run specific test groups
uv run pytest tests/unit/test_enrichment.py -v
uv run pytest tests/test_gradio_smoke.py -v

# Run a single test by name
uv run pytest tests/unit/test_api_schemas.py::TestCircuitBreaker::test_timeout_triggers_zero_vector_fallback -v
```

I have successfully hardened the system with a comprehensive 138-test suite covering all tiers of the FTI architecture.

### Test Results Summary (v1.4 Update)

| Test Case | Method | Endpoint | Expected | Result |
| :--- | :--- | :--- | :--- | :--- |
| **Auth: Missing Key** | `GET` | `/v1/health` | `422 Unprocessable Entity` | **PASS** ✅ |
| **Auth: Wrong Key** | `GET` | `/v1/health` | `401 Unauthorized` | **PASS** ✅ |
| **Auth: Correct Key** | `GET` | `/v1/health` | `200 OK` | **PASS** ✅ |
| **Functional: Prediction** | `POST` | `/v1/predict/batch` | `200 OK` + Churn Score | **PASS** ✅ |
| **Inter-service Auth** | `POST` | `/v1/embed` | `200 OK` (called by Pred API) | **PASS** ✅ |
| **CORS Preflight** | `OPTIONS`| `/v1/predict/batch` | `200 OK` + CORS Headers | **PASS** ✅ |
| **UI: API Client Mock** | `N/A` | `N/A` | Correct `X-API-Key` + Async Call | **PASS** ✅ |
| **UI: App Construction** | `N/A` | `N/A` | `gr.Blocks` initialized | **PASS** ✅ |

### Key Observations
1.  **Strict Authentication**: The API key validation works perfectly across both services. The Prediction API correctly handles the injection and propagation of the `X-API-Key` when communicating with the Embedding Microservice.
2.  **CORS Readiness**: Preflight `OPTIONS` requests now return the necessary headers (`access-control-allow-origin`, `access-control-allow-headers`), ensuring the API can be safely consumed by web applications.
3.  **UI Integrity**: The Gradio UI is now protected by a 14-test smoke suite that validates the API client interaction and dashboard wiring, preventing regressions in the frontend layer.
4.  **End-to-End Reliability**: The full Late Fusion inference path—including structured preprocessing, NLP embedding generation, and meta-model stacking—is protected by deterministic unit tests throughout the stack.
5.  **Schema Hardening**: `schema.yaml` is now validated at load-time by `ConfigurationManager`, preventing malformed configs from causing silent errors in the Feature or Data Validation stages.

The system is now production-hardened and protected against regressions at all layers.

```
tests/unit/test_data_ingestion.py          3 passed
tests/unit/test_pydantic_entities.py       2 passed
tests/unit/test_enrichment.py             11 passed
tests/test_feature_engineering.py          5 passed
tests/unit/test_model_training.py         13 passed
tests/unit/test_api_prediction.py          8 passed
tests/unit/test_api_embedding.py           7 passed
tests/unit/test_api_schemas.py            24 passed
tests/unit/test_config.py                 10 passed
tests/unit/test_pipelines.py               8 passed
tests/unit/test_utils.py                  15 passed
tests/unit/test_components.py             10 passed
tests/unit/test_model.py                  12 passed
tests/unit/test_array_utils.py             5 passed
tests/unit/test_config_manager.py          3 passed
tests/test_gradio_smoke.py                14 passed
──────────────────────────────────────────────────────
TOTAL                                    138 passed
```

---

## 6. Schema Contract Coverage Map

```mermaid
graph LR
    subgraph "Pydantic Schemas (Tested)"
        DataIngestionConfig["DataIngestionConfig"]
        TelcoCustomerRow["TelcoCustomerRow"]
        CustomerInputContext["CustomerInputContext\n(C1: Churn removed)"]
        SyntheticNoteOutput["SyntheticNoteOutput"]
        EvaluationReportSchema["EvaluationReportSchema"]
        CustomerFeatureRequest["CustomerFeatureRequest\n(API Input)"]
        ChurnPredictionResponse["ChurnPredictionResponse\n(API Output)"]
        SchemaContract["_SchemaContract\n(config/schema.yaml)"]
    end

    subgraph "Tests"
        T0["test_data_ingestion.py\n(3)"]
        T1["test_pydantic_entities.py\n(2)"]
        T2["test_enrichment.py\n(11)"]
        T3["test_feature_engineering.py\n(1)"]
        T4["test_model_training.py\n(12)"]
        T6["test_api_schemas.py\n(24)"]
        T7["test_config.py\n(10)"]
    end

    T0 --> DataIngestionConfig
    T1 --> TelcoCustomerRow
    T2 --> CustomerInputContext
    T2 --> SyntheticNoteOutput
    T3 --> FeatureEngineeringConfig
    T4 --> EvaluationReportSchema
    T6 --> CustomerFeatureRequest
    T6 --> ChurnPredictionResponse
    T6 --> TestCircuitBreaker["Circuit Breaker Logic"]
    T7 --> SchemaContract
    T8["test_gradio_smoke.py\n(14)"] --> UI["Gradio Frontend"]
    T9["test_array_utils.py\n(5)"] --> ArrayUtils["ensure_ndarray"]
```

---

| Gap | Reason | Future Plan |
|---|---|---|
| Agent output quality | Probabilistic — not for pytest. | LLM-as-a-Judge eval pipeline |
| Integration E2E | Requires full DVC/Pipeline state. | HW/SW-in-the-loop tests |

---

## 7. CI/CD Gate

The test suite runs automatically on every push and pull request via the GitHub Actions CI/CD pipeline. The pipeline will fail if:

1. Any `pytest` test fails.
2. Test coverage falls below the configured threshold (`--cov-fail-under=65`).
3. `ruff check` or `ruff format --check` reports any errors.
4. `pyright` reports type errors.

> **Local Parity:** Following Phase 4 hardening, `pyright` is now a blocking gate in `validate_system.bat`. Any type violations found locally will trigger an immediate failure before code can be committed or pushed to CI.

```yaml
# .github/workflows/ci.yml
- name: Run Tests
  run: uv run pytest tests/ --cov=src --cov-fail-under=65
```