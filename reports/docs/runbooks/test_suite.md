# Test Suite Architecture — Runbook

## 1. Purpose

This document describes the project's unit testing strategy — the first tier of the
**Testing Pyramid** defined in Rule 4.1 of the Antigravity MLOps Standard.

> **Testing Pyramid Layer 1 (Pytest):** Strictly for **Tools and Pipelines**.
> Ensure deterministic code works 100% of the time. Tests must be fast,
> isolated, and never make live API calls.

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

---

## 3. Test Files

### 3.1 `tests/unit/test_data_ingestion.py` — Data Ingestion Component

**Purpose:** Validates the core downloading, copying, and extraction logic for the pipeline's Stage 0 metadata. Ensures robust handling of remote HTTP URLs and local environment files.

**Module Under Test:** `src/components/data_ingestion.py`

| Test | Component Tested | What It Proves |
|---|---|---|
| `test_download_file_local_path` | `download_file` | Accurately copies local datasets using `shutil`. |
| `test_download_file_http_url` | `download_file` | Captures HTTP/HTTPS sources and successfully triggers `urllib`. |
| `test_download_file_already_exists` | `download_file` | Avoids redundant copies/downloads using idempotent logic. |

---

### 3.2 `tests/unit/test_pydantic_entities.py` — Phase 1 Data Contracts

**Purpose:** Validates the core Pydantic data contracts for the raw Telco dataset.

**Module Under Test:** `src/entity/config_entity.TelcoCustomerRow`

| Test | Schema Tested | What It Proves |
|---|---|---|
| `test_valid_row` | `TelcoCustomerRow` | A fully valid row is accepted without errors |
| `test_bad_row_rejected` | `TelcoCustomerRow` | `SeniorCitizen > 1`, `tenure < 0`, and `MonthlyCharges < 0` are rejected |

```python
# Example: Verifying the data contract blocks invalid data
def test_bad_row_rejected():
    with pytest.raises(ValidationError):
        TelcoCustomerRow(
            SeniorCitizen=5,   # Invalid: ge=0, le=1
            tenure=-1,         # Invalid: ge=0
            MonthlyCharges=-10 # Invalid: ge=0
        )
```

---

### 3.3 `tests/unit/test_enrichment.py` — Phase 2 Enrichment Contracts

**Purpose:** Validates the Pydantic schemas that form the I/O boundary of the Agentic
enrichment pipeline.

**Module Under Test:** `src/components/data_enrichment/schemas`

#### Test Coverage

| Test | Schema Tested | Constraint Enforced |
|---|---|---|
| `test_customer_input_context_valid` | `CustomerInputContext` | All valid fields accepted |
| `test_customer_input_context_invalid_tenure` | `CustomerInputContext` | `tenure < 0` → `ValidationError` |
| `test_customer_input_context_invalid_literals` | `CustomerInputContext` | Invalid `InternetService` / `Churn` → `ValidationError` |
| `test_synthetic_note_output_valid` | `SyntheticNoteOutput` | Valid note and tag accepted |
| `test_synthetic_note_output_invalid_tag` | `SyntheticNoteOutput` | `"Angry"` tag not in `Literal` → `ValidationError` |

```python
# Example: Verifying the LLM output contract rejects invalid sentiment tags
def test_synthetic_note_output_invalid_tag():
    with pytest.raises(ValidationError):
        SyntheticNoteOutput(
            ticket_note="Customer called experiencing outage.",
            primary_sentiment_tag="Angry",  # Not in Literal[Frustrated, Neutral, ...]
        )
```

---

## 4. Test Execution

```bash
# Run the full test suite
uv run pytest tests/ -v

# Run with coverage report
uv run pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
uv run pytest tests/test_enrichment.py -v

# Run a single test by name
uv run pytest tests/test_enrichment.py::test_synthetic_note_output_invalid_tag -v
```

**Output**:
```bash
============================= test session starts =============================
platform win32 -- Python 3.11.13, pytest-9.0.2, pluggy-1.6.0
rootdir: C:\Users\sebas\Desktop\ai-ml-telecom-customer-churn
configfile: pyproject.toml
plugins: anyio-4.12.1, langsmith-0.7.12, logfire-4.28.0, cov-7.0.0
collected 10 items

tests/unit/test_data_ingestion.py::test_download_file_local_path PASSED  [ 10%]
tests/unit/test_data_ingestion.py::test_download_file_http_url PASSED    [ 20%]
tests/unit/test_data_ingestion.py::test_download_file_already_exists PASSED [ 30%]
tests/unit/test_enrichment.py::test_customer_input_context_valid PASSED  [ 40%]
tests/unit/test_enrichment.py::test_customer_input_context_invalid_tenure PASSED [ 50%]
tests/unit/test_enrichment.py::test_customer_input_context_invalid_literals PASSED [ 60%]
tests/unit/test_enrichment.py::test_synthetic_note_output_valid PASSED   [ 70%]
tests/unit/test_enrichment.py::test_synthetic_note_output_invalid_tag PASSED [ 80%]
tests/unit/test_pydantic_entities.py::test_valid_row PASSED              [ 90%]
tests/unit/test_pydantic_entities.py::test_bad_row_rejected PASSED       [100%]

============================= 10 passed in 0.46s ==============================
```

---

## 5. Schema Contract Coverage Map

```mermaid
graph LR
    subgraph "Pydantic Schemas (Tested)"
        DataIngestionConfig["DataIngestionConfig\n(config_entity.py)"]
        TelcoCustomerRow["TelcoCustomerRow\n(config_entity.py)"]
        CustomerInputContext["CustomerInputContext\n(schemas.py)"]
        SyntheticNoteOutput["SyntheticNoteOutput\n(schemas.py)"]
    end

    subgraph "Tests"
        T0["test_data_ingestion.py\n(3 tests)"]
        T1["test_pydantic_entities.py\n(2 tests)"]
        T2["test_enrichment.py\n(5 tests)"]
    end

    T0 --> DataIngestionConfig
    T1 --> TelcoCustomerRow
    T2 --> CustomerInputContext
    T2 --> SyntheticNoteOutput
```

---

## 6. What Is Not Yet Covered

The following areas are intentionally left for future evaluation phases:

| Gap | Reason | Future Plan |
|---|---|---|
| `DataValidator` (GX) | GX requires filesystem or ephemeral context setup — integration test needed | `pytest-mock` + GX ephemeral context |
| `ConfigurationManager` | YAML loading is path-dependent — needs a fixture | Add tmpdir fixture |
| `EnrichmentOrchestrator` | Calls live LLM API — must be mocked | Mock `generate_ticket_note` with `pytest-asyncio` |
| `generate_ticket_note()` | Makes live API call | Mock the `pydantic-ai` `Agent.run()` with asynctest |
| Agent output quality | Probabilistic — not for pytest | LLM-as-a-Judge eval pipeline (Skill: `evals-llm-judge`) |

---

## 7. CI/CD Gate (Planned)

When the GitHub Actions CI/CD pipeline is implemented (Phase 8), the test suite will run
automatically on every push. The pipeline will fail if:

1. Any `pytest` test fails.
2. Test coverage falls below the configured threshold (e.g., 80%).
3. `ruff` linting reports any errors.

```yaml
# .github/workflows/ci.yml (Planned)
- name: Run Tests
  run: uv run pytest tests/ --cov=src --cov-fail-under=80
```
