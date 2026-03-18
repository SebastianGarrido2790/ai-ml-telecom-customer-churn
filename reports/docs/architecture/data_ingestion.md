# Data Ingestion Architecture Report

## 1. Purpose

The **Data Ingestion** component is the foundational Stage 00 of the Feature (F) pipeline within the project's FTI (Feature, Training, Inference) architecture. It handles the initial onboarding of raw data from external HTTP/HTTPS sources or local paths, preparing it for downstream validation and enrichment stages.

> **MLOps Principle (Independent Operational Pipelines):** Data Engineering (Ingestion) operates completely independently of models. It safely acquires and versions raw data, dropping it into the tracked `artifacts/` environment to securely decouple data sourcing from the processing logic, moving the project closer to real-world MLOps standards where data is fetched from external sources (S3, cloud, external APIs).

---

## 2. Core Operational Flow

The Data Ingestion phase is orchestrated through `src/pipeline/stage_00_data_ingestion.py` which triggers the `DataIngestion` component.

1.  **Configuration Loading**: Reads source URLs and local destination paths from `config/config.yaml` via the `ConfigurationManager`.
2.  **Smart Acquisition**:
    *   **HTTPS/HTTP**: If the source path provided is a URL, it automatically downloads it via `urllib`.
    *   **Local Paths**: If the source path is local, it acts as a secure local copy operation using `shutil`, allowing the pipeline to work identically in disconnected environments.
    *   **Idempotency**: It skips the download/copy operation if the target file already exists, speeding up runs.
3.  **Extraction**: Automatically detects if the target file is a `.zip` archive and seamlessly extracts it into the designated unzipped data directory.
4.  **Artifact Generation**: Emits the raw target CSV into `artifacts/data_ingestion/`.

---

## 3. Data Contracts & Configuration

The component relies on a strict Pydantic frozen dataclass to enforce its configuration:

```python
@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
```

This configuration is hydrated directly from the immutable `config.yaml`, ensuring there are no hidden or hardcoded strings bridging the ingestion logic with the broader system constraints.

---

## 4. DVC Pipeline Integration

The data ingestion stage is fully tracked by DVC, forming the new root of the Feature DAG:

*   **Command**: `uv run python -m src.pipeline.stage_00_data_ingestion`
*   **Dependencies**: `config/config.yaml`, `src/pipeline/stage_00_data_ingestion.py`, `src/components/data_ingestion.py`
*   **Outputs**: `artifacts/data_ingestion/WA_Fn-UseC_-Telco-Customer-Churn.csv`

All downstream tasks (like `validate_raw` and `enrich_data`) now strictly consume the output of this `data_ingestion` step, entirely deprecating the direct use of untracked `/data` folder files in the active DAG. This creates a fully traceable lineage from download to embedded artifact.

---

## 5. Testing & Validation

The component is rigorously tested by the primary pytest suite (see `test_suite.md`). The tests ensure:
*   Local paths invoke `shutil.copy2` correctly.
*   HTTP URLs invoke `urllib.request.urlretrieve` appropriately.
*   Existing downloaded files execute early-returns to avoid redundant networking.
