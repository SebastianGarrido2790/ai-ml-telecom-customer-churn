# DVC Pipeline Architecture — Architecture Report

## 1. Purpose

**DVC (Data Version Control)** is the backbone of the project's reproducibility and lineage tracking.
It manages the full DAG (Directed Acyclic Graph) of pipeline stages, ensuring that every run of
`dvc repro` produces **bit-for-bit identical outputs** given the same inputs.

> **MLOps Principle (Rule 2.4 — MLOps Integrity Check):** All pipeline stages must support
> data versioning via DVC. Any change to code, configuration, or data **automatically invalidates**
> the affected stage and all downstream stages.

---

## 2. Current Pipeline DAG

Four stages are currently registered in `dvc.yaml`, representing the Feature Pipeline (the "F" in FTI) handling validation, enrichment, and transformation into a final feature matrix.

```mermaid
flowchart LR
    RawCSV[("data/raw/\ntelco_churn.csv")]
    ParamsYAML["config/params.yaml"]
    SchemaYAML["config/schema.yaml"]
    ConfigYAML["config/config.yaml"]

    S0["Stage 0: data_ingestion\nstage_00_data_ingestion.py"]
    S1["Stage 1: validate_raw\nstage_01_data_validation.py"]
    S2["Stage 2: enrich_data\nstage_02_data_enrichment.py"]
    S3["Stage 3: validate_enriched\nstage_03_enriched_validation.py"]
    S4["Stage 4: feature_engineering\nstage_04_feature_engineering.py"]

    IngestedCSV[("artifacts/\nWA_Fn-...-Churn.csv")]
    EnrichedCSV[("artifacts/\nenriched_telco_churn.csv")]
    StatusRaw[("artifacts/data_validation/\nstatus.txt...")]
    StatusEnr[("artifacts/data_enrichment/\nstatus.txt...")]
    FinalData[("artifacts/feature_engineering/\ntrain/val/test.csv")]
    Preprocessor["artifacts/feature_engineering/\npreprocessor.pkl"]

    RawCSV --> S0
    ConfigYAML --> S0
    S0 --> IngestedCSV

    IngestedCSV --> S1
    SchemaYAML --> S1
    ConfigYAML --> S1

    IngestedCSV --> S2
    ParamsYAML --> S2
    ConfigYAML --> S2
    S1 --> S2
    S2 --> EnrichedCSV

    EnrichedCSV --> S3
    SchemaYAML --> S3
    ConfigYAML --> S3

    EnrichedCSV --> S4
    ParamsYAML --> S4
    ConfigYAML --> S4
    S3 --> S4
    S4 --> FinalData
    S4 --> Preprocessor

    S1 --> StatusRaw
    S3 --> StatusEnr
```

---

## 3. Stage Specifications

### Stage 0: `data_ingestion`

**Command:** `uv run python -m src.pipeline.stage_00_data_ingestion`

| Property | Value |
|---|---|
| **Purpose** | Fetches/copies the raw data into an artifact, preventing out-of-band updates |
| **Inputs** | Raw CSV (or external URL) + config.yaml + source scripts |
| **Output** | `artifacts/data_ingestion/WA_Fn-...-Churn.csv` |
| **Cache** | Invalidated if data or ingestion script changes |

### Stage 1: `validate_raw`

**Command:** `uv run python -m src.pipeline.stage_01_data_validation`

| Property | Value |
|---|---|
| **Purpose** | Validates the raw Telco CSV against GX suite before enrichment |
| **Inputs** | Raw CSV + schema.yaml + config.yaml + source scripts |
| **Output** | `artifacts/data_validation/status.txt`, `validation_report.json` |
| **Cache** | Invalidated if raw data, schema, or validation code changes |

### Stage 2: `enrich_data`

**Command:** `uv run python -m src.pipeline.stage_02_data_enrichment`

| Property | Value |
|---|---|
| **Purpose** | Runs the Agentic LLM enrichment pipeline to add ticket notes |
| **Inputs** | Raw CSV + params.yaml + config.yaml + all enrichment component files |
| **Output** | `artifacts/data_enrichment/enriched_telco_churn.csv` |
| **Cache** | Invalidated if any input changes (including `params.yaml` → `model_name`, `limit`) |

> **Reproducibility Note:** Because `params.yaml` is a DVC dependency, changing
> `limit` or `model_name` will force a full re-run. This guarantees that the enriched
> artifact always reflects the exact parameters used to create it.

### Stage 3: `validate_enriched`

**Command:** `uv run python -m src.pipeline.stage_03_enriched_validation`

| Property | Value |
|---|---|
| **Purpose** | Validates the LLM-generated columns before promoting to Feature Store |
| **Inputs** | Enriched CSV + schema.yaml + config.yaml + validation scripts |
| **Output** | `artifacts/data_enrichment/status.txt`, `validation_report.json` |
| **Cache** | Invalidated if enriched artifact or validation code changes |

### Stage 4: `feature_engineering`

**Command:** `uv run python -m src.pipeline.stage_04_feature_engineering`

| Property | Value |
|---|---|
| **Purpose** | Transforms validated data into ML-ready features (Scalar, OHE, NLP Embeddings + PCA) |
| **Inputs** | Enriched CSV + params.yaml + config.yaml + feature engineering scripts |
| **Output** | `train_features.csv`, `test_features.csv`, `val_features.csv`, `preprocessor.pkl` |
| **Cache** | Invalidated if data, transformers (utils), or configuration params change |

---

## 4. Full `dvc.yaml` Definition

```yaml
stages:
  data_ingestion:
    cmd: uv run python -m src.pipeline.stage_00_data_ingestion
    deps:
      - config/config.yaml
      - src/pipeline/stage_00_data_ingestion.py
      - src/components/data_ingestion.py
      - data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
    outs:
      - artifacts/data_ingestion/WA_Fn-UseC_-Telco-Customer-Churn.csv

  validate_raw:
    cmd: uv run python -m src.pipeline.stage_01_data_validation
    deps:
      - artifacts/data_ingestion/WA_Fn-UseC_-Telco-Customer-Churn.csv
      - src/pipeline/stage_01_data_validation.py
      - src/components/data_validation.py
      - src/config/configuration.py
      - src/utils/logger.py
      - src/utils/exceptions.py
      - config/config.yaml
      - config/schema.yaml
    outs:
      - artifacts/data_validation/status.txt
      - artifacts/data_validation/validation_report.json

  enrich_data:
    cmd: uv run python -m src.pipeline.stage_02_data_enrichment
    deps:
      - artifacts/data_ingestion/WA_Fn-UseC_-Telco-Customer-Churn.csv
      - src/pipeline/stage_02_data_enrichment.py
      - src/components/data_enrichment/orchestrator.py
      - src/components/data_enrichment/generator.py
      - src/components/data_enrichment/schemas.py
      - src/components/data_enrichment/prompts.py
      - src/config/configuration.py
      - src/utils/logger.py
      - config/config.yaml
      - config/params.yaml
    outs:
      - artifacts/data_enrichment/enriched_telco_churn.csv:
          persist: true  # This ensures DVC preserves existing data

  validate_enriched:
    cmd: uv run python -m src.pipeline.stage_03_enriched_validation
    deps:
      - artifacts/data_enrichment/enriched_telco_churn.csv
      - src/pipeline/stage_03_enriched_validation.py
      - src/components/data_validation.py
      - src/config/configuration.py
      - src/utils/logger.py
      - config/config.yaml
    outs:
      - artifacts/data_enrichment/status.txt
      - artifacts/data_enrichment/validation_report.json

  feature_engineering:
    cmd: uv run python -m src.pipeline.stage_04_feature_engineering
    deps:
      - artifacts/data_enrichment/enriched_telco_churn.csv
      - src/pipeline/stage_04_feature_engineering.py
      - src/components/feature_engineering.py
      - src/utils/feature_utils.py
      - src/config/configuration.py
      - src/utils/logger.py
      - config/config.yaml
      - config/params.yaml
    outs:
      - artifacts/feature_engineering/train_features.csv
      - artifacts/feature_engineering/test_features.csv
      - artifacts/feature_engineering/val_features.csv
      - artifacts/feature_engineering/preprocessor.pkl
```

---

## 5. Configuration Hierarchy

```mermaid
graph TD
    subgraph "Source of Truth (Tracked by DVC)"
        YAML["config/params.yaml\n(model, limit)"]
        Schema["config/schema.yaml\n(column definitions)"]
        Config["config/config.yaml\n(paths & directories)"]
    end

    CM["ConfigurationManager\n(single entry point)"]

    DC["DataEnrichmentConfig\n(frozen dataclass)"]
    DVC_RAW["DataValidationConfig\n(frozen dataclass)"]

    YAML --> CM
    Schema --> CM
    Config --> CM
    CM --> DC
    CM --> DVC_RAW
```

**Configuration precedence is strict:**
1. `config/params.yaml` → tunable hyperparameters (LLM model, limits)
2. `config/schema.yaml` → data contracts (column presence, types)
3. `config/config.yaml` → artifact paths and directories
4. Hardcoded defaults in `ConfigurationManager` → last resort fallback

**Environment variables are NOT used** to override DVC-tracked parameters, ensuring
that every `dvc repro` call is fully deterministic and reproducible.

---

## 6. Reproducing the Pipeline

```bash
# Run all stages (uses DVC cache if unchanged)
uv run dvc repro

# Force re-run of all stages (ignoring cache)
uv run dvc repro --force

# Run a specific stage only
uv run dvc repro enrich_data

# Inspect the pipeline DAG
uv run dvc dag
```

---

## 7. Planned Future Stages

The following stages will be added as the project progresses through the FTI pattern:

| Stage Name | Phase | Description |
|---|---|---|
| `train_model` | Training | XGBoost/LightGBM with MLflow tracking |
| `evaluate_model` | Training | Champion/challenger comparison gate |
| `serve_model` | Inference | FastAPI deployment trigger |
