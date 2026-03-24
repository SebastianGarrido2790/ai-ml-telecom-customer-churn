## Architectural Reassessment: Dual-Enhancement Plan

The critical insight is that both options share a **single foundational prerequisite**: Phase 4 must be modified. The current monolithic `preprocessor.pkl` makes both a Late Fusion experiment (Option A) and a decoupled Embedding Microservice (Option B) architecturally impossible without re-engineering inference. This modification is the load-bearing change — everything else follows from it.

---

## Phase 4 Modification: The Two-Preprocessor Split

**Current state:** One `preprocessor.pkl` containing three transformers (numeric, categorical, NLP) serialized together.

**Required state:** Two independent, separately serialized preprocessors, both fitted exclusively on the train set.

| Artifact | Contents | Consumers |
|---|---|---|
| `structured_preprocessor.pkl` | `NumericCleaner` + `SimpleImputer` + `StandardScaler` (numeric), `SimpleImputer` + `OneHotEncoder` (categorical) | Phase 5 Branch 1, Phase 6 Prediction API |
| `nlp_preprocessor.pkl` | `SimpleImputer` + `TextEmbedder` + `PCA` (NLP) | Phase 5 Branch 2, Phase 6 **Embedding Microservice** |

The train/val/test CSVs remain unchanged — they will still contain all merged features. The split happens at the serialization step, not the data storage step. The output of `stage_04_feature_engineering` therefore gains one additional DVC-tracked artifact.

**Rationale for this design (vs. alternatives):**

- **Alternatives considered:** (1) Splitting at inference time by column prefix (`nlp__*` vs `num__*` / `cat__*`) from the unified output — avoids re-engineering Phase 4 but makes the Anti-Skew guarantee implicit rather than explicit. (2) Keeping one preprocessor, duplicating logic in the embedding service — violates Rule 2.9 (DRY for features).
- **This design wins** because `nlp_preprocessor.pkl` IS the embedding service's artifact. Zero duplication. The PCA fitted parameters are identical between training and serving because they come from the same file.

---

## Option A: Late Fusion Architecture (Phase 5)

### Model Architecture

```
                ┌─────────────────────────────────────────┐
                │         Train Features (CSV)             │
                └──────────────┬──────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
  ┌─────────────────────┐         ┌────────────────────────┐
  │  Branch 1           │         │  Branch 2              │
  │  Structured Only    │         │  NLP Embeddings Only   │
  │                     │         │                        │
  │  structured_         │         │  nlp_preprocessor.pkl  │
  │  preprocessor.pkl   │         │  → 20 PCA components   │
  │  → numeric + OHE    │         │                        │
  │  + sentiment_tag    │         │  XGBoost / LightGBM    │
  │                     │         │  (tuned via Optuna)    │
  │  XGBoost (primary)  │         │                        │
  └──────────┬──────────┘         └───────────┬────────────┘
             │  P(churn | structured)          │  P(churn | nlp)
             └───────────────┬────────────────┘
                             ▼
              ┌──────────────────────────────┐
              │   Meta-Learner               │
              │   Logistic Regression        │
              │   (stacking via             │
              │    cross_val_predict OOF)    │
              └──────────────┬───────────────┘
                             ▼
                    Final Churn Score
```

### Training Protocol (Anti-Leakage Stacker)

The meta-learner must be trained on **Out-of-Fold (OOF) predictions**, not on the same training set used to train the base models. This is the standard stacking protocol that prevents leakage into the meta-learner:

1. Use `cross_val_predict(method='predict_proba')` on the train set for both Branch 1 and Branch 2 models — producing OOF probability arrays.
2. Stack the OOF arrays as input features for the Logistic Regression meta-learner.
3. Retrain both base models on the full train set.
4. At evaluation, get predictions from the fully retrained base models on val/test, stack them, and pass through the fitted meta-learner.

### MLflow Experiment Structure

Three tracked runs per experiment cycle:

| Run Name | Primary Metric | Tags |
|---|---|---|
| `structured_baseline` | Recall, F1, ROC-AUC | `branch=structured`, `model=xgboost` |
| `nlp_baseline` | Recall, F1, ROC-AUC | `branch=nlp`, `model=xgboost` |
| `late_fusion_stacked` | Recall, F1, ROC-AUC | `branch=fusion`, `model=stacked_lr` |

The fusion run logs the **delta** over the structured baseline as a custom metric: `recall_lift` and `f1_lift`. This is the business-facing number that proves the ROI of Phase 2's AI enrichment.

### New Config Entity Required

```python
@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    val_data_path: Path
    test_data_path: Path
    structured_preprocessor_path: Path
    nlp_preprocessor_path: Path
    structured_model_path: Path   # artifacts/model_training/structured_model.pkl
    nlp_model_path: Path          # artifacts/model_training/nlp_model.pkl
    meta_model_path: Path         # artifacts/model_training/meta_model.pkl
    target_column: str
    random_state: int
    mlflow_uri: str
    experiment_name: str
    cv_folds: int
```

### New `params.yaml` Section

```yaml
model_training:
  cv_folds: 5
  structured_branch:
    algorithm: "xgboost"     # Optuna will search over this space
    n_trials: 30
  nlp_branch:
    algorithm: "xgboost"
    n_trials: 20
  meta_learner:
    algorithm: "logistic_regression"
    C: 1.0
    max_iter: 1000
```

---

## Option B: Embedding Microservice Architecture (Phase 6)

### Service Topology (docker-compose)

```
┌─────────────────────────────────────────────────────────────────┐
│                        docker-compose                           │
│                                                                 │
│  ┌──────────────────────┐    ┌─────────────────────────────┐   │
│  │  embedding-service   │    │  prediction-api              │   │
│  │  Port: 8001          │    │  Port: 8000                  │   │
│  │                      │    │                              │   │
│  │  FastAPI             │    │  FastAPI                     │   │
│  │  nlp_preprocessor    │    │  structured_preprocessor     │   │
│  │  .pkl (loaded at     │    │  .pkl + meta_model.pkl +     │   │
│  │  startup)            │    │  branch models (loaded at    │   │
│  │                      │    │  startup via lifespan)       │   │
│  │  POST /v1/embed      │◄───│  POST /v1/predict           │   │
│  │  GET  /v1/health     │    │  POST /v1/predict/batch      │   │
│  │                      │    │  GET  /v1/health             │   │
│  └──────────────────────┘    └─────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────┐    ┌─────────────────────────────┐   │
│  │  mlflow-server       │    │  gradio-ui                  │   │
│  │  Port: 5000          │    │  Port: 7860                  │   │
│  └──────────────────────┘    └─────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Embedding Service Contract

```python
# POST /v1/embed
class EmbedRequest(BaseModel):
    ticket_notes: list[str]      # batch support

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]  # shape: (n, pca_components)
    model_version: str
    dim: int                       # = pca_components (20)
```

### Prediction API Inference Flow

```
Client → POST /v1/predict (CustomerFeatureRequest)
              │
              ├──► structured_preprocessor.pkl → structured feature vector
              │
              ├──► POST embedding-service:8001/v1/embed (ticket_note) → PCA vector
              │
              ├──► Branch 1 model.predict_proba(structured_vector) → P1
              ├──► Branch 2 model.predict_proba(pca_vector) → P2
              │
              └──► meta_model.predict_proba([[P1, P2]]) → final churn score
                            │
                    ChurnPredictionResponse
```

### Circuit Breaker for Embedding Service

The prediction API must implement a fallback if the embedding service is unreachable — a Rule 2.2 (Custom Exception Handling) requirement. If `POST /v1/embed` returns a non-200 or times out, the API falls back to a zero-vector of dimension 20 and adds `"nlp_branch_available": false` to the response. The structured Branch 1 prediction continues uninterrupted.

### New Config Entity Required

```python
@dataclass(frozen=True)
class EmbeddingServiceConfig:
    host: str          # "embedding-service" in Docker, "localhost" locally
    port: int          # 8001
    timeout_seconds: float
    nlp_preprocessor_path: Path

@dataclass(frozen=True)
class PredictionAPIConfig:
    host: str
    port: int          # 8000
    structured_preprocessor_path: Path
    structured_model_path: Path
    nlp_model_path: Path
    meta_model_path: Path
    embedding_service_url: str   # constructed from EmbeddingServiceConfig
```

---

## Updated DVC Pipeline DAG

```
Stage 0: data_ingestion
    ↓
Stage 1: validate_raw
    ↓
Stage 2: enrich_data
    ↓
Stage 3: validate_enriched
    ↓
Stage 4: feature_engineering          ← MODIFIED
    outputs:
    - train/val/test_features.csv
    - structured_preprocessor.pkl     ← NEW
    - nlp_preprocessor.pkl            ← NEW (replaces unified preprocessor.pkl)
    ↓
Stage 5: train_model                  ← NEW
    deps: train/val/test CSVs, both preprocessors
    outputs:
    - structured_model.pkl
    - nlp_model.pkl
    - meta_model.pkl
    - mlflow run artifacts (eval_report.json)
    ↓
Stage 6: evaluate_model               ← MODIFIED (champion/challenger gate)
    deps: test CSV, all models
    outputs: evaluation_report.json, champion_tag (MLflow registry tag)
```

---

## Full Directory Structure Changes

```text
src/
├── api/                              ← BUILT IN PHASE 6
│   ├── __init__.py
│   ├── embedding_service/
│   │   ├── __init__.py
│   │   ├── main.py                   # FastAPI lifespan, app factory
│   │   ├── router.py                 # /v1/embed, /v1/health
│   │   └── schemas.py                # EmbedRequest, EmbedResponse
│   └── prediction_service/
│       ├── __init__.py
│       ├── main.py
│       ├── router.py                 # /v1/predict, /v1/predict/batch, /v1/health
│       └── schemas.py                # CustomerFeatureRequest, ChurnPredictionResponse
│
├── components/
│   ├── model_training/               ← NEW IN PHASE 5
│   │   ├── __init__.py
│   │   ├── trainer.py                # Late Fusion training + OOF stacking
│   │   └── evaluator.py              # MLflow logging, champion comparison
│   └── ...existing...
│
├── pipeline/
│   ├── stage_05_model_training.py    ← NEW
│   └── stage_06_model_evaluation.py  ← NEW
│
artifacts/
├── feature_engineering/
│   ├── train_features.csv
│   ├── val_features.csv
│   ├── test_features.csv
│   ├── structured_preprocessor.pkl   ← NEW
│   └── nlp_preprocessor.pkl          ← NEW
│
├── model_training/                   ← NEW
│   ├── structured_model.pkl
│   ├── nlp_model.pkl
│   ├── meta_model.pkl
│   └── evaluation_report.json
```

---

## Summary of Files Requiring Changes

| File | Change Type | Reason |
|---|---|---|
| `src/components/feature_engineering.py` | Modify | Split `get_preprocessor()` into two methods; serialize two pkl files |
| `src/entity/config_entity.py` | Additive | Add `ModelTrainingConfig`, `EmbeddingServiceConfig`, `PredictionAPIConfig` |
| `src/config/configuration.py` | Additive | Add corresponding `get_*_config()` methods |
| `config/config.yaml` | Additive | Add `model_training` and `api` path sections |
| `config/params.yaml` | Additive | Add `model_training` hyperparameter section |
| `dvc.yaml` | Modify + Additive | Update Stage 4 outputs; add Stage 5 & 6 |
| `docker-compose.yaml` | Populate | Define all four services |

---

## Recommended Execution Sequence

**Step 1:** Modify Phase 4 — split the preprocessor. This is backward-compatible with existing DVC artifacts since it only adds outputs, and it unblocks both downstream phases. The train/val/test CSVs do not change.

**Step 2:** Implement Phase 5 (model training) — Late Fusion with MLflow tracking and Optuna tuning.

**Step 3:** Implement Phase 6 (serving) — embedding microservice first, then prediction API, then docker-compose integration.

This sequencing respects the FTI pipeline order and ensures each step produces testable artifacts before the next begins.
