## Phase 6: Full Implementation Plan

**Decision Confirmed: D2 (Separated)**

The three-layer separation in Phase 6 follows Brain vs. Brawn Rule and the Single Responsibility Principle (SRP). Here is the complete ownership map across all 14 files.

The three-layer architecture across both services follows a strict boundary: configuration owns artifact paths, `main.py` owns startup/shutdown, `router.py` owns HTTP, `inference.py` owns all computation, and schemas own validation. Here is the complete ownership map with the rationale for each boundary.Each boundary enforces a specific rule.

**`config.yaml` and `config_entity.py`** are the only files that know where artifacts live on disk. Every other file receives paths through dependency injection via `ConfigurationManager`. This means moving an artifact to a different directory requires changing one YAML key — nothing else.

**`main.py` (both services)** is the only file that calls `joblib.load()`. Artifact deserialization is a startup concern, not a request concern. Loading inside an endpoint handler would deserialize on every request — a performance failure. The lifespan pattern guarantees exactly one load, one teardown, with the loaded objects stored on `app.state` and accessible to all request handlers without global state.

**`router.py` (both services)** contains no `import numpy`, `import pandas`, or `import sklearn`. Its only job is to receive a validated Pydantic model, call one method on `app.state`, and return a typed response. Each endpoint is under 15 lines. This makes the routing layer trivially testable with FastAPI's `TestClient` without mocking any ML dependencies.

**`inference.py`** is the only file that knows about the Late Fusion architecture — it knows there are two branches, that they are stacked with `np.column_stack`, and that the circuit breaker zero-vector must have shape `(n, pca_components)`. If the model architecture changes in a future retraining (e.g. adding a third branch), only `inference.py` changes. The router, schemas, and config are untouched.

**`schemas.py` (both services)** has no business logic beyond Pydantic constraints. `TotalCharges: str | None` is a constraint, not logic — it reflects the raw dataset type. The preprocessing decision (converting `None` to `""` for `NumericCleaner`) lives in `inference.py._build_structured_df`, not in the schema.

**`test_api_schemas.py`** uses `MagicMock` for all model artifacts and `AsyncMock` with `patch` for all HTTP calls. It never imports `joblib`, `xgboost`, or `sentence_transformers`. This keeps the test suite fast (< 1s) and runnable without any model artifacts present — critical for CI/CD where artifacts are not checked into the repository.

The dashed arrow between the two services in the diagram represents the only coupling between them: a single `POST /v1/embed` call inside `InferenceService._get_embeddings`. The circuit breaker means this is a soft coupling — the prediction API degrades gracefully when the embedding service is absent, never propagating a 5xx to the caller.

---

### Service Topology

```
docker-compose (Phase 7)
│
├── embedding-service   :8001   src/api/embedding_service/
│   └── Loads: nlp_preprocessor.pkl
│   └── Exposes: POST /v1/embed, GET /v1/health
│
├── prediction-api      :8000   src/api/prediction_service/
│   └── Loads: structured_preprocessor.pkl, structured_model.pkl,
│              nlp_model.pkl, meta_model.pkl
│   └── Exposes: POST /v1/predict, POST /v1/predict/batch, GET /v1/health
│   └── Calls: embedding-service:8001/v1/embed (with circuit breaker)
│
└── [mlflow-server, gradio-ui — Phase 7]
```

---

### Inference Flow: `POST /v1/predict`

```
CustomerFeatureRequest (19 structured fields + ticket_note)
        │
        ├─── structured_preprocessor.pkl ──► 46-dim structured vector
        │
        ├─── POST :8001/v1/embed (ticket_note) ──► 20-dim PCA vector
        │         │
        │    [Circuit Breaker: timeout/error → zero-vector(20), nlp_branch_available=False]
        │
        ├─── structured_model.predict_proba(struct_vec) ──► P_struct
        ├─── nlp_model.predict_proba(nlp_vec) ──► P_nlp
        │
        └─── meta_model.predict_proba([[P_struct, P_nlp]]) ──► ChurnPredictionResponse
```

---

### Pydantic Schema Design

**`EmbedRequest` / `EmbedResponse`** (embedding service):
```python
class EmbedRequest(BaseModel):
    ticket_notes: list[str]         # batch — list allows bulk calls from batch endpoint

class EmbedResponse(BaseModel):
    embeddings: list[list[float]]   # shape: (n, 20)
    model_version: str              # e.g. "all-MiniLM-L6-v2-pca20"
    dim: int                        # = 20 (pca_components)
```

**`CustomerFeatureRequest`** (prediction API — raw pre-preprocessed fields):
```python
class CustomerFeatureRequest(BaseModel):
    customerID: str | None = None   # optional — for traceability, not used in model
    gender: str
    SeniorCitizen: int              # ge=0, le=1
    Partner: str
    Dependents: str
    tenure: int                     # ge=0
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float           # ge=0
    TotalCharges: str | None        # blank strings exist for tenure=0
    ticket_note: str                # required for NLP branch
```

**`ChurnPredictionResponse`**:
```python
class ChurnPredictionResponse(BaseModel):
    customerID: str | None
    churn_probability: float        # meta-model output, range [0, 1]
    churn_prediction: bool          # threshold=0.5
    p_structured: float             # Branch 1 probability (for interpretability)
    p_nlp: float                    # Branch 2 probability (0.0 if circuit breaker)
    nlp_branch_available: bool      # False when embedding service unreachable
    model_version: str              # "late-fusion-v2"
```

**`BatchPredictRequest`** and **`BatchPredictResponse`** wrap lists of the above.

---

### New Config Entities Required

Two new frozen dataclasses in `config_entity.py`:

```python
@dataclass(frozen=True)
class EmbeddingServiceConfig:
    host: str               # "localhost" (local) / "embedding-service" (Docker)
    port: int               # 8001
    timeout_seconds: float  # 5.0
    nlp_preprocessor_path: Path
    model_version: str      # "all-MiniLM-L6-v2-pca20"

@dataclass(frozen=True)
class PredictionAPIConfig:
    host: str               # "0.0.0.0"
    port: int               # 8000
    structured_preprocessor_path: Path
    structured_model_path: Path
    nlp_model_path: Path
    meta_model_path: Path
    embedding_service_url: str  # constructed: f"http://{host}:{port}"
    model_version: str          # "late-fusion-v2"
    pca_components: int         # 20 — for zero-vector fallback
```

New `api` section in `config.yaml`:
```yaml
api:
  embedding_service:
    host: "localhost"
    port: 8001
    timeout_seconds: 5.0
    model_version: "all-MiniLM-L6-v2-pca20"
  prediction_api:
    host: "0.0.0.0"
    port: 8000
    model_version: "late-fusion-v2"
```

---

### Files to Create or Modify

| File | Action | Purpose |
|---|---|---|
| `config/config.yaml` | Modify | Add `api` section |
| `src/entity/config_entity.py` | Modify | Add `EmbeddingServiceConfig`, `PredictionAPIConfig` |
| `src/config/configuration.py` | Modify | Add `get_embedding_service_config()`, `get_prediction_api_config()` |
| `src/api/__init__.py` | Already exists | Package marker (confirm only) |
| `src/api/embedding_service/__init__.py` | Create | Package marker |
| `src/api/embedding_service/schemas.py` | Create | `EmbedRequest`, `EmbedResponse` |
| `src/api/embedding_service/router.py` | Create | `/v1/embed`, `/v1/health` endpoints |
| `src/api/embedding_service/main.py` | Create | FastAPI app factory, lifespan |
| `src/api/prediction_service/__init__.py` | Create | Package marker |
| `src/api/prediction_service/schemas.py` | Create | `CustomerFeatureRequest`, `ChurnPredictionResponse`, batch variants |
| `src/api/prediction_service/router.py` | Create | `/v1/predict`, `/v1/predict/batch`, `/v1/health` endpoints |
| `src/api/prediction_service/main.py` | Create | FastAPI app factory, lifespan, circuit breaker |
| `tests/unit/test_api_schemas.py` | Create | Schema validation, circuit breaker fallback logic, response contracts |

**Total: 3 modified + 10 created = 13 files.**

---

### Local execution sequence
For this system to work, you must have both services running simultaneously.

```bash
# Terminal 1 — start prediction API (requires embedding service running)
uv run uvicorn src.api.prediction_service.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — start embedding service
uv run uvicorn src.api.embedding_service.main:app --host 0.0.0.0 --port 8001

# Test health endpoints
curl http://localhost:8000/v1/health
curl http://localhost:8001/v1/health

# Test single prediction
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"gender":"Female","SeniorCitizen":0,"Partner":"Yes","Dependents":"No",
       "tenure":1,"PhoneService":"No","MultipleLines":"No phone service",
       "InternetService":"Fiber optic","OnlineSecurity":"No","OnlineBackup":"No",
       "DeviceProtection":"No","TechSupport":"No","StreamingTV":"No",
       "StreamingMovies":"No","Contract":"Month-to-month","PaperlessBilling":"Yes",
       "PaymentMethod":"Electronic check","MonthlyCharges":95.50,
       "TotalCharges":"95.50",
       "ticket_note":"Customer called about high monthly charges."}'

# Test single prediction (PowerShell)
$body = @{
    gender = "Female"; SeniorCitizen = 0; Partner = "Yes"; Dependents = "No";
    tenure = 1; PhoneService = "No"; MultipleLines = "No phone service";
    InternetService = "Fiber optic"; OnlineSecurity = "No"; OnlineBackup = "No";
    DeviceProtection = "No"; TechSupport = "No"; StreamingTV = "No";
    StreamingMovies = "No"; Contract = "Month-to-month"; PaperlessBilling = "Yes";
    PaymentMethod = "Electronic check"; MonthlyCharges = 95.50;
    TotalCharges = "95.50";
    ticket_note = "Customer called about high monthly charges."
} | ConvertTo-Json

Invoke-RestMethod -Method Post -Uri "http://localhost:8000/v1/predict" -ContentType "application/json" -Body $body

# Stop Services
Get-Job | Remove-Job -Force

# Run tests (requires pytest-asyncio)
uv run pytest tests/unit/test_api_schemas.py -v
```
