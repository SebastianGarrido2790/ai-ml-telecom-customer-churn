## Phase 7A: Containerization — Full Implementation Plan

> Confirmed Decisions: **Decision 1 (E1 — bake model)**, **Decision 2 (F1 — bind mount artifacts)**, **Decision 3 (G1 — file store)**.

---

### Files to Deliver (11 total)

| # | File | Type |
|---|---|---|
| 1 | `docker/embedding_service/Dockerfile` | New |
| 2 | `docker/prediction_api/Dockerfile` | New |
| 3 | `docker/mlflow_server/Dockerfile` | New |
| 4 | `docker-compose.yaml` | New |
| 5 | `.dockerignore` | New |
| 6 | `.env.example` | New (compliance) |
| 7 | `config/config.yaml` | Modified (env var override for host) |
| 8 | `src/api/embedding_service/main.py` | Modified (HF cache path log) |
| 9 | `src/config/configuration.py` | Modified (env var host override) |
| 10 | `Makefile` | New (compliance) |
| 11 | `validate_system.sh` | New (Multi-Point Validation Gate) |

---

### Dockerfile Architecture

Both service Dockerfiles use a **two-stage build**:

```
Stage 1 (builder):  python:3.11-slim
  - Install uv
  - Install all dependencies from pyproject.toml into /app/.venv
  - Download SentenceTransformer model into /root/.cache/huggingface (E1 only)

Stage 2 (runtime): python:3.11-slim
  - Copy .venv from builder (no build tools in final image)
  - Copy src/ and config/
  - Run as non-root user (appuser)
  - EXPOSE port
  - HEALTHCHECK
  - CMD uvicorn
```

Key decisions enforced:
- Non-root user (`RUN adduser --disabled-password appuser`)
- `COPY` not `ADD` for all local files
- `.dockerignore` excludes `artifacts/`, `.venv/`, `mlruns/`, `__pycache__/`, `.dvc/`, `tests/`
- No `version:` key in `docker-compose.yaml` (deprecated)
- Health check on both services using the `/v1/health` endpoint
- `depends_on: condition: service_healthy` on prediction-api → embedding-service
- `start_period: 30s` on embedding service health check to accommodate warmup

---

### `docker-compose.yaml` Service Map

```
services:
  embedding-service   build: docker/embedding_service/   port: 8001
  prediction-api      build: docker/prediction_api/      port: 8000
  mlflow-server       build: docker/mlflow_server/        port: 5000
  gradio-ui           (placeholder — Phase 7B)            port: 7860

volumes:
  mlruns_data:        # named volume for MLflow file store

networks:
  churn-net:          # isolated bridge network for inter-service communication
```

---

### `config.yaml` Host Override

The `api.embedding_service.host` is currently `127.0.0.1` (correct for local terminals). Inside Docker Compose, services communicate via container DNS names (`embedding-service`). The host value needs to be overridable without changing the YAML file.

`ConfigurationManager.get_prediction_api_config()` will be updated to read `EMBEDDING_SERVICE_HOST` from the environment and fall back to the YAML value when the variable is not set:

```python
embed_host = os.environ.get(
    "EMBEDDING_SERVICE_HOST",
    api_cfg.embedding_service.host  # "127.0.0.1" local default
)
```

`docker-compose.yaml` sets `EMBEDDING_SERVICE_HOST=embedding-service` via the `environment:` key on the prediction-api service. No YAML change required for local vs. Docker — the same `config.yaml` works in both contexts.

---

### Startup Sequence (Docker Compose)

```
1. mlflow-server starts        (no dependencies)
2. embedding-service starts    (no dependencies)
   └─ warmup: ~13s for SentenceTransformer load
   └─ /v1/health → 200 only after warmup completes
3. prediction-api starts       (depends_on: embedding-service: service_healthy)
   └─ waits until embedding-service health check passes
   └─ loads 4 artifacts + creates InferenceService
4. gradio-ui starts            (Phase 7B — depends_on: prediction-api)
```

---

### `Makefile` Targets (compliance)

```makefile
lint          ruff check + ruff format --check
typecheck     pyright
test          pytest tests/ --cov=src --cov-fail-under=65
pipeline      uv run dvc repro
build         docker compose build
up            docker compose up -d
down          docker compose down
logs          docker compose logs -f
health        curl health endpoints on 8000 and 8001
validate      ./validate_system.sh
```

---

### `validate_system.sh` (Multi-Point Validation Gate)

Four-pillar gate:
1. `pyright` + `ruff check` — zero errors
2. `pytest --cov=src --cov-fail-under=65` — coverage gate
3. `dvc status` — no untracked modified artifacts
4. `curl /v1/health` on both ports — HTTP 200

---

## Integration Guide

All 11 files written. Copy to outputs. All 9 deliverable files presented (the `embedding_service/main.py` warmup fix was already delivered in the previous session). Here is the complete integration guide.

### First run sequence

```bash
# 1. Copy env template
cp .env.example .env
# Edit .env — add GOOGLE_API_KEY at minimum

# 2. Build all images
# Note: embedding-service build takes ~3-5 min — downloads the SentenceTransformer
# model (~90MB) during the build layer. Subsequent builds are cached.
docker compose build

# 3. Start all services (detached)
docker compose up -d

# 4. Watch embedding-service warm up (~13s after container starts)
docker compose logs -f embedding-service

# 5. Verify all services healthy
curl http://localhost:8000/v1/health
curl http://localhost:8001/v1/health

# 6. Stopping and removing containers
docker compose down
```

### What `docker compose build` does differently from the local setup

The embedding service Dockerfile runs `SentenceTransformer('all-MiniLM-L6-v2')` during `docker build` — this is the E1 bake. You will see the HuggingFace download progress in your build output once, then never again until `docker compose build-no-cache` is run. Every `docker compose up` after that starts instantly without any model download.

### The host override in action

When running locally (two terminals): `EMBEDDING_SERVICE_HOST` is not set → `configuration.py` falls back to `config.yaml`'s `127.0.0.1`. When running in Docker Compose: `docker-compose.yaml` sets `EMBEDDING_SERVICE_HOST=embedding-service` on the prediction-api service → the `os.environ.get()` in `configuration.py` returns `embedding-service` → the URL becomes `http://embedding-service:8001` → Docker's internal DNS resolves it to the embedding container. Same `config.yaml`, same code, two environments.

---

## Using The Services

Once you have verified that the services are up and healthy, you have several ways to interact with the **Telecom Churn MLOps** stack.

### 1. Quick Health Verification
Run the consolidated health check to verify that both the **NLP Embedding** and **Late Fusion Prediction** APIs are ready:

```bash
curl http://localhost:8000/v1/health
curl http://localhost:8001/v1/health
```

---

### 2. Explore the Experiment Registry (MLflow)
You can view all past training runs, model metrics (F1, AUC), and the registered **Late Fusion** model in the browser:
*   **URL:** [http://localhost:5000](http://localhost:5000)
*   **What to look for:** Check the `telco-churn-late-fusion` experiment to see the performance of the model currently loaded in the containers.

---

### 3. Send a Real-Time Prediction Request
The **Prediction API** expects a JSON payload containing 19 structured fields plus a `ticket_note`. Here is a sample `curl` command you can run in your terminal:

```bash
curl -X POST http://localhost:8000/v1/predict \
     -H "Content-Type: application/json" \
     -d '{
        "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
        "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check", "MonthlyCharges": 70.85, "TotalCharges": "840.12",
        "ticket_note": "Customer is complaining about slow internet speeds and considering switching."
     }'
```

```powershell
$json = '{"gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "tenure": 1, "PhoneService": "No", "MultipleLines": "No phone service", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check", "MonthlyCharges": 95.50, "TotalCharges": "95.50", "ticket_note": "Customer called about high monthly charges."}'
Invoke-RestMethod -Method Post -Uri "http://localhost:8000/v1/predict" -ContentType "application/json" -Body $json
```

**Expected Response:** You will get a `churn_probability` that combines the **Structured Branch** and the **NLP Branch** (processing the `ticket_note`).

```powershell
customerID           : 
churn_probability    : 0,700559
churn_prediction     : True
p_structured         : 0,889536
p_nlp                : 0,407729
nlp_branch_available : True
model_version        : late-fusion-v2
```

---

### 4. Interactive API Documentation (Swagger UI)
FastAPI automatically generates interactive documentation for both microservices. You can use these to test the endpoints without writing any code:

*   **Prediction API Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)
*   **Embedding Service Docs:** [http://localhost:8001/docs](http://localhost:8001/docs)

### 5. Troubleshooting with Logs
If you want to see the "Late Fusion" logic in action (including the circuit-breaker logs), you can stream the logs from all containers:

```bash
docker compose logs -f
```

---

Once the **Late Fusion Prediction** is working, you we'll see in the result a perfect example of how the two branches of the model cooperate:

*   **Branch 1 (Tabular):** `0.889`. The structured data (tenure, charges, etc.) thinks this customer is very likely to churn.
*   **Branch 2 (NLP):** `0.407`. The ticket note ("Customer called about high monthly charges") is seen as a risk, but isn't as definitive as the structured data.
*   **Final Result (Meta-Model):** `0.700`. The meta-learner combined both signals to give you a final churn risk of **70%**.
*   **`nlp_branch_available: True`**: This confirms the **Embedding Service** is correctly talking to the **Prediction API** inside the Docker network.
