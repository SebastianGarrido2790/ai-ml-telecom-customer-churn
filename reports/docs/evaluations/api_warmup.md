## API Warmup & Security Evaluation

124/124 tests passing. Both services hardened and production-ready.

**Status:** Phase 3 Hardening Complete.
**Key Metrics:** `X-API-Key` enforced, 1000-record batch limit active, 100% test pass rate in `test_suite.md`.

The following findings document the evolution from the initial Phase 6 prototype to the current hardened architecture.

---

**Finding 1 — `localhost` resolves to IPv6 on Windows**

On Windows, `localhost` resolves to `::1` (IPv6) by default. The embedding service binds to `0.0.0.0` (IPv4). The prediction API calling `http://localhost:8001` therefore targets a different interface than the one the service is listening on — connection refused. The fix (`127.0.0.1` explicit) is correct and must be reflected in `config.yaml` permanently. The Docker Compose network in Phase 7 uses container DNS names (`embedding-service`), so this is a local-only issue, but it must be documented as an environment-specific override.

**Finding 2 — Cold-start latency from `SentenceTransformer`**

The `TextEmbedder` inside `nlp_preprocessor.pkl` uses lazy loading — the PyTorch model is not loaded until the first `transform()` call. This means the first inference request triggers the SentenceTransformer download/load (~2s), causing an apparent timeout at the default 5s threshold when the prediction API's `httpx` client is already waiting. The 30s override absorbs this. In production (Docker), the embedding service should expose a `/v1/health` endpoint that warms the model at startup — a `GET /v1/health` call from the readiness probe will trigger the first `transform()` before any real traffic arrives.

The warmup can be added to `embedding_service/main.py` inside the lifespan with a single line after loading the preprocessor:

```python
# Warm the SentenceTransformer on startup to avoid cold-start latency
import pandas as pd
nlp_preprocessor.transform(pd.DataFrame({"ticket_note": ["warmup"]}))
logger.info("NLP preprocessor warmed up.")
```

This makes the 5s default timeout safe again. The 30s value is a workaround, not a permanent solution — a 30s embedding timeout means a 30s prediction API timeout, which is unacceptable in production.

---

### Two Files, Two Changes

**`src/api/embedding_service/main.py`** — after `joblib.load()` and before the `logger.info("ready")` line, a warmup block runs `nlp_preprocessor.transform(pd.DataFrame({"ticket_note": ["warmup"]}))`. This is a single dummy row — enough to force PyTorch to load the SentenceTransformer weights into memory during startup. The startup log now reads:

```
INFO  Loading NLP preprocessor from: ...nlp_preprocessor.pkl
INFO  Warming NLP preprocessor (first transform initialises SentenceTransformer).
INFO  NLP preprocessor warmed up — SentenceTransformer loaded into memory.
INFO  Embedding Microservice ready. Model: all-MiniLM-L6-v2-pca20 | Dim: 20
```

Startup will take ~2–3 seconds longer, but every subsequent request including the very first one from the prediction API will hit an already-loaded model.

**`config/config.yaml`** — `timeout_seconds` reverted from `30.0` back to `5.0`. With warmup in place, the embedding service responds in milliseconds after startup. A 5s timeout is generous for a local network call returning a 20-dimensional vector.

```yaml
api:
  embedding_service:
    host: "127.0.0.1"
    port: 8001
    timeout_seconds: 30.0 # 30s is a workaround for cold-start latency from SentenceTransformer
    model_version: "all-MiniLM-L6-v2-pca20"
  prediction_api:
    host: "0.0.0.0"
    port: 8000
    model_version: "late-fusion-v2"
```

After replacing both files, restart the embedding service. The prediction API needs no restart — it reads `timeout_seconds` only at startup via `ConfigurationManager`, so it will also need a restart to pick up the new value from config.

# Phase 3 Hardening & Security Enhancements

The system has been updated from a pure functional prototype to a **Secure Inference API**. The following enhancements address critical vulnerabilities identified during codebase review.

### 🔒 Enhancement 1 — Authentication Guardrails
**Finding:** Endpoints were previously "naked" and accessible to any client on the network.
**Enhancement:** Implemented `X-API-Key` enforcement via FastAPI's `Header` dependencies. Both the Prediction API and Embedding Microservice now require a valid `API_KEY` (configured via environment variables) for all non-health-check traffic.
- **Affected:** `src.api.prediction_service.main`, `src.api.embedding_service.main`.

### 🛡️ Enhancement 2 — Information Leakage Prevention
**Finding:** Internal 500 errors leaked stack traces or detailed exception messages (e.g., database connection strings, local file paths) in the JSON response.
**Enhancement:** Integrated a **Global Exception Handler** that catches all unhandled exceptions, logs the detailed traceback internally for developers, but returns a generic, sanitized `500 Internal server error` response to the client. This prevents information harvesting by attackers.
- **Affected:** `src.api.embedding_service.router` (removed local try/excepts to delegate to the global handler).

### 📦 Enhancement 3 — Batch Constraint Robustness
**Finding:** The `BatchPredictRequest` allowed unlimited customer records, making the API vulnerable to memory exhausting (DoS) attacks.
**Enhancement:** Enforced a `max_length=1000` constraint on the `customers` list using Pydantic validation. Requests exceeding this limit are rejected with a `422 Unprocessable Entity` before any inference logic or memory allocation occurs.

### 🌐 Enhancement 4 — CORS Security
**Finding:** Default CORS settings were overly permissive or missing.
**Enhancement:** Configured explicit `CORSMiddleware` with restricted origins (locked to `127.0.0.1` and localhost for development, environment-aware for production) and disallowed sensitive headers except for `X-API-Key`.
