## API Warmup Evaluation

24/24 tests passing. Both services healthy. Real prediction returned with both branches active — `churn_probability: 0.7006`, `nlp_branch_available: True`. Phase 6 is complete.

Two configuration findings worth documenting before moving to Phase 7.

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

```bash
# Restart both services to pick up the config and code changes
# (CTRL+C the existing processes, then)
uv run uvicorn src.api.embedding_service.main:app --host 0.0.0.0 --port 8001
uv run uvicorn src.api.prediction_service.main:app --host 0.0.0.0 --port 8000
```
