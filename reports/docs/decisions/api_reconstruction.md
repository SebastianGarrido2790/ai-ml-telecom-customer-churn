## API Architectural Decision

The `structured_preprocessor.pkl` contains a `ColumnTransformer` that expects a pandas DataFrame with the exact column names from training (`gender`, `SeniorCitizen`, etc.). At inference time, the raw `CustomerFeatureRequest` fields must be reconstructed into a DataFrame before passing to the preprocessor.

Two options for where this reconstruction happens:

**Option D1 — In `router.py` (endpoint layer):** The router converts the Pydantic model to a DataFrame inline before calling the preprocessor. Simple, but mixes transformation logic with HTTP routing.

**Option D2 — In a dedicated `inference.py` service module:** A standalone `InferenceService` class (inside `prediction_service/`) owns the DataFrame reconstruction, preprocessor application, embedding service call, and meta-model prediction. The router is a pure HTTP conductor — it validates input, calls `InferenceService.predict()`, and returns the response. This is the correct separation of concerns per the Single Responsibility Principle (SRP).

**Recommendation: Option D2.** Adds `src/api/prediction_service/inference.py` as the 14th file. The router stays clean (< 30 lines per endpoint), the inference logic is independently testable, and the circuit breaker logic lives in one place.