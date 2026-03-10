# Telecom Customer Churn Prediction — Rules & Guardrails Runbook

**Project:** Telecom Customer Churn Prediction System
**Document Type:** Runbook · The Rules
**Version:** 1.0
**Date:** 2026-03-10
**Status:** Active — Authoritative Reference

---

## 1. Purpose & Scope

This runbook is the **single source of truth** for all constraints, prohibitions, coding standards, and operational guardrails governing the Telecom Customer Churn Prediction project. It applies to every human or AI agent contributing to the codebase and must be consulted before any architectural change, new feature implementation, or modification of existing pipelines.

This document does **not** describe how the system works (see `reports/docs/architecture/`) or why decisions were made (see `reports/docs/decisions/`). It describes the **boundaries within which all work must operate**.

---

## 2. Core Philosophy

> **"The Brain (Agent) directs; The Hands (Tools) execute."**

All design decisions in this system flow from this principle. LLM agents (LangGraph + Gemini) are probabilistic interpreters meant for qualitative NLP synthesis and orchestration. Deterministic Python tools are the only entities permitted to compute, fetch, rigorously validate, or transform quantitative data. Any violation of this separation is a **critical architectural defect**.

---

## 3. Python Code Standards

### 3.1 Typing

| Rule | Requirement | Enforcement |
| :--- | :--- | :--- |
| **Type Hints** | 100% coverage on all functions, methods, and class attributes | `pyright` (Strict Mode) |
| **py.typed** | Mandatory marker file in `src/` to signal PEP 561 compliance | Project Skeleton Rule |
| **Pydantic Models** | Every external input (API, agent schema, config) must use a `BaseModel` | Code review |
| **No Untyped Dicts** | `dict` must never cross module or agent/tool boundaries | Linter (`ruff`), `pyright` |

**✅ DO:**
```python
class TelcoCustomerRow(BaseModel):
    tenure: int = Field(..., ge=0, description="Number of months the customer has stayed with the company.")
    MonthlyCharges: float = Field(..., ge=0.0, description="The amount charged to the customer monthly.")
```

**❌ DO NOT:**
```python
def predict_churn(data: dict): ...   # Naked dict — rejected at review
```

### 3.2 Linting & Formatting

All code must pass the following checks **before any commit**:

```bash
uv run ruff check . --output-format=github
uv run ruff format --check .
```

Active `ruff` rule sets (from `pyproject.toml`):

| Code | Ruleset | Notes |
| :--- | :--- | :--- |
| `E`, `W` | pycodestyle | Style and whitespace/warnings |
| `F` | Pyflakes | Unused imports, undefined names |
| `I` | isort | Import ordering |
| `UP` | pyupgrade | Modern Python idioms |
| `B` | flake8-bugbear | Potential bugs and design issues |
| `SIM` | flake8-simplify | Code simplification |
| `PTH` | flake8-use-pathlib | Pathlib enforcement |

### 3.3 Docstrings

**Google-style docstrings are mandatory** on every public function and class.

```python
def generate_ticket_note(self, customer: TelcoCustomerRow) -> str:
    """Generates a synthetic qualitative ticket note for a customer based on their demographic and usage features.

    Args:
        customer: A TelcoCustomerRow containing structured telecom features.

    Returns:
        A strictly formatted text string containing the synthetic customer note.

    Raises:
        LLMGenerationError: If the downstream foundation model fails to render valid output.
    """
```

> **Why this matters:** LLM agents rely on docstrings to understand tool capabilities. Poor documentation cripples Agentic routing.

### 3.4 Dependency Management

| Rule | Requirement |
| :--- | :--- |
| **Runtime** | Always use `uv` — never `pip install` directly. |
| **Lockfile** | `uv.lock` must be committed with every dependency change. |
| **Project Config** | All metadata, dependencies, and tool config live in `pyproject.toml`. |
| **Dev extras** | Testing (`pytest`) and linting (`ruff`, `pyright`) tools must be declared under `[project.optional-dependencies] dev =`. |

---

## 4. Agentic Architecture Guardrails

### 4.1 Strict Separation — Brain vs. Hands

| Allowed for Agents (Brain) | Prohibited for Agents |
| :--- | :--- |
| Reasoning, synthesis of ticket notes, sentiment analysis | Arithmetic, scaling, encoding features |
| Generation of highly-specific NLP abstractions | Raw data fetching from Parquet/CSV via naked string manipulation |
| Tool orchestration and FTI pipelining | XGBoost model inference execution natively |
| Business interpretation of SHAP/Churn scores | Any direct `exec()` or `eval()` |

**Violation Class: CRITICAL.** If the Gemini model attempts to mathematically scale numeric features or execute statistical predictions instead of querying the ML Endpoint, the architecture has failed.

### 4.2 No Naked Prompts

System prompts are **forbidden** from being hardcoded inline.
- All system prompts must reside in a dedicated modular directory (e.g. `src/enrichment/prompts/`) or configuration files.
- Version control your prompts. If a prompt performs poorly, do not refactor the Python loop; fix the prompt artifact.

### 4.3 Structured Output Enforcement

- Agents tasked with generating Synthetic Ticket notes **must** be constrained to return structured data (e.g., via `PydanticOutputParser` or native JSON mode) before the data hits the Feature Store. Free text output without schema validation is blocked.
- Conversational output explaining a churn probability must accurately synthesize the backend ML prediction.

### 4.4 Tool Design Rules

Every agent tool must satisfy all of the following:
- [ ] Has a Pydantic `BaseModel` input schema.
- [ ] Has a clear Google-style docstring.
- [ ] Is **deterministic** — identical models + data exactly reproduce identical metrics.
- [ ] Guards against network/system failures gracefully by capturing exceptions and returning descriptive strings, thereby enabling Agentic Healing without crashing the whole graph.
- [ ] Is **stateless**.

### 4.5 Data Leakage Prevention

Agent tools must **never** expose or infer the `Churn` target variable when answering queries in production contexts (Inference Phase). Synthetic note generation during the *Feature Phase* is the only stage permitted to conditionally observe the `Churn` label to simulate historical fidelity. 

### 4.6 Provider Hot-Swapping via Config

Google API Keys and AWS keys are managed exclusively via `.env` files. No API keys or hardcoded provider details are allowed inside the application code.

---

## 5. MLOps Pipeline Rules

### 5.1 FTI Pipeline Independence

The FTI pipelines are inherently decoupled:

| Pipeline | Tools | Must NOT depend on |
| :--- | :--- | :--- |
| **Feature** | Great Expectations (GX), Pandas, LangGraph Enrichment | Live Inference REST calls |
| **Training** | Optuna, XGBoost, MLflow, Scikit-learn | FastApi serving modules |
| **Inference** | FastAPI, Gradio, Uvicorn | Raw data ELT cycles / Model Training |

**❌ PROHIBITED:** Deep coupling between the agent's real-time API routes and the massive batch processing logic of the feature engineering layer.

### 5.2 DVC — Versioned Artifacts

- Data versioning is legally mandated via DVC.
- Both the raw dataset and the **AI-Enriched NLP representation** must be versioned. 
- You must tie specific models in MLflow to specific DVC data SHAs.

### 5.3 MLflow & Evaluation Guardrails

- Training runs must log hyperparameters, model topologies, feature importance (SHAP), and standardized metrics.
- **Recall First:** Given the asymmetric cost of churn, models are primarily gated based on **Recall > 0.85**. Secondary thresholds apply to F1-Score. Avoid blindly maximizing Accuracy.
- Explicit `random_state` seeding is required for all Stochastic logic (SMOTE splitting, Tree initialization).

### 5.4 Data Contracts

`Great Expectations (GX)` is the definitive gateway.
- Enriched generated features (ticket notes length, sentiment classification labels) must pass strict suite validations before hitting the ML pipeline to guarantee the LLM hasn't hallucinated corrupt data frames.

---

## 6. API Service Rules

### 6.1 Prediction API (`src/api/predict_api.py`)

The FastAPI service must always:
- Preload the serialized `champion` ML model natively on the `lifespan` startup event, not per individual HTTP request.
- Provide a clear `/health` metric module for ECS deployment liveness.

### 6.2 Graceful Degradation

If the NLP vectorization endpoint or an upstream service times out while scoring a user's risk, the system must degrade safely. Rather than blowing up an unhandled 500, return a localized payload instructing the Gradio framework of degraded inference capacity (e.g. "Model available, Sentiment unavailable").

---

## 7. Testing Rules

### 7.1 Testing Pyramid

| Layer | Tool | Scope |
| :--- | :--- | :--- |
| **Unit** | `pytest` | Validating pure Python utility functions, Pydantic entities, and data validation modules. |
| **Evals** | `LLM-as-a-Judge` | Evaluating the quality and faithfulness of the AI-generated synthetic ticket notes. |
| **Integration** | `pytest` | API Endpoint parsing and FastAPI lifecycle validation. |

### 7.2 CI Enforcement

Pushing to master triggers GitHub Actions enforcing:
```
Lint/Format (Ruff) ──► Type Safety (Pyright) ──► Unit Tests & Cov ──► Docker ECS Hooks
```
Builds failing typing or linting are strictly denied.

---

## 8. Containerization Rules

### 8.1 Dockerfile Standards

- Utilize slim base images.
- Separate build dependencies from runtime utilizing Multi-stage builds.
- Images run under non-root permissions strictly.
- MLflow tracking server runs completely decoupled from the FastAPI interface containers.

---

## 9. Documentation Rules

### 9.1 The Master Source of Truth

Updates adhere strictly to the 5 pillars. Verify compliance within `reports/docs/` subdirectories **before executing or pushing logic updates**.

| Pillar | Location |
| :--- | :--- |
| **The Why (Decisions)** | `reports/docs/decisions/` |
| **The Map (Architecture)** | `reports/docs/architecture/` |
| **The Rules (Guardrails)** | `reports/docs/runbooks/rules_and_guardrails.md` |
| **The Evals (Quality)** | `reports/docs/evaluations/` |
| **The Workflows (Implementation)** | `reports/docs/workflows/` |

---

## 10. The "Do Not Do This" List

> Hard-Stop reference applicable to ALL contributors & orchestrating Agents.

| # | Prohibition | Impact if Violated |
| :--- | :--- | :--- |
| **R-01** | ❌ DO NOT allow LLMs to directly read `Churn=Yes/No` in the Inference (Live Prediction) phase. | Severe data leakage resulting in perfect prediction hallucinations, destroying business utility. |
| **R-02** | ❌ DO NOT pass purely unstructured dicts to any pipeline tools or API wrappers. | Replaces deterministic safety with runtime attribute explosions. |
| **R-03** | ❌ DO NOT optimize for model Accuracy over Recall. | Brawn becomes misaligned with the telecom business requirement of retaining missing distinct risk profiles. |
| **R-04** | ❌ DO NOT commit `.csv` or `.joblib` massive binary assets natively to the Git registry. | Punitively inflates repository footprint; DVC manages assets natively. |
| **R-05** | ❌ DO NOT install external components via native `pip` bypassing `uv.lock`. | Environment reproducible builds fracture immediately across distinct environments. |
| **R-06** | ❌ DO NOT hardcode Gemini prompt configurations directly inside executable processing code loops. | Restricts agentic observability and makes iterations overly precarious. |

---

## 11. Incident Response Quick Reference

| Symptom | Likely Cause | First Action |
| :--- | :--- | :--- |
| LLM-Generated notes missing structural limits | AI bypassed output schemas | Enforce PydanticOutputParser or strict JSON structured native callbacks on the Prompt context. |
| Model performance seems "too perfect" (R2 > .98, AUC = 1) | Data Leakage | Ensure `ticket_notes` generation during testing/inference unconditionally masks the `Churn` label. |
| Agent errors out complaining of Type Errors | `dict` was passed to a `Pydantic` enforced Tool | Evaluate the Python tracing to ensure the object is properly unpacked (`**dict`) or instantiated as the Model prior to dispatching to the tool. |
| API service returns unhandled 500s | Dependency ML Artifact not located natively on container | Verify `dvc pull` mapped the proper models synchronously into the image deployment hooks dynamically. |
