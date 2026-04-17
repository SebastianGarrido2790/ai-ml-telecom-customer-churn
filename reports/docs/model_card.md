# Model Card — Telecom Customer Churn Late Fusion Stacker

> Following the Model Cards for Model Reporting standard (Mitchell et al., arXiv:1810.03993)

| Field | Value |
|:---|:---|
| **Model Name** | `telco-churn-late-fusion` |
| **Version** | 1.0.0 |
| **Date** | 2026-04-17 |
| **Model Type** | Late Fusion Stacking Ensemble |
| **Registry** | MLflow Model Registry — `telco-churn-late-fusion` |
| **License** | MIT |
| **Contact** | Sebastian Garrido |

---

## 1. Model Details

### 1.1 Architecture

The model implements a **3-model Late Fusion Stacking** architecture that processes two independently trained branches and combines them via a meta-learner:

```
Raw Input
    ├─► Branch 1: Structured XGBoost
    │       Inputs: 19 tabular features (tenure, contract, charges, …)
    │       Feature prep: ColumnTransformer (OHE + StandardScaler)
    │       Optimiser: Optuna TPE, 30 trials, Recall objective
    │       Output: P(churn | structured)
    │
    └─► Branch 2: NLP XGBoost
            Inputs: 20 PCA dimensions from sentence-transformer embeddings
            Embedding model: all-MiniLM-L6-v2 (384-dim → PCA-20)
            Optimiser: Optuna TPE, 20 trials, Recall objective
            Output: P(churn | ticket_note)

Meta-Learner: Logistic Regression (C=0.1, max_iter=1000)
    Inputs: [P(churn | structured), P(churn | ticket_note)]  (Out-of-Fold)
    Output: final P(churn)  →  threshold 0.5 → binary label
```

### 1.2 Training Framework

| Component | Library / Version |
|:---|:---|
| Base learners | `xgboost` ≥ 2.0 |
| Meta-learner | `scikit-learn` LogisticRegression |
| Hyperparameter search | `optuna` TPE sampler |
| Embedding model | `sentence-transformers` all-MiniLM-L6-v2 |
| Dimensionality reduction | `scikit-learn` PCA |
| Class imbalance | `imbalanced-learn` SMOTE (per branch) |
| Experiment tracking | MLflow (3-run structure) |
| Pipeline versioning | DVC |

### 1.3 Leakage Prevention Decisions

- **Decision A2:** `primary_sentiment_tag` is excluded from all training branches via `DIAGNOSTIC_COLS` because it is a near-deterministic proxy of the target (99.3% accuracy in isolation). It is logged to MLflow as a diagnostic artifact only.
- **Decision B1:** SMOTE is applied independently per branch after feature extraction so synthetic neighbour computation operates in each branch's own geometric space.
- **Anti-Skew Mandate:** Custom transformers (`TextEmbedder`, `NumericCleaner`) live in `src/utils/feature_utils.py`, the single source of truth imported by both training and inference.

---

## 2. Intended Use

### 2.1 Primary Intended Use

- **Proactive churn intervention:** identify customers with high churn probability (≥ 0.5) so retention teams can offer targeted discounts or contract upgrades before cancellation.
- **Priority queue for customer success:** rank at-risk customers for outbound calls based on `churn_probability` score.
- **Explainability for retention specialists:** SHAP waterfall plots surface the top feature drivers per customer (e.g., `Month-to-month contract`, `Fiber optic internet`) to support human-in-the-loop decisions.

### 2.2 Primary Intended Users

- Customer retention analysts and success managers.
- Marketing automation tools consuming the `/v1/predict` REST endpoint.
- MLOps engineers monitoring model drift via the MLflow experiment registry.

### 2.3 Out-of-Scope Uses

> [!WARNING]
> The following uses are explicitly **not** supported and may produce misleading outputs.

- **Autonomous termination of customer accounts** — predictions are risk scores to guide human decisions, not automated triggers.
- **Deployment on datasets from outside the telecommunications domain** — the model was trained on the IBM Telco dataset; feature semantics (e.g., `InternetService`, `Contract`) are domain-specific.
- **Real-time streaming inference with < 100 ms SLA** — the NLP branch requires an embedding service call; circuit-breaker fallback yields structured-only predictions with no NLP signal.
- **Predicting churn for enterprise B2B accounts** — the training data represents individual residential consumers only.

---

## 3. Factors

### 3.1 Relevant Factors

| Factor | Range / Values | Notes |
|:---|:---|:---|
| **Tenure** | 0–72 months | Strong negative driver; new customers (0–12 months) disproportionately at risk |
| **Contract type** | Month-to-month, One year, Two year | Month-to-month is the dominant positive churn driver |
| **Internet service** | DSL, Fiber optic, No | Fiber optic has higher churn correlation |
| **Monthly charges** | USD 18–120 | Higher charges positively correlated with churn |
| **Ticket note sentiment** | Frustrated, Dissatisfied, Neutral, Satisfied | NLP branch; only observable from support tickets |

### 3.2 Evaluation Factors

The model optimises for **Recall** (sensitivity) because the business cost of a false negative (missing a churner) is higher than a false positive (unnecessary retention offer).

---

## 4. Metrics

### 4.1 Performance Summary

> These values reflect the best-run checkpoint logged in MLflow. Update after each production retraining cycle.

| Branch | Recall | F1-Score | AUC-ROC | AUC-PR |
|:---:|:---:|:---:|:---:|:---:|
| Structured XGBoost (Branch 1) | 0.7714 | 0.6457 | 0.8500 | — |
| NLP XGBoost (Branch 2) | 0.7107 | 0.4932 | 0.6810 | — |
| **Late Fusion Stacker** | **0.6536** | **0.6224** | **0.8476** | **—** |

> [!NOTE]
> Populate the table above after running `dvc repro` on the full enriched dataset. Metric artifacts are logged under the `telco-churn-late-fusion-v1` MLflow experiment.

### 4.2 Decision Threshold

Default classification threshold: **0.5** on `churn_probability`. Retention teams may lower this to 0.3–0.4 to increase recall at the cost of more false positives (more proactive outreach calls).

### 4.3 Custom Lift Metrics

MLflow logs two business-aligned lift metrics per run:

| Metric | Formula | Interpretation |
|:---|:---|:---|
| `recall_lift` | Recall(Fusion) − Recall(Structured) | Improvement in churn detection from adding NLP |
| `f1_lift` | F1(Fusion) − F1(Structured) | Net precision-recall trade-off from fusion |

---

## 5. Training Data

### 5.1 Source

- **Dataset:** IBM Watson Analytics — Telco Customer Churn (`WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- **Size:** 7,043 rows × 21 columns (raw); 7,043 × 23 columns after Phase 2 enrichment
- **Source URL:** `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` (DVC-tracked)
- **DVC hash:** see `dvc.lock` for exact content hash

### 5.2 Enrichment (Phase 2 — Agentic Data Enrichment)

Each customer record was enriched with two synthetic fields via the `pydantic-ai` 3-tier LLM pipeline:

- `ticket_note`: AI-generated customer support complaint note (≥ 10 characters, 95% non-null SLA).
- `primary_sentiment_tag`: AI-classified sentiment (`Frustrated`, `Dissatisfied`, `Neutral`, `Satisfied`, `Billing Inquiry`, `Technical Issue`).

> [!IMPORTANT]
> `primary_sentiment_tag` is a **synthetic, diagnostic-only field** excluded from model training. It must not be re-introduced as a training feature without a full leakage audit.

### 5.3 Class Distribution (Post-Enrichment)

| Class | Count | Proportion |
|:---:|:---:|:---:|
| No Churn | ~5,174 | ~73.5% |
| Churn | ~1,869 | ~26.5% |

**Imbalance ratio:** 2.77 : 1 — addressed with per-branch SMOTE (Decision B1).

### 5.4 Train / Validation / Test Split

| Split | Proportion | Stratified |
|:---:|:---:|:---:|
| Train | 70% | ✅ |
| Validation | 10% | ✅ |
| Test | 20% | ✅ |

Split performed before any feature transformation to prevent leakage.

---

## 6. Evaluation Data

- **Test set:** Sealed 20% stratified split of the enriched dataset (see §5.4).
- **Holdout integrity:** The test set is never used during Optuna hyperparameter search or OOF stacking. It is accessed once per run at final evaluation.
- **Validation set:** Used during Optuna trial pruning and early-stopping for XGBoost.

---

## 7. Ethical Considerations

### 7.1 Potential for Discriminatory Outcomes

- **Gender / Senior Citizen:** `gender` and `SeniorCitizen` are present in the raw dataset and passed as features. The model may learn spurious correlations between protected attributes and churn that reflect historical service quality gaps rather than actual churn propensity.
- **Recommended mitigation:** Run a **fairness audit** using `fairlearn` or `aif360` to measure Equalized Odds across `gender` and `SeniorCitizen` before production deployment.

### 7.2 Data Provenance

- The training data is a **publicly available IBM sample dataset** and does not contain real customer PII.
- In a production deployment, this must be replaced with actual customer records. All PII handling must comply with the organisation's data governance policy and applicable regulations (GDPR, CCPA).

### 7.3 Synthetic Data Warning

- `ticket_note` was **generated by a large language model** (Gemini/Ollama). It may contain hallucinated or stereotypical language patterns that correlate with protected attributes. The enrichment prompt was hardened (C1 Fix) to remove the churn label from context, but this risk cannot be fully eliminated.

### 7.4 Human-in-the-Loop Requirement

> [!IMPORTANT]
> All customer-facing retention actions triggered by model predictions **must** route through a human agent review layer. The model output is a decision-support tool, not an autonomous action system.

---

## 8. Caveats and Recommendations

### 8.1 Known Limitations

| Limitation | Severity | Mitigation |
|:---|:---:|:---|
| NLP branch requires embedding service | Medium | Circuit-breaker zero-vector fallback — structured-only predictions served when service is unavailable |
| `ticket_note` is synthetic in training | Medium | Replace with real support ticket notes if available; compare performance delta |
| No temporal validation (cohort leakage) | Medium | Apply temporal train/test split when retraining on time-stamped production data |
| Gender / Senior Citizen fairness not quantified | High | Run `fairlearn` Equalized Odds audit before any customer-facing deployment |
| Threshold fixed at 0.5 | Low | Calibrate threshold using PR-curve analysis against business cost matrix |

### 8.2 Retraining Triggers

Retrain when any of the following conditions are met:
- **Data drift detected** by the Great Expectations enriched-data suite failure rate rising above 2%.
- **Recall drops below 0.70** on a rolling 30-day production evaluation window.
- **New contract types or payment methods** are introduced that fall outside the training distribution.
- **Quarterly retraining schedule** as a baseline governance practice.

### 8.3 Monitoring

| Signal | Tool | Alert Threshold |
|:---|:---|:---|
| Schema drift (new columns) | Great Expectations | Any unexpected column |
| Embedding service availability | Prediction API circuit-breaker | `nlp_branch_available: false` in > 10% of responses |
| Churn probability mean shift | MLflow / Grafana | Δ > 0.05 from baseline distribution mean |
| API latency | OpenTelemetry → Jaeger | p99 > 500 ms |
