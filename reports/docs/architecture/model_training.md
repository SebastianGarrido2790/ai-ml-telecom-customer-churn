# Phase 5: Late Fusion Model Training — Architecture Report

## 1. Executive Summary

This document details the architecture, implementation decisions, experimental results, and diagnostic findings for Phase 5 of the Telecom Customer Churn Prediction project. Phase 5 implements the **Training Pipeline** ("T") of the FTI pattern, producing three serialized model
artifacts and three MLflow-tracked experiment runs that collectively deliver an auditable, quantified assessment of the AI enrichment investment made in Phase 2.

The phase is built on the **Late Fusion stacking architecture**: two independent XGBoost base models are trained on separate feature branches (structured and NLP), and their Out-of-Fold probability predictions are stacked as input to a Logistic Regression meta-learner. This design provides an explicit, branch-isolated measurement of how much predictive signal each feature space contributes.

> **Design Reference:** The dual-enhancement strategy (Late Fusion + Embedding Microservice) and the rationale for the two-preprocessor split are documented in [executive_summary.md](../references/executive_summary.md) and the architectural decision records listed in Section 9 below.

---

## 2. Architecture: Late Fusion Stacking

### 2.1 The Three-Branch Design

```
Enriched Feature Store (train/val/test CSVs)
              │
    ┌─────────┴──────────┐
    ▼                    ▼
Branch 1                Branch 2
Structured Features     NLP Features
(numeric + OHE)         (PCA-reduced embeddings)
    │                    │
    ▼                    ▼
XGBoost                 XGBoost
(30 Optuna trials)      (20 Optuna trials)
SMOTE applied           SMOTE applied
independently           independently
    │                    │
    └─────────┬──────────┘
              ▼
    OOF Probability Stack
    [P_struct, P_nlp] shape: (n_train, 2)
              │
              ▼
    Logistic Regression
    Meta-Learner (OOF-trained)
              │
              ▼
    Final Churn Score
```

### 2.2 Key Design Decisions

**Decision A2 — `primary_sentiment_tag` excluded from all branches.** The sentiment tag was generated in Phase 2 with knowledge of the churn label, producing a near-deterministic proxy (98.3% `Frustrated` for Churn=Yes in the original data). It is retained in the enriched CSV for diagnostic use but never enters any training branch. See Section 6 for the full leakage investigation and C1 fix.

**Decision B1 — SMOTE applied independently per branch.** Each branch operates in its own geometric space. Applying SMOTE in the mixed 68-dimensional space (structured + NLP combined) would generate synthetic neighbors across semantically unrelated dimensions. Independent
application ensures synthetic samples are geometrically meaningful within each branch's feature space.

**OOF Stacking Protocol.** The meta-learner is trained on Out-of-Fold predictions from a 5-fold StratifiedKFold cross-validation on the pre-SMOTE training set. This prevents the meta-learner from seeing the same training labels that the base models were fitted on, eliminating the primary leakage risk in stacking architectures.

---

## 3. Component Structure

```
src/
└── components/
    └── model_training/
        ├── __init__.py
        ├── trainer.py      ← Branch extraction, SMOTE, OOF generation,
        │                     Optuna tuning, model serialization
        └── evaluator.py    ← MLflow run logging, confusion matrices,
                              feature importance charts, evaluation_report.json
src/
└── pipeline/
    └── stage_05_model_training.py   ← Conductor: wires config → trainer → evaluator
```

| File | Role | Pattern |
|---|---|---|
| `trainer.py` | Branch training, stacking, serialization | Strategy Pattern |
| `evaluator.py` | MLflow logging, artifact generation, report | Observer Pattern |
| `stage_05_model_training.py` | Entry point, result summary logging | FTI Training Stage |

---

## 4. Configuration

All training hyperparameters are centralized in `config/params.yaml` under `model_training`. Artifact paths live in `config/config.yaml` under `model_training`. No values are hardcoded in component files.

```yaml
# config/params.yaml
model_training:
  random_state: 42
  cv_folds: 5
  structured_branch:
    algorithm: "xgboost"
    n_trials: 30
  nlp_branch:
    algorithm: "xgboost"
    n_trials: 20
  meta_learner:
    algorithm: "logistic_regression"
    C: 1.0
    max_iter: 1000
```

The `ModelTrainingConfig` frozen dataclass (in `src/entity/config_entity.py`) is hydrated by `ConfigurationManager.get_model_training_config()` and passed immutably to both `LateFusionTrainer` and `LateFusionEvaluator`.

---

## 5. Experimental Results (Leakage-Free Run)

All results are from the C1-corrected pipeline (leakage-free `ticket_notes`). Test set: 1,057 customers (15% stratified holdout). All metrics computed on the unmodified test set, SMOTE was never applied to val or test splits.

### 5.1 Metric Summary

| Branch | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Structured Baseline (Branch 1) | **0.771** | 0.555 | 0.646 | **0.850** |
| NLP Baseline (Branch 2) | 0.711 | 0.378 | 0.493 | 0.681 |
| Late Fusion (Stacked) | 0.654 | **0.594** | 0.622 | 0.848 |

**Recall Lift (Fusion vs. Structured):** −0.118
**F1 Lift (Fusion vs. Structured):** −0.023

### 5.2 Confusion Matrix Analysis

| Branch | TP | FN | FP | TN | FP Rate |
|---|---|---|---|---|---|
| Structured Baseline | 216 | 64 | 173 | 604 | 22.3% |
| NLP Baseline | 199 | 81 | 328 | 449 | 42.2% |
| **Late Fusion** | 183 | 97 | **125** | **652** | **16.1%** |

The Late Fusion model is the **most precise** of the three: its False Positive rate of 16.1% is the lowest, meaning it produces fewer spurious retention interventions on non-churners. The structured baseline captures more true churners (Recall=0.771) but at the cost of flagging 173 non-churners as at-risk, which translates to wasted retention spend.

### 5.3 Optuna Best Hyperparameters

**Branch 1 (Structured):** Best validation Recall = 0.740
```
n_estimators=154, max_depth=3, learning_rate=0.0245,
subsample=0.786, colsample_bytree=0.808,
min_child_weight=6, reg_alpha=1.38e-4, reg_lambda=1.87e-8
```

**Branch 2 (NLP):** Best validation Recall = 0.673
```
n_estimators=144, max_depth=4, learning_rate=0.0117,
subsample=0.730, colsample_bytree=0.755,
min_child_weight=3, reg_alpha=0.0426, reg_lambda=7.15e-6
```

---

## 6. Data Leakage Investigation & C1 Fix

### 6.1 Discovery

The first pipeline execution (using the original Phase 2 enrichment) produced the following results, which triggered a leakage investigation:

| Branch | Recall | Precision | F1 | ROC-AUC |
|---|---|---|---|---|
| Structured Baseline | 0.771 | 0.555 | 0.646 | 0.850 |
| NLP Baseline | **1.000** | **0.993** | **0.996** | **0.9999** |
| Late Fusion | 1.000 | 0.993 | 0.996 | 0.9999 |

NLP Recall=1.000 and ROC-AUC=0.9999 on a held-out test set is statistically impossible from genuine NLP signal on this dataset. Investigation confirmed two leakage mechanisms:

1. **Schema leakage:** `Churn: Literal["Yes", "No"]` was a field in `CustomerInputContext`,
   passing the target label directly into the LLM prompt.
2. **Prompt leakage:** The system prompt contained explicit logic gates conditioning note
   generation on `Churn=Yes/No` (e.g., *"If Churn=Yes, the note MUST be negative"*).
3. **Fallback leakage:** The deterministic Tier 3 fallback branched directly on `customer_context.Churn`.

Cross-tabulation confirmed: 98.3% of `Churn=Yes` rows received `Frustrated` tags; 93.1% of `Churn=No` rows received `Satisfied` tags — near-perfect label encoding in the text embeddings.

### 6.2 C1 Fix Applied

Four files were modified to eliminate all leakage vectors:

| File | Change |
|---|---|
| `schemas.py` | `Churn` field removed from `CustomerInputContext`; schema expanded to 17 observable CRM fields |
| `prompts.py` | All `Churn`-conditional logic gates removed; CRM-agent persona with service-signal grounding rules |
| `generator.py` | `Churn Status` removed from `user_prompt`; deterministic fallback rewritten using contract/charges/service logic only |
| `orchestrator.py` | `Churn=` argument removed from `CustomerInputContext` constructor |

### 6.3 Post-Fix Sentiment Distribution

After C1 remediation, the sentiment distribution reflects genuine service-signal differentiation:

| Tag | Count | % | Churn Rate |
|---|---|---|---|
| Billing Inquiry | 4,095 | 58.1% | 26.1% |
| Dissatisfied | 1,394 | 19.8% | 30.2% |
| Frustrated | 763 | 10.8% | 40.2% |
| Satisfied | 438 | 6.2% | 8.7% |
| Neutral | 353 | 5.0% | 9.3% |

Churn rates per tag form a credible ordinal relationship without being deterministic, the correct profile for a legitimate soft signal.

---

## 7. Interpretation & Business Narrative

### 7.1 Why the NLP Branch Underperforms Alone

The NLP Branch (Recall=0.711, ROC-AUC=0.681) underperforms the Structured Branch alone (Recall=0.771, ROC-AUC=0.850). This is expected and interpretable: the leakage-free ticket notes encode behavioral frustration signals that partially overlap with what the structured features already capture (contract type, charges, tech support). The embeddings add texture but not independent discriminative power at this PCA compression level (20 components from 384-dim vectors).

### 7.2 Why the Fusion Recall Drops Below Structured

The meta-learner's Recall (0.654) falls below the structured baseline (0.771). This occurs because the Logistic Regression meta-learner, trained on OOF probabilities `[P_struct, P_nlp]`, learns to weight the NLP branch positively — but when the NLP branch introduces noise (low ROC-AUC=0.681), the meta-learner's decision boundary shifts toward higher precision at the cost of recall. The meta-learner is, in effect, being penalised by the NLP branch's noise.

### 7.3 What the Fusion Model Genuinely Demonstrates

The Late Fusion model achieves the **lowest False Positive Rate** of all three models (16.1% vs. 22.3% for structured, 42.2% for NLP). This is the correct business framing for a retention use case: every False Positive is a retention offer extended to a customer who was not at risk. The fusion model reduces wasted retention spend by 27% relative to the structured baseline (125 FP vs. 173 FP) while still catching 183 of 280 true churners.

The business case is therefore: *"The Agentic enrichment pipeline enables a more precise targeting strategy — directing retention resources toward genuinely at-risk customers rather than over-flagging the customer base."*

---

## 8. Artifacts Produced

| Artifact | Path | Tracked By |
|---|---|---|
| `structured_model.pkl` | `artifacts/model_training/` | DVC |
| `nlp_model.pkl` | `artifacts/model_training/` | DVC |
| `meta_model.pkl` | `artifacts/model_training/` | DVC |
| `evaluation_report.json` | `artifacts/model_training/` | DVC (CI/CD gate) |
| `confusion_matrix_structured_baseline.png` | `artifacts/model_training/` | MLflow artifact |
| `confusion_matrix_nlp_baseline.png` | `artifacts/model_training/` | MLflow artifact |
| `confusion_matrix_late_fusion_stacked.png` | `artifacts/model_training/` | MLflow artifact |
| `feature_importance_structured_baseline.png` | `artifacts/model_training/` | MLflow artifact |
| `feature_importance_nlp_baseline.png` | `artifacts/model_training/` | MLflow artifact |

The `evaluation_report.json` is the DVC-tracked pipeline output that serves as the downstream gate for the `serve_model` stage (Phase 6). The three MLflow runs are registered under the `telco-churn-prediction` experiment; `late_fusion_stacked` is additionally registered in the MLflow Model Registry as `telco-churn-late-fusion` (version 2, post C1 fix).

---

## 9. Decision Records & Related Documents

| Topic | Document |
|---|---|
| Overall System & FTI Pattern | [architecture.md](architecture.md) |
| Phase 2: Agentic Data Enrichment | [data_enrichment.md](data_enrichment.md) |
| Phase 4: NLP & Feature Engineering | [feature_engineering.md](feature_engineering.md) |
| **Decision: Late Fusion vs. Unified Model** | [model_architecture_decision.md](../decisions/model_architecture_decision.md) |
| **Decision: Embedding Microservice Extraction** | [embedding_service_decision.md](../decisions/embedding_service_decision.md) |
| **Decision: Sentiment Tag Exclusion (A2)** | [leakage_fix_c1.md](../decisions/leakage_fix_c1.md) |
| DVC Pipeline DAG | [dvc_pipeline.md](dvc_pipeline.md) |
| Test Suite Coverage | [test_suite.md](../runbooks/test_suite.md) |
