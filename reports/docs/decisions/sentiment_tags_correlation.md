## Phase 5: Implementation Plan

The dataset analysis reveals a critical design decision point before planning begins.

The sentiment-churn cross-tabulation tells a decisive story:

| Sentiment | Churn=No | Churn=Yes |
|---|---|---|
| Frustrated | 13 | **1,838** |
| Satisfied | 4,815 | 14 |
| Neutral | 242 | 2 |

The AI agent in Phase 2 generated sentiment tags that are **near-deterministically correlated with the target variable.** `Frustrated` predicts `Churn=Yes` with 99.3% accuracy on its own. This is a **data leakage risk** that must be addressed before designing the training pipeline. It does not invalidate Phase 2 — the ticket notes themselves are legitimate enrichment — but it does affect how `primary_sentiment_tag` is handled across the two branches.

---

### Decision 1 — `primary_sentiment_tag` Assignment

The sentiment tag was derived by the LLM using the `Churn` label as context (from the `CustomerInputContext` input contract in Phase 2). It is therefore a synthetic signal, not an observable one. In production, a real customer does not arrive with a pre-labeled sentiment tag — that tag would itself need to be inferred from the ticket note.

Two viable options:

**Option A1 — Sentiment Tag in Branch 1 (Structured), excluded from Branch 2 (NLP)**
The tag is treated as a structured categorical feature alongside contract type, tenure, etc. Branch 2 uses only the raw PCA embeddings from `ticket_note`. This is the most defensible design: Branch 1 tests the maximum value of structured + qualitative signals, Branch 2 tests pure NLP signal, and the fusion measures whether combining them adds anything beyond what sentiment alone achieves.

**Option A2 — Sentiment Tag excluded from both branches, used only as a diagnostic feature**
Removes the tag entirely from training. Branch 1 uses only the 19 raw structured features. This is the strictest leakage-free baseline. The tag's correlation with the target makes it act as a near-perfect predictor, which would artificially inflate Branch 1 performance and obscure whether the NLP embeddings add genuine value.

**Recommendation: Option A2.** The goal of the Late Fusion experiment is to isolate the contribution of `ticket_note` embeddings. Including a near-deterministic proxy of the target in Branch 1 would make the baseline artificially strong and undermine the ROI narrative. The sentiment tag is better used as a diagnostic artifact — logged to MLflow as a standalone feature importance reference, not as a training signal.

---

### Decision 2 — SMOTE Placement

Class imbalance is 2.77:1 (Churn=No: 73.5%, Churn=Yes: 26.5%). Two options:

**Option B1 — SMOTE applied independently per branch**
Each branch's training data is oversampled separately after branch-specific feature extraction. Clean separation, branch-specific imbalance handling.

**Option B2 — SMOTE applied once on the full training set before branch split**
Single oversampling step on all features together before splitting into structured/NLP columns. Simpler, but SMOTE operates in a mixed feature space (structured + embeddings) which is geometrically less meaningful — synthetic samples generated across 19 structured + 20 NLP dimensions simultaneously.

**Recommendation: Option B1.** Apply SMOTE independently within each branch's training pipeline after feature extraction. Each branch operates in its own geometric space, so synthetic neighbor computation is more meaningful. It also keeps each branch's pipeline fully self-contained, which is required for the OOF cross-validation stacking protocol.

---

### Full Phase 5 Plan
**Decision 1 — sentiment tag placement (A2), Decision 2 — SMOTE placement (B1)**

#### Step 1 — Phase 4 Modification

Modify `src/components/feature_engineering.py` to serialize two preprocessors instead of one:

- `get_structured_preprocessor()` → Numeric + Categorical pipelines only (drops `ticket_note`).
- `get_nlp_preprocessor()` → `TextEmbedder` + `PCA` pipeline only.
- Both fitted on Train set exclusively. Both saved as separate pkl files.
- `dvc.yaml` Stage 4 outputs updated: `structured_preprocessor.pkl` + `nlp_preprocessor.pkl` replace unified `preprocessor.pkl`.
- `config.yaml`, `params.yaml`, `config_entity.py`, and `configuration.py` updated with new paths and `ModelTrainingConfig`.

#### Step 2 — Model Training Component

New file: `src/components/model_training/trainer.py`

Responsibilities:
1. Load `structured_preprocessor.pkl` → transform structured columns → apply SMOTE (Branch 1 train set).
2. Load `nlp_preprocessor.pkl` → transform `ticket_note` column → apply SMOTE (Branch 2 train set).
3. Run Optuna for Branch 1 XGBoost (30 trials, Recall-weighted objective).
4. Run Optuna for Branch 2 XGBoost (20 trials, same objective).
5. Generate OOF predictions via `cross_val_predict(method='predict_proba')` for both base models on the (pre-SMOTE) train set.
6. Stack OOF arrays → train Logistic Regression meta-learner.
7. Retrain both base models on full SMOTE-augmented train set.
8. Evaluate stacked ensemble on held-out test set.

#### Step 3 — Evaluator Component

New file: `src/components/model_training/evaluator.py`

Responsibilities:
1. Log three MLflow runs: `structured_baseline`, `nlp_baseline`, `late_fusion_stacked`.
2. Per run: Recall, F1, ROC-AUC, confusion matrix (as artifact), feature importance chart.
3. Fusion run additionally logs `recall_lift` and `f1_lift` over structured baseline.
4. Serialize `structured_model.pkl`, `nlp_model.pkl`, `meta_model.pkl` to `artifacts/model_training/`.
5. Write `evaluation_report.json` for DVC tracking and CI/CD gate.
6. Register champion model in MLflow Model Registry with tag `production`.

#### Step 4 — Pipeline Stage

New file: `src/pipeline/stage_05_model_training.py` — conductor only, no business logic.

#### Step 5 — DVC Stage

New `train_model` stage in `dvc.yaml`:
- **deps:** both preprocessors, train/val/test CSVs, trainer and evaluator source files, config/params.
- **outputs:** `structured_model.pkl`, `nlp_model.pkl`, `meta_model.pkl`, `evaluation_report.json`.

#### Step 6 — Unit Tests

New file: `tests/unit/test_model_training.py`:
- Test OOF array shape correctness.
- Test SMOTE only modifies train set, not val/test.
- Test meta-learner input shape matches `[P1, P2]` stacked format.
- Test `evaluation_report.json` schema via Pydantic.

---

### Files to be created or modified

| File | Action |
|---|---|
| `src/components/feature_engineering.py` | Modify — split into two `get_*_preprocessor()` methods |
| `src/entity/config_entity.py` | Modify — add `ModelTrainingConfig` |
| `src/config/configuration.py` | Modify — add `get_model_training_config()` |
| `config/config.yaml` | Modify — add `model_training` paths section |
| `config/params.yaml` | Modify — add `model_training` hyperparameters section |
| `dvc.yaml` | Modify — update Stage 4 outputs; add Stage 5 |
| `src/components/model_training/__init__.py` | Create |
| `src/components/model_training/trainer.py` | Create |
| `src/components/model_training/evaluator.py` | Create |
| `src/pipeline/stage_05_model_training.py` | Create |
| `tests/unit/test_model_training.py` | Create |
