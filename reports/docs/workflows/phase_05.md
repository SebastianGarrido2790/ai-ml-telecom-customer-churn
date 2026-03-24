## Phase 5: Implementation Plan

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
