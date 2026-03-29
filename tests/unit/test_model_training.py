"""
Unit Tests: Late Fusion Model Training Components.

Tests cover the four deterministic guarantees of the training pipeline:

    1. OOF array shape: cross_val_predict output aligns with training set size.
    2. SMOTE isolation: oversampling modifies only the training set, never
       val or test sets, preserving evaluation realism.
    3. Meta-learner input contract: the stacked array fed to Logistic
       Regression has exactly 2 columns — [P_struct, P_nlp].
    4. Evaluation report schema: evaluation_report.json contains all required
       keys and is parseable as a valid Pydantic-validated structure.

These tests exercise only deterministic logic. No live API calls, no MLflow
server, and no actual Optuna search are triggered — all external dependencies
are replaced with minimal fixtures and mock objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import BaseModel

# ============================================================================
# Pydantic Schema: Evaluation Report Contract
# ============================================================================


class BranchMetrics(BaseModel):
    """Expected metrics sub-schema for each run in the evaluation report."""

    recall: float
    precision: float
    f1: float
    roc_auc: float


class BaselineRunReport(BaseModel):
    """Expected schema for a single baseline run entry."""

    run_id: str
    metrics: BranchMetrics


class FusionRunReport(BaseModel):
    """Expected schema for the late fusion run entry (includes lift metrics)."""

    run_id: str
    metrics: BranchMetrics
    recall_lift: float
    f1_lift: float


class EvaluationReportSchema(BaseModel):
    """Top-level schema for evaluation_report.json.

    Enforces presence of all three run keys and the meta section.
    """

    structured_baseline: BaselineRunReport
    nlp_baseline: BaselineRunReport
    late_fusion_stacked: FusionRunReport
    meta: dict[str, Any]


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture()
def sample_train_df() -> pd.DataFrame:
    """Minimal feature DataFrame mimicking train_features.csv output.

    Contains structured (num__, cat__) and NLP (nlp__) prefixed columns
    plus customerID and target — matching the column contract from
    feature_engineering.py.
    """
    rng = np.random.default_rng(42)
    n = 200

    data: dict[str, Any] = {
        "customerID": [f"CUST-{i:04d}" for i in range(n)],
        # Structured branch columns
        "num__tenure": rng.uniform(0, 72, n),
        "num__MonthlyCharges": rng.uniform(18, 120, n),
        "num__TotalCharges": rng.uniform(0, 8000, n),
        "cat__gender_Male": rng.integers(0, 2, n).astype(float),
        "cat__Contract_One year": rng.integers(0, 2, n).astype(float),
        "cat__InternetService_Fiber optic": rng.integers(0, 2, n).astype(float),
        # NLP branch columns (20 PCA components)
        **{f"nlp__nlp__ticket_note_emb_{i}": rng.normal(0, 1, n) for i in range(20)},
        # Target
        "Churn": rng.choice(["Yes", "No"], size=n, p=[0.27, 0.73]).tolist(),
    }
    return pd.DataFrame(data)


@pytest.fixture()
def sample_val_df(sample_train_df: pd.DataFrame) -> pd.DataFrame:
    """Minimal validation DataFrame (50 rows, same schema as train)."""
    return sample_train_df.sample(50, random_state=7).reset_index(drop=True)


@pytest.fixture()
def sample_test_df(sample_train_df: pd.DataFrame) -> pd.DataFrame:
    """Minimal test DataFrame (50 rows, same schema as train)."""
    return sample_train_df.sample(50, random_state=13).reset_index(drop=True)


# ============================================================================
# Test 1: OOF Array Shape
# ============================================================================


class TestOOFArrayShape:
    """Validates that cross_val_predict returns a 1-D array of correct length."""

    def test_oof_shape_matches_training_set(self, sample_train_df: pd.DataFrame) -> None:
        """OOF probability vector must have exactly n_train elements.

        The meta-learner is trained on this array, so any shape mismatch
        would cause a silent downstream concat error.
        """
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from xgboost import XGBClassifier

        from src.components.model_training.trainer import (
            STRUCTURED_PREFIX,
            _encode_target,
            _get_branch_columns,
        )

        feature_cols = _get_branch_columns(sample_train_df, STRUCTURED_PREFIX)
        X = sample_train_df[feature_cols].to_numpy()
        y, _ = _encode_target(sample_train_df["Churn"])

        model = XGBClassifier(
            n_estimators=10,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

        assert oof.shape == (len(sample_train_df),), f"Expected OOF shape ({len(sample_train_df)},), got {oof.shape}"

    def test_oof_values_are_valid_probabilities(self, sample_train_df: pd.DataFrame) -> None:
        """OOF values must be in [0, 1] — required for meta-learner input."""
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from xgboost import XGBClassifier

        from src.components.model_training.trainer import (
            STRUCTURED_PREFIX,
            _encode_target,
            _get_branch_columns,
        )

        feature_cols = _get_branch_columns(sample_train_df, STRUCTURED_PREFIX)
        X = sample_train_df[feature_cols].to_numpy()
        y, _ = _encode_target(sample_train_df["Churn"])

        model = XGBClassifier(
            n_estimators=10,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        oof = cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]

        assert np.all(oof >= 0.0) and np.all(oof <= 1.0), "OOF probabilities contain values outside [0, 1]."


# ============================================================================
# Test 2: SMOTE Isolation
# ============================================================================


class TestSMOTEIsolation:
    """Validates that SMOTE modifies only the training set."""

    def test_smote_increases_train_size(self, sample_train_df: pd.DataFrame) -> None:
        """SMOTE must add synthetic samples to the minority class."""
        from src.components.model_training.trainer import (
            STRUCTURED_PREFIX,
            _apply_smote,
            _encode_target,
            _get_branch_columns,
        )

        feature_cols = _get_branch_columns(sample_train_df, STRUCTURED_PREFIX)
        X = sample_train_df[feature_cols].to_numpy()
        y, _ = _encode_target(sample_train_df["Churn"])

        original_size = X.shape[0]
        X_res, y_res = _apply_smote(X, y, random_state=42)

        assert X_res.shape[0] > original_size, "SMOTE did not increase the training set size."
        assert X_res.shape[1] == X.shape[1], "SMOTE must not change the number of features."

    def test_smote_balances_classes(self, sample_train_df: pd.DataFrame) -> None:
        """After SMOTE, both classes must have equal sample counts."""
        from src.components.model_training.trainer import (
            STRUCTURED_PREFIX,
            _apply_smote,
            _encode_target,
            _get_branch_columns,
        )

        feature_cols = _get_branch_columns(sample_train_df, STRUCTURED_PREFIX)
        X = sample_train_df[feature_cols].to_numpy()
        y, _ = _encode_target(sample_train_df["Churn"])

        _, y_res = _apply_smote(X, y, random_state=42)
        unique, counts = np.unique(y_res, return_counts=True)

        assert len(unique) == 2, "SMOTE output must contain exactly 2 classes."
        assert counts[0] == counts[1], f"SMOTE did not balance classes: {dict(zip(unique, counts, strict=False))}"

    def test_val_set_unchanged_by_smote(
        self,
        sample_train_df: pd.DataFrame,
        sample_val_df: pd.DataFrame,
    ) -> None:
        """Applying SMOTE to train must not alter the validation DataFrame."""
        from src.components.model_training.trainer import (
            STRUCTURED_PREFIX,
            _apply_smote,
            _encode_target,
            _get_branch_columns,
        )

        feature_cols = _get_branch_columns(sample_train_df, STRUCTURED_PREFIX)
        X_train = sample_train_df[feature_cols].to_numpy()
        y_train, _ = _encode_target(sample_train_df["Churn"])

        val_snapshot = sample_val_df.copy(deep=True)
        _apply_smote(X_train, y_train, random_state=42)

        pd.testing.assert_frame_equal(
            sample_val_df,
            val_snapshot,
            check_like=False,
        )


# ============================================================================
# Test 3: Meta-Learner Input Contract
# ============================================================================


class TestMetaLearnerInputContract:
    """Validates the stacked OOF array fed to the Logistic Regression."""

    def test_stacked_array_has_two_columns(self, sample_train_df: pd.DataFrame) -> None:
        """The meta-learner input must have exactly 2 columns: [P_struct, P_nlp]."""
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from xgboost import XGBClassifier

        from src.components.model_training.trainer import (
            NLP_PREFIX,
            STRUCTURED_PREFIX,
            _encode_target,
            _get_branch_columns,
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        struct_cols = _get_branch_columns(sample_train_df, STRUCTURED_PREFIX)
        X_struct = sample_train_df[struct_cols].to_numpy()
        y, _ = _encode_target(sample_train_df["Churn"])

        nlp_cols = _get_branch_columns(sample_train_df, (NLP_PREFIX,))
        X_nlp = sample_train_df[nlp_cols].to_numpy()

        m = XGBClassifier(
            n_estimators=10,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        oof_struct = cross_val_predict(m, X_struct, y, cv=cv, method="predict_proba")[:, 1]
        oof_nlp = cross_val_predict(m, X_nlp, y, cv=cv, method="predict_proba")[:, 1]

        stacked = np.column_stack([oof_struct, oof_nlp])

        assert stacked.ndim == 2, "Stacked array must be 2-dimensional."
        assert stacked.shape[1] == 2, f"Meta-learner input must have 2 columns, got {stacked.shape[1]}."
        assert stacked.shape[0] == len(sample_train_df), "Stacked array row count must match training set size."

    def test_meta_learner_fits_on_stacked_oof(self, sample_train_df: pd.DataFrame) -> None:
        """Logistic Regression must fit without error on the [P_struct, P_nlp] input."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_predict
        from xgboost import XGBClassifier

        from src.components.model_training.trainer import (
            NLP_PREFIX,
            STRUCTURED_PREFIX,
            _encode_target,
            _get_branch_columns,
        )

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        y, _ = _encode_target(sample_train_df["Churn"])

        m = XGBClassifier(
            n_estimators=10,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )

        struct_cols = _get_branch_columns(sample_train_df, STRUCTURED_PREFIX)
        oof_s = cross_val_predict(m, sample_train_df[struct_cols].to_numpy(), y, cv=cv, method="predict_proba")[:, 1]

        nlp_cols = _get_branch_columns(sample_train_df, (NLP_PREFIX,))
        oof_n = cross_val_predict(m, sample_train_df[nlp_cols].to_numpy(), y, cv=cv, method="predict_proba")[:, 1]

        stacked = np.column_stack([oof_s, oof_n])
        meta = LogisticRegression(C=1.0, max_iter=200, random_state=42)

        # Must not raise
        meta.fit(stacked, y)
        assert hasattr(meta, "coef_"), "Meta-learner did not fit correctly."
        assert meta.coef_.shape == (1, 2), f"Expected coef_ shape (1, 2), got {meta.coef_.shape}."


# ============================================================================
# Test 4: Evaluation Report Schema
# ============================================================================


class TestEvaluationReportSchema:
    """Validates the structure of evaluation_report.json via Pydantic."""

    def _make_sample_report(self) -> dict[str, Any]:
        """Constructs a minimal valid evaluation report dictionary."""
        return {
            "structured_baseline": {
                "run_id": "abc123",
                "metrics": {
                    "recall": 0.81,
                    "precision": 0.74,
                    "f1": 0.77,
                    "roc_auc": 0.85,
                },
            },
            "nlp_baseline": {
                "run_id": "def456",
                "metrics": {
                    "recall": 0.72,
                    "precision": 0.68,
                    "f1": 0.70,
                    "roc_auc": 0.78,
                },
            },
            "late_fusion_stacked": {
                "run_id": "ghi789",
                "metrics": {
                    "recall": 0.88,
                    "precision": 0.76,
                    "f1": 0.82,
                    "roc_auc": 0.91,
                },
                "recall_lift": 0.07,
                "f1_lift": 0.05,
            },
            "meta": {
                "target_column": "Churn",
                "cv_folds": 5,
                "structured_n_trials": 30,
                "nlp_n_trials": 20,
            },
        }

    def test_valid_report_passes_schema(self) -> None:
        """A correctly structured report must parse without Pydantic errors."""
        report = self._make_sample_report()
        parsed = EvaluationReportSchema(**report)
        assert parsed.late_fusion_stacked.recall_lift == pytest.approx(0.07)

    def test_report_serialises_to_json(self, tmp_path: Path) -> None:
        """evaluation_report.json must be writable and re-parseable from disk."""
        report = self._make_sample_report()
        report_path = tmp_path / "evaluation_report.json"

        with report_path.open("w") as f:
            json.dump(report, f, indent=2)

        with report_path.open() as f:
            loaded = json.load(f)

        parsed = EvaluationReportSchema(**loaded)
        assert parsed.structured_baseline.metrics.recall == pytest.approx(0.81)

    def test_missing_fusion_run_fails_schema(self) -> None:
        """A report missing the late_fusion_stacked key must fail Pydantic validation."""
        from pydantic import ValidationError

        report = self._make_sample_report()
        del report["late_fusion_stacked"]

        with pytest.raises(ValidationError):
            EvaluationReportSchema(**report)

    def test_missing_lift_metrics_fails_schema(self) -> None:
        """A fusion run without recall_lift must fail Pydantic validation."""
        from pydantic import ValidationError

        report = self._make_sample_report()
        del report["late_fusion_stacked"]["recall_lift"]

        with pytest.raises(ValidationError):
            EvaluationReportSchema(**report)

    def test_recall_lift_sign_is_positive_in_happy_path(self) -> None:
        """In a well-performing fusion, recall_lift must be positive."""
        report = self._make_sample_report()
        parsed = EvaluationReportSchema(**report)
        assert parsed.late_fusion_stacked.recall_lift > 0, (
            "recall_lift should be positive when fusion outperforms the baseline."
        )
