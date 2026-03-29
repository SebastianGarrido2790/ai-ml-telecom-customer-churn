"""
Unit tests for ML model training and evaluation logic.

This module validates the Late Fusion architecture, ensuring that target
encoding, SMOTE resampling, and hybrid branch training/evaluation behave
correctly under both deterministic and stochastic conditions.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.components.model_training.evaluator import LateFusionEvaluator, _compute_metrics
from src.components.model_training.trainer import (
    STRUCTURED_PREFIX,
    LateFusionTrainer,
    _apply_smote,
    _encode_target,
    _get_branch_columns,
)
from src.entity.config_entity import ModelTrainingConfig


@pytest.fixture
def temp_dir(tmp_path):
    """Provides a temporary directory for model artifact tests.

    Args:
        tmp_path: Pytest built-in fixture for temporary directory path.

    Returns:
        Path: Path object pointing to a temporary directory.
    """
    return tmp_path


# --- Helper Functions ---
def test_get_branch_columns():
    """Verifies that branch columns are correctly identified by prefix."""
    df = pd.DataFrame(columns=["num__1", "cat__2", "nlp__3", "other"])
    cols = _get_branch_columns(df, STRUCTURED_PREFIX)
    assert "num__1" in cols
    assert "cat__2" in cols
    assert "nlp__3" not in cols
    assert "other" not in cols


def test_encode_target():
    """Verifies that the target variable is correctly numerically encoded."""
    y = pd.Series(["No", "Yes", "No"])
    y_enc, le = _encode_target(y)
    assert np.array_equal(y_enc, [0, 1, 0])
    assert le.classes_.tolist() == ["No", "Yes"]


def test_apply_smote():
    """Verifies that SMOTE correctly balances an imbalanced target distribution."""
    X = np.random.rand(10, 5)
    y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # Imbalanced

    with patch("src.components.model_training.trainer.SMOTE") as mock_smote:
        mock_instance = mock_smote.return_value
        mock_instance.fit_resample.return_value = (
            np.random.rand(16, 5),
            np.array([0] * 8 + [1] * 8),
        )

        X_res, y_res = _apply_smote(X, y, 42)
        assert len(y_res) == 16
        mock_smote.assert_called_once_with(random_state=42)


# --- LateFusionTrainer ---
@patch("src.components.model_training.trainer.pd.read_csv")
@patch("src.components.model_training.trainer.joblib.dump")
@patch("src.components.model_training.trainer._tune_xgboost")
@patch("src.components.model_training.trainer.cross_val_predict")
def test_trainer_train(mock_cvp, mock_tune, mock_dump, mock_read, temp_dir):
    """Verifies the training orchestration for the LateFusion architecture."""
    config = ModelTrainingConfig(
        root_dir=temp_dir,
        train_data_path=temp_dir / "train.csv",
        test_data_path=temp_dir / "test.csv",
        val_data_path=temp_dir / "val.csv",
        structured_model_path=temp_dir / "s.pkl",
        nlp_model_path=temp_dir / "n.pkl",
        meta_model_path=temp_dir / "m.pkl",
        evaluation_report_path=temp_dir / "rep.json",
        target_column="Churn",
        random_state=42,
        cv_folds=3,
        structured_n_trials=1,
        nlp_n_trials=1,
        meta_C=1.0,
        meta_max_iter=100,
        mlflow_uri="http://test",
        experiment_name="test-exp",
        structured_preprocessor_path=temp_dir / "s_pre.pkl",
        nlp_preprocessor_path=temp_dir / "n_pre.pkl",
    )

    # Mock data with branch prefixes
    train_df = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "num__feat": [1.0, 2.0],
            "nlp__feat": [0.1, 0.2],
            "Churn": ["No", "Yes"],
        }
    )
    mock_read.return_value = train_df

    # Mock cross_val_predict output (probabilities)
    mock_cvp.return_value = np.array([[0.1, 0.9], [0.8, 0.2]])

    # Mock tuned model
    mock_model = MagicMock()
    mock_tune.return_value = mock_model

    trainer = LateFusionTrainer(config=config)
    s_m, n_m, m_m = trainer.train()

    assert mock_dump.call_count == 3
    assert s_m == mock_model
    assert n_m == mock_model


# --- LateFusionEvaluator ---
def test_compute_metrics():
    """Verifies that classification metrics (recall, AUC) are calculated correctly."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.8, 0.7])

    metrics = _compute_metrics(y_true, y_pred, y_prob)
    assert "recall" in metrics
    assert "roc_auc" in metrics
    assert metrics["recall"] == 1.0  # Both positive cases caught


@patch("src.components.model_training.evaluator.mlflow")
@patch("src.components.model_training.evaluator.joblib.load")
@patch("src.components.model_training.evaluator.pd.read_csv")
@patch("src.components.model_training.evaluator.plt.subplots")
def test_evaluator_evaluate(mock_plt, mock_read, mock_load, mock_mlflow, temp_dir):
    """Verifies the evaluation report generation and MLflow logging orchestration."""
    config = ModelTrainingConfig(
        root_dir=temp_dir,
        train_data_path=temp_dir / "train.csv",
        test_data_path=temp_dir / "test.csv",
        val_data_path=temp_dir / "val.csv",
        structured_model_path=temp_dir / "s.pkl",
        nlp_model_path=temp_dir / "n.pkl",
        meta_model_path=temp_dir / "m.pkl",
        evaluation_report_path=temp_dir / "rep.json",
        target_column="Churn",
        random_state=42,
        cv_folds=3,
        structured_n_trials=1,
        nlp_n_trials=1,
        meta_C=1.0,
        meta_max_iter=100,
        mlflow_uri="http://test",
        experiment_name="test-exp",
        structured_preprocessor_path=temp_dir / "s_pre.pkl",
        nlp_preprocessor_path=temp_dir / "n_pre.pkl",
    )

    # Mock subplots
    mock_fig = MagicMock()
    mock_plt.return_value = (mock_fig, MagicMock())

    # Mock mlflow runs
    mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "test-run-id"

    # Mock models
    m1 = MagicMock()
    m1.predict.return_value = np.array([0, 1])
    m1.predict_proba.return_value = np.array([[0.9, 0.1], [0.2, 0.8]])

    m2 = MagicMock()
    m2.predict.return_value = np.array([0, 1])
    m2.predict_proba.return_value = np.array([[0.8, 0.2], [0.1, 0.9]])

    mm = MagicMock()
    mm.predict.return_value = np.array([0, 1])
    mm.predict_proba.return_value = np.array([[0.9, 0.1], [0.3, 0.7]])

    mock_load.side_effect = [m1, m2, mm]

    # Mock data
    df = pd.DataFrame({"num__feat": [1.0, 2.0], "nlp__feat": [0.1, 0.2], "Churn": ["No", "Yes"]})
    mock_read.return_value = df

    evaluator = LateFusionEvaluator(config=config)
    report = evaluator.evaluate()

    assert "late_fusion_stacked" in report
    assert (temp_dir / "rep.json").exists()
    assert mock_mlflow.start_run.call_count == 3
