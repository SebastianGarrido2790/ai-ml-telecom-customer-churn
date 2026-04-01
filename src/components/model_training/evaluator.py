"""
Late Fusion Model Evaluator.

This module computes performance metrics for all three model branches,
logs them to MLflow, and produces a structured evaluation report artifact.

MLflow Run Structure:
    Three runs are created within the same experiment:
        - structured_baseline: Branch 1 metrics (structured features only).
        - nlp_baseline:        Branch 2 metrics (NLP embeddings only).
        - late_fusion_stacked: Fusion metrics + lift over structured baseline.

    The fusion run additionally logs recall_lift and f1_lift as custom MLflow
    metrics, providing a direct, auditable ROI measure for the Phase 2 AI
    enrichment investment.

Artifacts logged per run:
    - Confusion matrix plot (PNG).
    - Feature importance chart (PNG, XGBoost branches only).
    - Classification report (JSON).
    - evaluation_report.json: Consolidated report across all three runs,
      DVC-tracked as a pipeline output and used as the CI/CD quality gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow import sklearn, xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.components.model_training.trainer import (
    NLP_PREFIX,
    STRUCTURED_PREFIX,
    _encode_target,
    _get_branch_columns,
)
from src.entity.config_entity import ModelTrainingConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> dict[str, float]:
    """Computes the standard classification metric suite.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted positive-class probabilities.

    Returns:
        Dictionary of metric name → float value.
    """
    return {
        "recall": float(recall_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def _log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    run_name: str,
    artifact_dir: Path,
) -> None:
    """Saves a confusion matrix PNG and logs it as an MLflow artifact.

    Args:
        y_true: Ground-truth binary labels.
        y_pred: Predicted binary labels.
        run_name: Used as the filename prefix.
        artifact_dir: Local directory to save the PNG before logging.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No Churn", "Churn"],
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix — {run_name}")
    fig.tight_layout()

    output_path = artifact_dir / f"confusion_matrix_{run_name}.png"
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(output_path))
    logger.info(f"Confusion matrix saved: {output_path}")


def _log_feature_importance(
    model: XGBClassifier,
    run_name: str,
    artifact_dir: Path,
    top_n: int = 20,
) -> None:
    """Saves an XGBoost feature importance chart and logs it as an MLflow artifact.

    Args:
        model: Fitted XGBClassifier instance.
        run_name: Used as the filename prefix.
        artifact_dir: Local directory to save the PNG before logging.
        top_n: Number of top features to display in the chart.
    """
    importance = model.feature_importances_
    n_features = min(top_n, len(importance))
    top_idx = np.argsort(importance)[-n_features:]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(n_features), importance[top_idx])
    ax.set_yticks(range(n_features))
    ax.set_yticklabels([f"feature_{i}" for i in top_idx], fontsize=8)
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {n_features} Feature Importances — {run_name}")
    fig.tight_layout()

    output_path = artifact_dir / f"feature_importance_{run_name}.png"
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    mlflow.log_artifact(str(output_path))
    logger.info(f"Feature importance chart saved: {output_path}")


class LateFusionEvaluator:
    """Evaluates all three model branches and produces consolidated MLflow runs.

    Attributes:
        config: Immutable training configuration entity.
    """

    def __init__(self, config: ModelTrainingConfig) -> None:
        """Initializes the LateFusionEvaluator.

        Args:
            config: ModelTrainingConfig with all paths and hyperparameters.
        """
        self.config = config
        self._artifact_dir = Path(config.root_dir)

    def _load_models(
        self,
    ) -> tuple[XGBClassifier, XGBClassifier, LogisticRegression]:
        """Loads all three serialized model artifacts from disk.

        Returns:
            Tuple of (structured_model, nlp_model, meta_model).
        """
        structured_model: XGBClassifier = joblib.load(self.config.structured_model_path)
        nlp_model: XGBClassifier = joblib.load(self.config.nlp_model_path)
        meta_model: LogisticRegression = joblib.load(self.config.meta_model_path)
        return structured_model, nlp_model, meta_model

    def _get_val_test_arrays(
        self,
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Loads and extracts branch-specific arrays for val and test splits.

        Returns:
            Tuple of six arrays:
                X_val_struct, X_val_nlp, y_val,
                X_test_struct, X_test_nlp, y_test.
        """
        val_df = pd.read_csv(self.config.val_data_path)
        test_df = pd.read_csv(self.config.test_data_path)
        target = self.config.target_column

        def _extract(df: pd.DataFrame, prefixes: tuple[str, ...]) -> np.ndarray:
            cols = _get_branch_columns(df, prefixes)
            return df[cols].to_numpy()

        X_val_struct = _extract(val_df, STRUCTURED_PREFIX)
        X_val_nlp = _extract(val_df, (NLP_PREFIX,))
        y_val_series: pd.Series = cast(pd.Series, val_df[target])
        y_val, _ = _encode_target(y_val_series)

        X_test_struct = _extract(test_df, STRUCTURED_PREFIX)
        X_test_nlp = _extract(test_df, (NLP_PREFIX,))
        y_test_series: pd.Series = cast(pd.Series, test_df[target])
        y_test, _ = _encode_target(y_test_series)

        return (
            X_val_struct,
            X_val_nlp,
            y_val,
            X_test_struct,
            X_test_nlp,
            y_test,
        )

    def evaluate(self) -> dict[str, Any]:
        """Runs all three evaluation runs and writes the consolidated report.

        Workflow:
            1. Load models and feature splits.
            2. Run MLflow-tracked evaluation for Branch 1 (structured).
            3. Run MLflow-tracked evaluation for Branch 2 (NLP).
            4. Run MLflow-tracked evaluation for Late Fusion (stacked).
            5. Compute and log recall_lift and f1_lift over structured baseline.
            6. Register champion model in the MLflow Model Registry.
            7. Write evaluation_report.json for DVC tracking.

        Returns:
            Consolidated evaluation report as a dictionary.
        """
        mlflow.set_tracking_uri(self.config.mlflow_uri)
        mlflow.set_experiment(self.config.experiment_name)

        structured_model, nlp_model, meta_model = self._load_models()

        (
            X_val_struct,
            X_val_nlp,
            y_val,
            X_test_struct,
            X_test_nlp,
            y_test,
        ) = self._get_val_test_arrays()

        report: dict[str, Any] = {}

        # ----------------------------------------------------------------
        # Run 1: Structured Baseline
        # ----------------------------------------------------------------
        logger.info("=== MLflow Run: structured_baseline ===")
        with mlflow.start_run(run_name="structured_baseline") as run_struct:
            y_pred_struct = structured_model.predict(X_test_struct)
            y_prob_struct = structured_model.predict_proba(X_test_struct)[:, 1]
            metrics_struct = _compute_metrics(y_test, y_pred_struct, y_prob_struct)

            mlflow.log_params({"branch": "structured", "model": "xgboost"})
            mlflow.log_metrics(metrics_struct)
            # Use explicit submodule import if needed, but here we assume it's correctly imported as mlflow.xgboost
            # Pyright might need help:
            xgboost.log_model(structured_model, artifact_path="model")  # type: ignore

            _log_confusion_matrix(y_test, y_pred_struct, "structured_baseline", self._artifact_dir)
            _log_feature_importance(structured_model, "structured_baseline", self._artifact_dir)

            report["structured_baseline"] = {
                "run_id": run_struct.info.run_id,
                "metrics": metrics_struct,
            }
            logger.info(f"Structured baseline metrics: {metrics_struct}")

        # ----------------------------------------------------------------
        # Run 2: NLP Baseline
        # ----------------------------------------------------------------
        logger.info("=== MLflow Run: nlp_baseline ===")
        with mlflow.start_run(run_name="nlp_baseline") as run_nlp:
            y_pred_nlp = nlp_model.predict(X_test_nlp)
            y_prob_nlp = nlp_model.predict_proba(X_test_nlp)[:, 1]
            metrics_nlp = _compute_metrics(y_test, y_pred_nlp, y_prob_nlp)

            mlflow.log_params({"branch": "nlp", "model": "xgboost"})
            mlflow.log_metrics(metrics_nlp)
            xgboost.log_model(nlp_model, artifact_path="model")  # type: ignore

            _log_confusion_matrix(y_test, y_pred_nlp, "nlp_baseline", self._artifact_dir)
            _log_feature_importance(nlp_model, "nlp_baseline", self._artifact_dir)

            report["nlp_baseline"] = {
                "run_id": run_nlp.info.run_id,
                "metrics": metrics_nlp,
            }
            logger.info(f"NLP baseline metrics: {metrics_nlp}")

        # ----------------------------------------------------------------
        # Run 3: Late Fusion (Stacked)
        # ----------------------------------------------------------------
        logger.info("=== MLflow Run: late_fusion_stacked ===")
        with mlflow.start_run(run_name="late_fusion_stacked") as run_fusion:
            # Stack test probabilities from both base models
            p_struct = structured_model.predict_proba(X_test_struct)[:, 1]
            p_nlp = nlp_model.predict_proba(X_test_nlp)[:, 1]
            stack_test = np.column_stack([p_struct, p_nlp])

            y_pred_fusion = meta_model.predict(stack_test)
            y_prob_fusion = meta_model.predict_proba(stack_test)[:, 1]
            metrics_fusion = _compute_metrics(y_test, y_pred_fusion, y_prob_fusion)

            # Business-facing lift metrics over structured baseline
            recall_lift = metrics_fusion["recall"] - metrics_struct["recall"]
            f1_lift = metrics_fusion["f1"] - metrics_struct["f1"]

            mlflow.log_params(
                {
                    "branch": "fusion",
                    "model": "stacked_logistic_regression",
                    "meta_C": self.config.meta_C,
                    "cv_folds": self.config.cv_folds,
                }
            )
            mlflow.log_metrics(
                {
                    **metrics_fusion,
                    "recall_lift": recall_lift,
                    "f1_lift": f1_lift,
                }
            )
            sklearn.log_model(
                meta_model,
                artifact_path="model",
                registered_model_name="telco-churn-late-fusion",
            )

            _log_confusion_matrix(y_test, y_pred_fusion, "late_fusion_stacked", self._artifact_dir)

            report["late_fusion_stacked"] = {
                "run_id": run_fusion.info.run_id,
                "metrics": metrics_fusion,
                "recall_lift": recall_lift,
                "f1_lift": f1_lift,
            }
            logger.info(f"Late Fusion metrics: {metrics_fusion}")
            logger.info(f"Recall lift over structured baseline: {recall_lift:+.4f}")
            logger.info(f"F1 lift over structured baseline: {f1_lift:+.4f}")

        # ----------------------------------------------------------------
        # Write consolidated evaluation_report.json (DVC output / CI gate)
        # ----------------------------------------------------------------
        report["meta"] = {
            "target_column": self.config.target_column,
            "cv_folds": self.config.cv_folds,
            "structured_n_trials": self.config.structured_n_trials,
            "nlp_n_trials": self.config.nlp_n_trials,
        }

        report_path = Path(self.config.evaluation_report_path)
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)

        logger.info(f"Evaluation report written to: {self.config.evaluation_report_path}")
        return report
