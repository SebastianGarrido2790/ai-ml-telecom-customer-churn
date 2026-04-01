"""
Late Fusion Model Trainer.

This module implements the three-stage Late Fusion stacking architecture:
    - Branch 1 (Structured Baseline): XGBoost trained on structured features.
    - Branch 2 (NLP Baseline): XGBoost trained on PCA-reduced NLP embeddings.
    - Meta-Learner (Late Fusion): Logistic Regression stacker trained on
      Out-of-Fold (OOF) probability predictions from both base models.

Design Decisions:
    - SMOTE is applied independently per branch (Decision B1), ensuring
      synthetic neighbor computation operates in each branch's own geometric
      space rather than across the mixed structured+NLP feature space.
    - OOF stacking is used to train the meta-learner, preventing leakage
      from the base model training set into the meta-learner fit.
    - primary_sentiment_tag is excluded from all branches (Decision A2) as
      it was generated with churn label context and functions as a
      near-deterministic proxy of the target variable.
    - Both base models are retrained on the full SMOTE-augmented training
      set after OOF generation, maximising the signal available for inference.

Mlflow ui:
    uv run mlflow ui --backend-store-uri file:./mlruns
"""

from __future__ import annotations

from typing import cast

import joblib
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.entity.config_entity import ModelTrainingConfig
from src.utils.logger import get_logger

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column prefix constants — must match feature_engineering.py output headers
# ---------------------------------------------------------------------------
STRUCTURED_PREFIX = ("num__", "cat__")
NLP_PREFIX = "nlp__"
ID_COL = "customerID"


def _get_branch_columns(df: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    """Returns feature column names matching the given prefixes.

    Args:
        df: Feature DataFrame with prefixed column names.
        prefixes: Tuple of column name prefixes to match.

    Returns:
        List of matching column names, excluding ID and target columns.
    """
    return [c for c in df.columns if c.startswith(prefixes)]


def _encode_target(y: pd.Series) -> tuple[np.ndarray, LabelEncoder]:
    """Encodes the binary string target to integer labels.

    Args:
        y: Target Series with string values (e.g., 'Yes' / 'No').

    Returns:
        Tuple of (encoded integer array, fitted LabelEncoder).
    """
    le = LabelEncoder()
    encoded: np.ndarray = le.fit_transform(y)  # type: ignore
    return encoded.astype(int), le


def _apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies SMOTE oversampling to balance the binary training set.

    SMOTE is applied exclusively to the training set. Validation and test
    sets are never oversampled to preserve evaluation realism.

    Args:
        X: Training feature matrix.
        y: Encoded training target array.
        random_state: Seed for reproducibility.

    Returns:
        Tuple of (resampled feature matrix, resampled target array).
    """
    smote = SMOTE(random_state=random_state)
    resampled = smote.fit_resample(X, y)
    X_res: np.ndarray = resampled[0]  # type: ignore
    y_res: np.ndarray = resampled[1]  # type: ignore
    logger.info(
        f"SMOTE applied: {X.shape[0]} → {X_res.shape[0]} samples "
        f"(added {X_res.shape[0] - X.shape[0]} synthetic minority samples)"
    )
    return X_res, y_res


def _optuna_xgb_objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
) -> float:
    """Optuna objective function for XGBoost hyperparameter search.

    Optimises for Recall on the validation set (primary business metric).
    Recall is prioritised over F1 to minimise False Negatives — missing
    a churner is more expensive than a False Positive.

    Args:
        trial: Optuna trial object for hyperparameter suggestion.
        X_train: Training feature matrix (post-SMOTE).
        y_train: Encoded training target array (post-SMOTE).
        X_val: Validation feature matrix (pre-SMOTE, unmodified).
        y_val: Encoded validation target array.
        random_state: Seed for reproducibility.

    Returns:
        Recall score on the validation set.
    """
    from sklearn.metrics import recall_score

    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 3, 9),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "random_state": random_state,
        "eval_metric": "logloss",
        "use_label_encoder": False,
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = model.predict(X_val)
    return recall_score(y_val, y_pred)


def _tune_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int,
    random_state: int,
    branch_name: str,
) -> XGBClassifier:
    """Runs Optuna hyperparameter search and returns the best XGBoost model.

    The best model is refitted on the full SMOTE-augmented training set
    using the optimal hyperparameters found during search.

    Args:
        X_train: SMOTE-augmented training feature matrix.
        y_train: SMOTE-augmented training target array.
        X_val: Unmodified validation feature matrix.
        y_val: Unmodified validation target array.
        n_trials: Number of Optuna trials to run.
        random_state: Seed for reproducibility.
        branch_name: Human-readable branch label for logging.

    Returns:
        XGBClassifier fitted on the full training set with optimal params.
    """
    logger.info(f"[{branch_name}] Starting Optuna search ({n_trials} trials).")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(
        lambda trial: _optuna_xgb_objective(trial, X_train, y_train, X_val, y_val, random_state),
        n_trials=n_trials,
        show_progress_bar=False,
    )

    best_params = study.best_params
    best_recall = study.best_value
    logger.info(f"[{branch_name}] Best recall: {best_recall:.4f} | Params: {best_params}")

    best_model = XGBClassifier(
        **best_params,
        random_state=random_state,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    best_model.fit(X_train, y_train)
    return best_model


class LateFusionTrainer:
    """Orchestrates the full Late Fusion stacking training protocol.

    Produces three serialized model artifacts:
        - structured_model.pkl: XGBoost trained on structured features.
        - nlp_model.pkl: XGBoost trained on PCA-reduced NLP embeddings.
        - meta_model.pkl: Logistic Regression stacker (OOF-trained).

    Attributes:
        config: Immutable training configuration entity.
    """

    def __init__(self, config: ModelTrainingConfig) -> None:
        """Initializes the LateFusionTrainer.

        Args:
            config: ModelTrainingConfig with all paths and hyperparameters.
        """
        self.config = config

    def _load_splits(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Loads train, validation, and test feature CSVs.

        Returns:
            Tuple of (train_df, val_df, test_df) DataFrames.
        """
        logger.info("Loading feature splits from Feature Store.")
        train_df = pd.read_csv(self.config.train_data_path)
        val_df = pd.read_csv(self.config.val_data_path)
        test_df = pd.read_csv(self.config.test_data_path)
        logger.info(f"Loaded — Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")
        return train_df, val_df, test_df

    def _extract_branch(
        self,
        df: pd.DataFrame,
        prefixes: tuple[str, ...],
        target_col: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extracts branch-specific feature matrix and encoded target array.

        Args:
            df: Full feature DataFrame including all branches and target.
            prefixes: Column name prefixes identifying this branch's features.
            target_col: Name of the target column.

        Returns:
            Tuple of (feature matrix as ndarray, encoded target as ndarray).
        """
        feature_cols = _get_branch_columns(df, prefixes)
        X = df[feature_cols].to_numpy()
        y_series: pd.Series = cast(pd.Series, df[target_col])
        y, _ = _encode_target(y_series)
        return X, y

    def train(self) -> tuple[XGBClassifier, XGBClassifier, LogisticRegression]:
        """Executes the complete Late Fusion training protocol.

        Protocol:
            1. Load train/val/test splits from the Feature Store.
            2. Extract structured and NLP feature columns per branch.
            3. Apply SMOTE independently to each branch's training set.
            4. Generate Out-of-Fold (OOF) probability predictions for both
               base models using StratifiedKFold cross-validation on the
               pre-SMOTE training set (prevents meta-learner leakage).
            5. Train Logistic Regression meta-learner on stacked OOF arrays.
            6. Retrain both base models on the full SMOTE-augmented train set.
            7. Serialize all three model artifacts.

        Returns:
            Tuple of (structured_model, nlp_model, meta_model).
        """
        train_df, val_df, _ = self._load_splits()
        target = self.config.target_column
        rs = self.config.random_state

        # ----------------------------------------------------------------
        # Branch 1: Structured features
        # ----------------------------------------------------------------
        logger.info("=== Branch 1: Structured Baseline ===")
        X_train_struct, y_train_struct = self._extract_branch(train_df, STRUCTURED_PREFIX, target)
        X_val_struct, y_val_struct = self._extract_branch(val_df, STRUCTURED_PREFIX, target)

        X_train_struct_sm, y_train_struct_sm = _apply_smote(X_train_struct, y_train_struct, rs)

        # OOF predictions on pre-SMOTE train set (prevents meta-learner leakage)
        logger.info("[Branch 1] Generating OOF predictions for stacking.")
        struct_oof_model = XGBClassifier(
            n_estimators=300,
            random_state=rs,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=rs)
        cv_res_struct = cast(
            np.ndarray,
            cross_val_predict(
                struct_oof_model,
                X_train_struct,
                y_train_struct,
                cv=cv,
                method="predict_proba",
            ),
        )
        oof_struct = cv_res_struct[:, 1]  # probability of Churn=Yes

        # Tune and retrain on full SMOTE-augmented train set
        structured_model = _tune_xgboost(
            X_train_struct_sm,
            y_train_struct_sm,
            X_val_struct,
            y_val_struct,
            n_trials=self.config.structured_n_trials,
            random_state=rs,
            branch_name="Branch 1 / Structured",
        )

        # ----------------------------------------------------------------
        # Branch 2: NLP features
        # ----------------------------------------------------------------
        logger.info("=== Branch 2: NLP Baseline ===")
        X_train_nlp, y_train_nlp = self._extract_branch(train_df, (NLP_PREFIX,), target)
        X_val_nlp, y_val_nlp = self._extract_branch(val_df, (NLP_PREFIX,), target)

        X_train_nlp_sm, y_train_nlp_sm = _apply_smote(X_train_nlp, y_train_nlp, rs)

        logger.info("[Branch 2] Generating OOF predictions for stacking.")
        nlp_oof_model = XGBClassifier(
            n_estimators=300,
            random_state=rs,
            eval_metric="logloss",
            use_label_encoder=False,
        )
        cv_res_nlp = cast(
            np.ndarray,
            cross_val_predict(
                nlp_oof_model,
                X_train_nlp,
                y_train_nlp,
                cv=cv,
                method="predict_proba",
            ),
        )
        oof_nlp = cv_res_nlp[:, 1]

        nlp_model = _tune_xgboost(
            X_train_nlp_sm,
            y_train_nlp_sm,
            X_val_nlp,
            y_val_nlp,
            n_trials=self.config.nlp_n_trials,
            random_state=rs,
            branch_name="Branch 2 / NLP",
        )

        # ----------------------------------------------------------------
        # Meta-Learner: Logistic Regression on stacked OOF arrays
        # ----------------------------------------------------------------
        logger.info("=== Meta-Learner: Late Fusion Stacking ===")
        # Ensure oof arrays are 1D and of correct type for column_stack
        oof_stack = np.column_stack([oof_struct.astype(float), oof_nlp.astype(float)])
        logger.info(f"OOF stack shape: {oof_stack.shape} (samples × [P_struct, P_nlp])")

        meta_model = LogisticRegression(
            C=self.config.meta_C,
            max_iter=self.config.meta_max_iter,
            random_state=rs,
        )
        meta_model.fit(oof_stack, y_train_struct)  # same y; both branches share target
        logger.info("Meta-learner fitted on OOF stacked probabilities.")

        # ----------------------------------------------------------------
        # Serialize all three model artifacts
        # ----------------------------------------------------------------
        logger.info(f"Saving structured model → {self.config.structured_model_path}")
        joblib.dump(structured_model, self.config.structured_model_path)

        logger.info(f"Saving NLP model → {self.config.nlp_model_path}")
        joblib.dump(nlp_model, self.config.nlp_model_path)

        logger.info(f"Saving meta-model → {self.config.meta_model_path}")
        joblib.dump(meta_model, self.config.meta_model_path)

        return structured_model, nlp_model, meta_model
