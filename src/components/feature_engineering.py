"""
Feature Engineering Component.

This module implements the categorical encoding, numerical scaling, and NLP-based
text embedding transformations for the Telco Churn dataset. It follows the FTI
Feature layer pattern to provide two production-ready, independently serialized
preprocessors that enforce the Anti-Skew Mandate.

Preprocessor Split Design:
    - structured_preprocessor.pkl: Numeric + Categorical pipelines only.
      Used by Branch 1 (Structured Baseline) in Phase 5 and the Prediction
      API in Phase 6.
    - nlp_preprocessor.pkl: TextEmbedder + PCA pipeline only.
      Used by Branch 2 (NLP Baseline) in Phase 5 and the Embedding
      Microservice in Phase 6.

    primary_sentiment_tag is intentionally excluded from both preprocessors.
    Its near-deterministic correlation with the target variable (Decision A2)
    makes it unsuitable as a training signal. It is retained in the raw
    enriched CSV for diagnostic and interpretability use only.
"""

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.entity.config_entity import FeatureEngineeringConfig
from src.utils.feature_utils import NumericCleaner, TextEmbedder
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Column definitions — single source of truth for both preprocessors.
# Modifying these lists automatically propagates to training and inference.
# ---------------------------------------------------------------------------
NUMERIC_COLS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]

CATEGORICAL_COLS: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

NLP_COLS: list[str] = ["ticket_note"]

# Excluded from training per Decision A2 (near-deterministic target proxy).
# Retained in raw CSV for diagnostic purposes.
DIAGNOSTIC_COLS: list[str] = ["primary_sentiment_tag"]


class FeatureEngineering:
    """Component for applying split NLP and structured ML transformations.

    Implements the 'Mechanic' layer of the FTI pattern. Produces two
    independently serialized preprocessors — structured and NLP — each
    fitted exclusively on the training set to prevent training-serving skew.
    """

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        """Initializes the Feature Engineering component.

        Args:
            config: Configuration for artifact paths and hyperparameters.
        """
        self.config = config

    def get_structured_preprocessor(self) -> ColumnTransformer:
        """Constructs the structured (numeric + categorical) preprocessor.

        This preprocessor handles only tabular features. It is the artifact
        loaded by Branch 1 (Phase 5) and the Prediction API (Phase 6).

        Returns:
            ColumnTransformer: Structured feature preprocessor pipeline.
        """
        numeric_pipeline = Pipeline(
            steps=[
                ("cleaner", NumericCleaner()),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                (
                    "ohe",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, NUMERIC_COLS),
                ("cat", categorical_pipeline, CATEGORICAL_COLS),
            ],
            remainder="drop",
        )

    def get_nlp_preprocessor(self) -> ColumnTransformer:
        """Constructs the NLP (TextEmbedder + PCA) preprocessor.

        This preprocessor handles only the ticket_note text column. It is
        the artifact loaded by Branch 2 (Phase 5) and the Embedding
        Microservice (Phase 6).

        Returns:
            ColumnTransformer: NLP feature preprocessor pipeline.
        """
        nlp_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                ("embedder", TextEmbedder(model_name=self.config.embedding_model_name)),
                (
                    "pca",
                    PCA(
                        n_components=self.config.pca_components,
                        random_state=self.config.random_state,
                    ),
                ),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("nlp", nlp_pipeline, NLP_COLS),
            ],
            remainder="drop",
        )

    def _align_to_dataframe(
        self,
        transformed: pd.DataFrame | object,
        reference_index: pd.Index,
        preprocessor: ColumnTransformer,
    ) -> pd.DataFrame:
        """Forces index alignment on transformer output to prevent concat issues.

        Args:
            transformed: Output of fit_transform or transform.
            reference_index: Original DataFrame index to restore.
            preprocessor: The fitted ColumnTransformer (for feature name extraction).

        Returns:
            DataFrame with aligned index and named columns.
        """
        if isinstance(transformed, pd.DataFrame):
            transformed.index = reference_index
            return transformed

        try:
            cols = preprocessor.get_feature_names_out()
        except AttributeError:
            cols = None

        return pd.DataFrame(transformed, index=reference_index, columns=cols)

    def initiate_feature_engineering(self) -> None:
        """Executes the complete feature engineering process.

        Workflow:
            1. Load the enriched, validated dataset.
            2. Perform stratified 3-way split (train / val / test).
            3. Fit both preprocessors on the training set ONLY.
            4. Transform all three splits with each preprocessor identically.
            5. Merge transformed features with identifiers and target.
            6. Serialize both preprocessors and save all feature CSVs.

        Note:
            primary_sentiment_tag is excluded from all preprocessors (Decision A2).
            The column remains in the enriched CSV for diagnostic inspection.
        """
        logger.info(f"Loading enriched data from: {self.config.input_data_path}")
        df = pd.read_csv(self.config.input_data_path)

        target = self.config.target_column
        identifiers = ["customerID"]
        excluded = identifiers + [target] + DIAGNOSTIC_COLS

        X = df.drop(columns=excluded)
        y = df[target]

        logger.info(f"Performing 3-way stratified split (val={self.config.val_size}, test={self.config.test_size})")

        from typing import Any, cast

        X_temp: pd.DataFrame
        X_test: pd.DataFrame
        y_temp: pd.Series
        y_test: pd.Series

        (X_temp, X_test, y_temp, y_test) = cast(
            Any,
            train_test_split(
                X,
                y,
                test_size=self.config.test_size,
                random_state=self.config.random_state,
                stratify=y,
            ),
        )

        val_prop = self.config.val_size / (1.0 - self.config.test_size)

        X_train: pd.DataFrame
        X_val: pd.DataFrame
        y_train: pd.Series
        y_val: pd.Series

        (X_train, X_val, y_train, y_val) = cast(
            Any,
            train_test_split(
                X_temp,
                y_temp,
                test_size=val_prop,
                random_state=self.config.random_state,
                stratify=y_temp,
            ),
        )

        logger.info(f"Split sizes — Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

        # ----------------------------------------------------------------
        # Structured preprocessor: fit on train, transform all splits
        # ----------------------------------------------------------------
        logger.info("Fitting structured preprocessor on training set.")
        structured_preprocessor = self.get_structured_preprocessor()
        structured_preprocessor.set_output(transform="pandas")

        X_train_struct = self._align_to_dataframe(
            structured_preprocessor.fit_transform(X_train[NUMERIC_COLS + CATEGORICAL_COLS]),
            X_train.index,
            structured_preprocessor,
        )
        X_val_struct = self._align_to_dataframe(
            structured_preprocessor.transform(X_val[NUMERIC_COLS + CATEGORICAL_COLS]),
            X_val.index,
            structured_preprocessor,
        )
        X_test_struct = self._align_to_dataframe(
            structured_preprocessor.transform(X_test[NUMERIC_COLS + CATEGORICAL_COLS]),
            X_test.index,
            structured_preprocessor,
        )

        # ----------------------------------------------------------------
        # NLP preprocessor: fit on train, transform all splits
        # ----------------------------------------------------------------
        logger.info("Fitting NLP preprocessor on training set.")
        nlp_preprocessor = self.get_nlp_preprocessor()
        nlp_preprocessor.set_output(transform="pandas")

        X_train_nlp = self._align_to_dataframe(
            nlp_preprocessor.fit_transform(X_train[NLP_COLS]),
            X_train.index,
            nlp_preprocessor,
        )
        X_val_nlp = self._align_to_dataframe(
            nlp_preprocessor.transform(X_val[NLP_COLS]),
            X_val.index,
            nlp_preprocessor,
        )
        X_test_nlp = self._align_to_dataframe(
            nlp_preprocessor.transform(X_test[NLP_COLS]),
            X_test.index,
            nlp_preprocessor,
        )

        # ----------------------------------------------------------------
        # Merge structured + NLP features and re-attach identifiers/target
        # ----------------------------------------------------------------
        logger.info("Merging structured and NLP feature sets.")

        def _build_full(
            struct: pd.DataFrame,
            nlp: pd.DataFrame,
            y_split: pd.Series,
            idx: pd.Index,
        ) -> pd.DataFrame:
            return pd.concat(
                [df.loc[idx, identifiers], struct, nlp, y_split],
                axis=1,
            )

        train_full = _build_full(X_train_struct, X_train_nlp, y_train, X_train.index)
        val_full = _build_full(X_val_struct, X_val_nlp, y_val, X_val.index)
        test_full = _build_full(X_test_struct, X_test_nlp, y_test, X_test.index)

        # ----------------------------------------------------------------
        # Persist feature CSVs and both preprocessor artifacts
        # ----------------------------------------------------------------
        logger.info("Saving feature CSVs.")
        train_full.to_csv(self.config.train_data_path, index=False)
        val_full.to_csv(self.config.val_data_path, index=False)
        test_full.to_csv(self.config.test_data_path, index=False)

        logger.info(f"Saving structured preprocessor to: {self.config.structured_preprocessor_path}")
        joblib.dump(structured_preprocessor, self.config.structured_preprocessor_path)

        logger.info(f"Saving NLP preprocessor to: {self.config.nlp_preprocessor_path}")
        joblib.dump(nlp_preprocessor, self.config.nlp_preprocessor_path)

        logger.info(
            f"Feature engineering complete. "
            f"Train shape: {train_full.shape}, "
            f"Val shape: {val_full.shape}, "
            f"Test shape: {test_full.shape}"
        )
