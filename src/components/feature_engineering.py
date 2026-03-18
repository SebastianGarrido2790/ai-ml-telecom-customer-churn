"""
Feature Engineering Component.

This module implements the categorical encoding, numerical scaling, and NLP-based
text embedding transformations for the Telco Churn dataset. It follows the FTI
Feature layer pattern to provide a production-ready, serialized preprocessor.
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


class FeatureEngineering:
    """Component for applying the NLP Feature Engineering & ML Transformations.

    Implements the 'Mechanic' layer of the FTI pattern. Applies text embeddings
    and standard scaler/encoders to the data, completely fitting on Train
    and applying identical logic to Val/Test to prevent Train-Serving Skew.
    """

    def __init__(self, config: FeatureEngineeringConfig) -> None:
        """Initializes the Feature Engineering component.

        Args:
            config: Configuration for artifact paths and hyperparameters.
        """
        self.config = config

    def get_preprocessor(self) -> ColumnTransformer:
        """Constructs the unified Scikit-Learn pipeline.

        Returns:
            ColumnTransformer: The combined preprocessor pipeline.
        """
        # Define column groupings
        numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
        categorical_cols = [
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
            "primary_sentiment_tag",
        ]
        nlp_cols = ["ticket_note"]  # List forces 2D DataFrame for SimpleImputer

        # 1. Numeric Pipeline
        numeric_pipeline = Pipeline(
            steps=[
                ("cleaner", NumericCleaner()),
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # 2. Categorical Pipeline
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )

        # 3. NLP Pipeline (Ticket Notes)
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

        # Assemble the full preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, numeric_cols),
                ("cat", categorical_pipeline, categorical_cols),
                ("nlp", nlp_pipeline, nlp_cols),
            ],
            remainder="drop",  # Target and IDs will be handled separately
        )

        return preprocessor

    def initiate_feature_engineering(self) -> None:
        """Executes the complete feature engineering process.

        1. Loads the enriched data.
        2. Splits into Train, Validation, and Test sets.
        3. Fits the preprocessor on Train ONLY.
        4. Transforms all three sets.
        5. Combines features back with IDs and target.
        6. Saves resulting CSVs and the preprocessor artifact.
        """
        logger.info(f"Loading data from {self.config.input_data_path}")
        df = pd.read_csv(self.config.input_data_path)

        # Separate Identifiers and Target
        target = self.config.target_column
        identifiers = ["customerID"]

        # Drop identifiers and target from X
        X = df.drop(columns=identifiers + [target])
        y = df[target]

        logger.info(
            f"Performing 3-way split (Val: {self.config.val_size}, Test: {self.config.test_size})"
        )

        # Split 1: Extract Test Set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=self.config.random_state, stratify=y
        )

        # Calculate validation proportion from the remaining 'temp' portion
        val_prop = self.config.val_size / (1.0 - self.config.test_size)

        # Split 2: Extract Train and Val Sets
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp,
            y_temp,
            test_size=val_prop,
            random_state=self.config.random_state,
            stratify=y_temp,
        )

        logger.info(
            f"Split completed. Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}"
        )

        # Construct and fit the pipeline
        preprocessor = self.get_preprocessor()

        # For sklearn column transformers to output custom DataFrame headers we can use set_output
        preprocessor.set_output(transform="pandas")

        logger.info("Fitting the preprocessor on the Train set ONLY.")
        X_train_transformed = preprocessor.fit_transform(X_train)

        logger.info("Transforming Val and Test sets.")
        X_val_transformed = preprocessor.transform(X_val)
        X_test_transformed = preprocessor.transform(X_test)

        # Force index alignment to prevent outer join issues in pd.concat
        if isinstance(X_train_transformed, pd.DataFrame):
            X_train_transformed.index = X_train.index
            X_val_transformed.index = X_val.index
            X_test_transformed.index = X_test.index
        else:
            try:
                # If set_output('pandas') succeeded on inner steps but not the outer, we can try to get column names:
                cols = preprocessor.get_feature_names_out()
            except AttributeError:
                cols = None
            X_train_transformed = pd.DataFrame(
                X_train_transformed, index=X_train.index, columns=cols
            )
            X_val_transformed = pd.DataFrame(X_val_transformed, index=X_val.index, columns=cols)
            X_test_transformed = pd.DataFrame(
                X_test_transformed, index=X_test.index, columns=cols
            )

        # Re-attach Identifiers and Target for saving
        train_full = pd.concat(
            [df.loc[X_train.index, identifiers], X_train_transformed, y_train], axis=1
        )
        val_full = pd.concat([df.loc[X_val.index, identifiers], X_val_transformed, y_val], axis=1)
        test_full = pd.concat(
            [df.loc[X_test.index, identifiers], X_test_transformed, y_test], axis=1
        )

        # Save the datasets
        logger.info("Saving transformed dataset artifacts.")
        train_full.to_csv(self.config.train_data_path, index=False)
        val_full.to_csv(self.config.val_data_path, index=False)
        test_full.to_csv(self.config.test_data_path, index=False)

        # Serialize the preprocessor
        logger.info(f"Saving serialized preprocessor to {self.config.preprocessor_path}")
        joblib.dump(preprocessor, self.config.preprocessor_path)

        logger.info("Feature Engineering phase completed successfully.")
