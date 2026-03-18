"""
Feature Engineering Utilities.

This module contains custom Scikit-Learn transformers used in the Feature Engineering
stage. These classes MUST be imported here to satisfy the Anti-Skew Mandate.
By maintaining the logic in this standalone utility file, the inference API can safely
unpickle the preprocessor pipeline without duplicating the logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextEmbedder(BaseEstimator, TransformerMixin):
    """Custom Scikit-Learn Transformer for generating text embeddings.

    Uses the `sentence-transformers` library to convert a column of text
    into a high-dimensional numpy array of embeddings.

    To prevent `joblib` from inflating the size of the serialized pipeline
    by pickling the entire PyTorch model, the model is loaded lazily and
    discarded during pickling via `__getstate__`.

    Attributes:
        model_name (str): The Hugging Face model identifier for SentenceTransformers.
        _model: The lazily loaded SentenceTransformer instance.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
        """Initializes the TextEmbedder.

        Args:
            model_name: The SentenceTransformer model to load.
        """
        self.model_name = model_name
        self._model: Any = None

    @property
    def model(self) -> Any:
        """Lazily loads the SentenceTransformer model.

        Returns:
            The SentenceTransformer model instance.
        """
        if self._model is None:
            logger.info(f"Loading SentenceTransformer model: {self.model_name}")
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def fit(self, X: Any, y: Any = None) -> TextEmbedder:
        """Fits the transformer.

        Since this is a pre-trained embedding model, fit does nothing
        but return self.

        Args:
            X: Input text data (not used during fit).
            y: Target variable (ignored).

        Returns:
            The fitted transformer instance (self).
        """
        return self

    def transform(self, X: Any) -> np.ndarray[Any, Any]:
        """Transforms a list or array of text into embeddings.

        Args:
            X: Input text data (Series, DataFrame, or numpy array).

        Returns:
            A numpy array of shape (n_samples, embedding_dim).
        """
        # Ensure 1D list of strings
        X_flat = np.array(X).ravel()
        texts = X_flat.tolist()

        # Handle potential NaNs by converting them to empty strings
        texts = [str(text) if pd.notna(text) else "" for text in texts]

        # Use the property to ensure the model is loaded
        embeddings = self.model.encode(texts, show_progress_bar=False)
        return np.array(embeddings)

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        """Returns feature names for the embeddings based on model dimension."""
        dim = self.model.get_sentence_embedding_dimension()
        prefix = (
            input_features[0]
            if input_features is not None and len(input_features) > 0
            else "emb"
        )
        return np.array([f"{prefix}_{i}" for i in range(dim)], dtype=object)

    def __getstate__(self) -> dict[str, Any]:
        """Custom unpickling to drop the PyTorch model.

        Returns:
            dict: The state dictionary to serialize, minus the heavy model.
        """
        state = self.__dict__.copy()
        # Do not pickle the PyTorch model to save disk space and
        # prevent potential unpickling errors cross-platform.
        state["_model"] = None
        return state


class NumericCleaner(BaseEstimator, TransformerMixin):
    """Robust numeric coercer for dealing with blank strings in numeric columns.

    In the raw Telco data, `TotalCharges` contains blank strings (" ") for customers
    with 0 tenure. This transformer safely coerces them to `np.nan` so they can be
    handled correctly by downstream imputers (e.g., SimpleImputer).
    """

    def fit(self, X: pd.DataFrame | np.ndarray[Any, Any], y: Any = None) -> NumericCleaner:
        """Stateless transformer, returns self."""
        return self

    def transform(self, X: pd.DataFrame | np.ndarray[Any, Any]) -> pd.DataFrame:
        """Coerce all inputs to numeric, setting errors to NaN."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # Coerce each column to numeric
        return X.apply(pd.to_numeric, errors="coerce")

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        """Preserve feature names."""
        return input_features
