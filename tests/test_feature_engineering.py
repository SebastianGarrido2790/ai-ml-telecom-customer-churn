"""
Unit Tests for the Feature Engineering Stage.

This suite ensures the correctness of custom Scikit-Learn transformers (NumericCleaner,
TextEmbedder) and the overall FeatureEngineering component's logic.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from src.components.feature_engineering import FeatureEngineering
from src.entity.config_entity import FeatureEngineeringConfig
from src.utils.feature_utils import NumericCleaner, TextEmbedder


class TestNumericCleaner:
    """Suite for testing the NumericCleaner custom transformer."""

    def test_fit_transform(self):
        """Test that NumericCleaner correctly coerces numeric data."""
        df = pd.DataFrame(
            {
                "A": ["1", "2.5", " ", "", np.nan, "10"],
                "B": [1.0, 2.5, np.nan, np.nan, np.nan, 10.0],
            }
        )

        cleaner = NumericCleaner()
        # Fit does nothing, but check it returns self
        assert cleaner.fit(df) is cleaner

        # Transform should coerce 'A' correctly, handling empty strings as NaNs
        transformed_df = cleaner.transform(df[["A"]])

        # Expected output includes NaNs for " ", "", and np.nan
        expected_A = [1.0, 2.5, np.nan, np.nan, np.nan, 10.0]

        # Check equality (pd.testing.assert_series_equal for exact pandas match)
        pd.testing.assert_series_equal(transformed_df["A"], pd.Series(expected_A, name="A"))

    def test_feature_names_out(self):
        """Test that get_feature_names_out returns the correct input feature names."""
        cleaner = NumericCleaner()
        input_features = np.array(["Col1", "Col2"])
        output_features = cleaner.get_feature_names_out(input_features)

        np.testing.assert_array_equal(output_features, input_features)


class TestTextEmbedder:
    """Suite for testing the TextEmbedder custom transformer with lazy model loading."""

    def test_lazy_loading(self, mock_sentence_transformer):
        """Test that the model isn't loaded until the 'model' property is accessed."""
        embedder = TextEmbedder("test-model")
        # Should not be initialized immediately
        assert embedder._model is None
        mock_sentence_transformer.assert_not_called()

        # Accessing the property loads the model
        _ = embedder.model
        mock_sentence_transformer.assert_called_once_with("test-model")
        assert embedder._model is not None

    def test_fit_transform(self, mock_sentence_transformer):
        """Test that fit_transform generates expected output structure."""
        embedder = TextEmbedder("test-model")
        texts = pd.Series(["hello world", "foo bar", "test string"])

        # Fit should return self
        assert embedder.fit(texts) is embedder

        # Transform should produce a numpy array
        result = embedder.transform(texts)
        assert isinstance(result, np.ndarray)
        assert result.shape == (3, 384)

    def test_get_feature_names_out(self, mock_sentence_transformer):
        """Test get_feature_names_out returns expected column names."""
        embedder = TextEmbedder("test-model")
        feature_names = embedder.get_feature_names_out(input_features=["custom_col"])

        assert len(feature_names) == 384
        assert feature_names[0] == "custom_col_0"

    def test_pickling_behavior(self, mock_sentence_transformer):
        """Test that unpickling gracefully handles the lack of the deep Torch model."""
        embedder = TextEmbedder("test-model")
        # Force initialization
        _ = embedder.model

        # Simulate pickling and unpickling
        pickled_embedder = pickle.dumps(embedder)
        unpickled_embedder = pickle.loads(pickled_embedder)

        # Verify the model property was dropped during pickling
        assert unpickled_embedder._model is None
        assert unpickled_embedder.model_name == "test-model"


@pytest.fixture
def feature_engineering_config(tmp_path):
    # Simulated config setup using tmp_path for artifacts
    fe_dir = tmp_path / "artifacts" / "feature_engineering"
    fe_dir.mkdir(parents=True, exist_ok=True)
    return FeatureEngineeringConfig(
        root_dir=fe_dir,
        input_data_path=fe_dir / "dummy_path.csv",  # Fixed to use tmp_path
        train_data_path=fe_dir / "train.csv",
        test_data_path=fe_dir / "test.csv",
        val_data_path=fe_dir / "val.csv",
        structured_preprocessor_path=fe_dir / "structured_preprocessor.pkl",
        nlp_preprocessor_path=fe_dir / "nlp_preprocessor.pkl",
        embedding_model_name="test-model",
        pca_components=2,  # Tiny PCA
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        target_column="Churn",
    )


class TestFeatureEngineeringComponent:
    """Suite for integrated testing of the FeatureEngineering component execution."""

    def test_data_splitting_and_processing(
        self, feature_engineering_config, mock_sentence_transformer, sample_telco_df
    ):
        """Test that initiate_feature_engineering accurately splits its synthetic data."""

        # Use the centralized sample_telco_df
        data = sample_telco_df

        # Write dummy data to the temporary input file
        data.to_csv(feature_engineering_config.input_data_path, index=False)

        feature_engine = FeatureEngineering(config=feature_engineering_config)
        feature_engine.initiate_feature_engineering()

        # Check files were created
        assert feature_engineering_config.train_data_path.exists()
        assert feature_engineering_config.test_data_path.exists()
        assert feature_engineering_config.val_data_path.exists()
        assert feature_engineering_config.structured_preprocessor_path.exists()
        assert feature_engineering_config.nlp_preprocessor_path.exists()

        # Check file dimensions / splits
        train_df = pd.read_csv(feature_engineering_config.train_data_path)
        test_df = pd.read_csv(feature_engineering_config.test_data_path)
        val_df = pd.read_csv(feature_engineering_config.val_data_path)

        # 20 samples: 20*0.2 test = 4, 16 remaining;
        # 16*0.1 val = 1.6 (~2), 14 train approx, depending on stratification
        assert len(train_df) + len(test_df) + len(val_df) == 20
        # verify identifiers passed through
        assert "customerID" in train_df.columns
        assert "Churn" in train_df.columns
