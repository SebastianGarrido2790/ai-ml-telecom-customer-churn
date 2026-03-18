# Phase 4: NLP Feature Engineering & Feature Store Integration Architecture

## 1. Executive Summary

This document details the architecture and implementation of Phase 4 of the Telecom Customer Churn Prediction project. This phase focuses on the **Mechanic Layer** of the FTI (Feature, Training, Inference) pattern. It transforms validated, raw data (including newly enriched NLP text) into a machine-learning-ready feature matrix that adheres strictly to the Anti-Skew Mandate.

## 2. The Anti-Skew Mandate & The Scikit-Learn Pipeline

A core MLOps principle enforced in this architecture is the prevention of Training-Serving Skew. Features computed during training must be mathematically identical to features computed during inference.

To achieve this, the entire feature transformation process is encapsulated within a single `sklearn.compose.ColumnTransformer`.

### 2.1 The Unified Preprocessor

The preprocessor is responsible for routing different data types through specific transformation pipelines:

1.  **Numerical Pipeline:**
    *   `NumericCleaner`: A custom Scikit-Learn transformer that coerces string representations of numbers (e.g., in the `TotalCharges` column) into floats and handles empty spaces by converting them to `np.nan`.
    *   `SimpleImputer`: Fills remaining `np.nan` values using the median.
    *   `StandardScaler`: Scales numerical features to have zero mean and unit variance.

2.  **Categorical Pipeline:**
    *   `SimpleImputer`: Fills missing categorical values with the most frequent value.
    *   `OneHotEncoder`: Converts categorical strings into one-hot encoded vectors, dropping the first category to avoid multicollinearity (`drop='first'`). It strictly ignores unknown categories during inference.

3.  **NLP Pipeline (Text Notes):**
    *   `TextEmbedder`: A custom transformer that wraps the `sentence-transformers/all-MiniLM-L6-v2` model. It converts the generated `ticket_note` features into 384-dimensional dense vectors. It uses lazy loading to ensure the model is only downloaded/loaded into memory when explicitly called, preventing massive memory spikes during environment initialization.
    *   `PCA` (Principal Component Analysis): Reduces the 384-dimensional embeddings down to 10 principal components. This step is crucial for reducing the feature space, improving training speed, and preventing the tree-based models (XGBoost/LightGBM) from overfitting on sparse, high-dimensional textual data.

## 3. Custom Scikit-Learn Transformers

To integrate domain-specific logic seamlessly into the `ColumnTransformer`, we implemented custom transformers that extend `sklearn.base.BaseEstimator` and `sklearn.base.TransformerMixin`:

### 3.1 `NumericCleaner` 
- **Purpose:** Specifically designed to target columns like `TotalCharges` that arrive as objects (strings) but are semantically numeric.
- **Logic:** Applies `pd.to_numeric(..., errors='coerce')` within the `transform` step.
- **Benefits:** Ensures that subsequent imputers and scalers receive clean, typed numerical arrays.

### 3.2 `TextEmbedder`
- **Purpose:** The bridge between Deep Learning NLP (SentenceTransformers) and classical ML (Scikit-Learn). 
- **Lazy Loading Implementation:** The underlying PyTorch model is initialized dynamically within a `@property` decorator.
- **Serialization Safety:** PyTorch models cannot be easily pickled using standard joblib. The `TextEmbedder` overrides `__getstate__` and `__setstate__` to drop the `.model` reference before pickling. When loaded during the Inference Phase, it automatically re-initializes itself from the local HuggingFace cache.

## 4. Artifact Generation & Data Splitting

The `FeatureEngineering` component performs the following orchestrated steps:

1.  **Ingestion:** Reads the output of Phase 3 (`enriched_churn_data_validated.csv`).
2.  **Splitting:** Performs a stratified split based on the target variable (`Churn`) to guarantee consistent class distributions across `train`, `val`, and `test` sets. The ratios are configurable in `config/params.yaml` (default: 80% train, 10% validation, 10% test).
3.  **Fitting (Anti-Skew):** The unified `ColumnTransformer` is **`fit()` exclusively on the `train` dataset**.
4.  **Transformation:** The fitted preprocessor `transform()`s the `train`, `val`, and `test` sets identically.
5.  **Concatenation:** Crucially, the target column (`Churn`) and identifiers (`customerID`) are separated before transformation to prevent data leakage and are concatenated back to the transformed feature matrix, ensuring row indices align properly.
6.  **Serialization:**
    *   The transformed DataFrames are saved as `train.csv`, `val.csv`, and `test.csv` in `artifacts/feature_engineering/`.
    *   The **fitted** `ColumnTransformer` is serialized using `joblib` and saved as `preprocessor.pkl`. This is the artifact that will be loaded by the Inference API (The Brawn) in Phase 6.

## 5. DVC Integration

The entire Phase 4 pipeline is tracked via `dvc.yaml`:

*   **Dependencies:** `stage_04_feature_engineering.py`, `src/components/feature_engineering.py`, `src/utils/feature_utils.py`, `config/config.yaml`, `config/params.yaml`, and the input valid artifact (`artifacts/data_enrichment/enriched_churn_data_validated.csv`).
*   **Outputs:** `train.csv`, `val.csv`, `test.csv`, and `preprocessor.pkl`.

This ensures that any change in configuration, code logic, or upstream data automatically invalidates the feature engineering cache, triggering a reproducible pipeline run.
