"""
Array utilities for NumPy and SciPy transformations.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np


def ensure_ndarray(transformed: Any) -> np.ndarray:
    """Ensures that the input is converted to a NumPy ndarray.

    Handles outputs from scikit-learn transformers which may return
    sparse matrices (with .toarray()), pandas DataFrames (with .to_numpy()
    or .values), or standard arrays.

    Args:
        transformed: The object to convert (sparse matrix, DataFrame, Series, or array).

    Returns:
        NumPy ndarray.
    """
    if hasattr(transformed, "toarray"):
        # Handle SciPy sparse matrices
        return cast(Any, transformed).toarray()
    if hasattr(transformed, "to_numpy"):
        # Handle modern Pandas DataFrames/Series
        return cast(Any, transformed).to_numpy()
    if hasattr(transformed, "values"):
        # Handle older Pandas DataFrames/Series
        return cast(Any, transformed).values

    # Fallback to standard array conversion
    return np.asarray(transformed)
