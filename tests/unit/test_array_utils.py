"""
Unit tests for the centralized array handling utilities.

This module validates the ensure_ndarray utility, ensuring robust conversion
of various data formats (Numpy, Pandas, Scipy Sparse) into dense arrays.
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from src.utils.array_utils import ensure_ndarray


def test_ensure_ndarray_from_numpy():
    """Test conversion from an existing numpy array."""
    data = np.array([1, 2, 3])
    result = ensure_ndarray(data)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, data)


def test_ensure_ndarray_from_list():
    """Test conversion from a Python list."""
    data = [1, 2, 3]
    result = ensure_ndarray(data)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array(data))


def test_ensure_ndarray_from_sparse():
    """Test conversion from a Scipy sparse matrix."""
    data = csr_matrix([[1, 0], [0, 2]])
    result = ensure_ndarray(data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result[0, 0] == 1
    assert result[1, 1] == 2


def test_ensure_ndarray_from_dataframe():
    """Test conversion from a Pandas DataFrame."""
    data = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = ensure_ndarray(data)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert result[0, 0] == 1


def test_ensure_ndarray_from_series():
    """Test conversion from a Pandas Series."""
    data = pd.Series([1, 2, 3])
    result = ensure_ndarray(data)
    assert isinstance(result, np.ndarray)
    np.testing.assert_array_equal(result, np.array([1, 2, 3]))
