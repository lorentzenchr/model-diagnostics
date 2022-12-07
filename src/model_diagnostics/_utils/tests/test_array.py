import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numpy.testing import assert_array_equal

from model_diagnostics._utils.array import (
    array_name,
    validate_2_arrays,
    validate_same_first_dimension,
)


@pytest.mark.parametrize(
    "a, b",
    [
        (list(range(5)), np.zeros(5)),
        (np.zeros(5), np.zeros((5, 15))),
        (pd.Series(range(5)), np.zeros(5)),
        (pa.array(range(5)), np.zeros(5)),
        (pd.DataFrame({"a": [0, 1], "b": 0.5})["b"], np.zeros(2)),
        (pa.table({"a": [0, 1], "b": ["A", "B"]}).column("b"), np.zeros(2)),
    ],
)
def test_validate_same_first_dimension_passes(a, b):
    """Test that validate_same_first_dimension succeeds."""
    assert validate_same_first_dimension(a, b)


@pytest.mark.parametrize(
    "a, b, msg",
    [
        (
            list(range(5)),
            np.zeros(3),
            "The two array-like objects don't have the same length",
        ),
        (
            np.zeros(5),
            np.zeros((3, 15)),
            "The two array-like objects don't have the same length",
        ),
        (
            pd.Series(range(5)),
            np.zeros(3),
            "The two array-like objects don't have the same length",
        ),
        (
            pa.array(range(5)),
            np.zeros(3),
            "The two array-like objects don't have the same length",
        ),
        (
            pd.DataFrame({"a": [0, 1], "b": 0.5})["b"],
            np.zeros(3),
            "The two array-like objects don't have the same length",
        ),
        (
            pa.table({"a": [0, 1], "b": ["A", "B"]}).column("b"),
            np.zeros(3),
            "The two array-like objects don't have the same length",
        ),
        (
            5,
            np.zeros(3),
            "Unable to determine array-like object's length of first dimension.",
        ),
        (
            np.array(5),
            np.zeros(3),
            "Array-like object has zero length first dimension.",
        ),
    ],
)
def test_validate_same_first_dimension_raises(a, b, msg):
    """Test that validate_same_first_dimension succeeds."""
    with pytest.raises(ValueError, match=msg):
        validate_same_first_dimension(a, b)


@pytest.mark.parametrize(
    "a",
    [
        list(range(5)),
        pd.Series(range(5), dtype=float),
        pa.array(range(5), type=pa.float64()),
    ],
)
def test_validate_2_arrays(a):
    """Test expected behavior of validate_2_arrays."""
    b = np.arange(5, dtype=np.float64)
    validate_2_arrays(a, b)
    assert_array_equal(a, b)


def test_validate_2_arrays_raises():
    """Test validate_2_arrays raises errors."""
    a = np.arange(5)
    b = np.arange(5)[:, None]
    with pytest.raises(ValueError, match="Arrays must have the same dimension"):
        validate_2_arrays(a, b)

    b = np.arange(4)
    with pytest.raises(ValueError, match="Arrays must have the same shape"):
        validate_2_arrays(a, b)


@pytest.mark.parametrize(
    "a, name",
    [
        (np.array([1]), ""),
        (pd.Series([1], name="Numbers"), "Numbers"),
        # pa.array gives no control to set arra._name
        (pa.table({"t": [1.0]}).column("t"), "t"),
    ],
)
def test_array_name(a, name):
    """Test that array_name gives correct name."""
    assert array_name(a) == name


def test_array_name_none():
    """Test that array_name gives default if name is None."""
    a = pd.Series([1, 2])  # We dont set name is Series => it is None.
    assert array_name(a, default="default") == "default"
