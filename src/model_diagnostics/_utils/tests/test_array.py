import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_array_equal

from model_diagnostics._utils.array import (
    array_name,
    get_array_min_max,
    get_second_dimension,
    get_sorted_array_names,
    is_pyarrow_table,
    length_of_first_dimension,
    length_of_second_dimension,
    safe_index_rows,
    validate_2_arrays,
    validate_same_first_dimension,
)
from model_diagnostics._utils.test_helper import (
    SkipContainer,
    pa_array,
    pa_float64,
    pa_table,
    pd_DataFrame,
    pd_Series,
)


@pytest.mark.parametrize(
    ("a", "n"),
    [
        (list(range(5)), 5),
        (np.arange(5), 5),
        (np.ones((5, 3)), 5),
        (pa_array(range(5)), 5),
        (pa_table({"a": [0, 1], "b": ["A", "B"]}), 2),
        (pd_Series(range(5)), 5),
        (pd_DataFrame({"a": [0, 1], "b": 0.5}), 2),
        (pl.Series(range(5)), 5),
        (pl.DataFrame({"a": [0, 1], "b": 0.5}), 2),
    ],
)
def test_length_of_first_dimension(a, n):
    """Test length of first dimenion."""
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")
    assert length_of_first_dimension(a) == n


@pytest.mark.parametrize(
    ("a", "msg"),
    [
        (np.array(2), "Array-like object has zero length first dimension."),
        (2, "Unable to determine array-like object's length of first dimension."),
    ],
)
def test_length_of_first_dimension_raises(a, msg):
    """Test that test_length_of_first_dimension raises error."""
    with pytest.raises(ValueError, match=msg):
        length_of_first_dimension(a)


@pytest.mark.parametrize(
    ("a", "n"),
    [
        (list(range(5)), 0),
        (([2, 3, 4], [1, 2, 3]), 3),
        (np.arange(5), 0),
        (np.ones((5, 3)), 3),
        (pa_array(range(5)), 0),
        (pa_table({"a": [0, 1], "b": ["A", "B"]}), 2),
        (pd_Series(range(5)), 0),
        (pd_DataFrame({"a": [0, 1], "b": 0.5}), 2),
        (pl.Series(range(5)), 0),
        (pl.DataFrame({"a": [0, 1], "b": 0.5}), 2),
    ],
)
def test_length_of_second_dimension(a, n):
    """Test length of second dimension."""
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")
    assert length_of_second_dimension(a) == n


@pytest.mark.parametrize(
    ("a", "msg"),
    [
        (np.ones((2, 2, 2)), "Array-like has more than 2 dimensions."),
        ([[[0], [1]], [[0], [1]]], "Array-like has more than 2 dimensions."),
    ],
)
def test_length_of_second_dimension_raises(a, msg):
    """Test that test_length_of_second_dimension raises error."""
    with pytest.raises(ValueError, match=msg):
        length_of_second_dimension(a)


@pytest.mark.parametrize(
    ("a", "i", "result"),
    [
        ([[1, 2], [3, 4]], 1, [2, 4]),
        (np.ones((5, 3)), 2, np.ones(5)),
        (pa_table({"a": [0, 1], "b": ["A", "B"]}), 1, pa_array(["A", "B"])),
        (pd_DataFrame({"a": [0, 1], "b": 0.5}), 0, pd_Series([0, 1], name="a")),
        (pl.DataFrame({"a": [0, 1], "b": 0.5}), 0, pl.Series(values=[0, 1], name="a")),
    ],
)
def test_get_second_dimension(a, i, result):
    """Test that get_second_dimension works correctly."""
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")
    np.testing.assert_array_equal(get_second_dimension(a, i), result)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (list(range(5)), np.zeros(5)),
        (np.zeros(5), np.zeros((5, 15))),
        (pa_array(range(5)), np.zeros(5)),
        # (pa_chunked_array([["A", "B"]]), np.zeros(2)),
        # Note that the same pyarrow ChunkedArray is obtained with
        (pa_table({"a": [0, 1], "b": ["A", "B"]}).column("b"), np.zeros(2)),
        (pd_Series(range(5)), np.zeros(5)),
        (pd_DataFrame({"a": [0, 1], "b": 0.5})["b"], np.zeros(2)),
        (pl.Series(range(5)), np.zeros(5)),
        (pl.DataFrame({"a": [0, 1], "b": 0.5})["b"], np.zeros(2)),
    ],
)
def test_validate_same_first_dimension_passes(a, b):
    """Test that validate_same_first_dimension succeeds."""
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")
    assert validate_same_first_dimension(a, b)


@pytest.mark.parametrize(
    ("a", "b", "msg"),
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
            pd_Series(range(5)),
            np.zeros(3),
            "The two array-like objects don't have the same length",
        ),
        (
            pa_array(range(5)),
            np.zeros(3),
            "The two array-like objects don't have the same length",
        ),
        (
            pd_DataFrame({"a": [0, 1], "b": 0.5})["b"],
            np.zeros(3),
            "The two array-like objects don't have the same length",
        ),
        (
            pa_table({"a": [0, 1], "b": ["A", "B"]}).column("b"),
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
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")
    with pytest.raises(ValueError, match=msg):
        validate_same_first_dimension(a, b)


@pytest.mark.parametrize(
    "a",
    [
        list(range(5)),
        pd_Series(range(5), dtype=float),
        pa_array(range(5), type=pa_float64()),
    ],
)
def test_validate_2_arrays(a):
    """Test expected behavior of validate_2_arrays."""
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")
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
    ("a", "name"),
    [
        (np.array([1]), ""),
        (pl.Series(values=[1], name="Numbers"), "Numbers"),
        # pa_array gives no control to set arra._name
        (pa_table({"t": [1.0]}).column("t"), "t"),
        (pd_Series([1], name="Numbers"), "Numbers"),
    ],
)
def test_array_name(a, name):
    """Test that array_name gives correct name."""
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")
    assert array_name(a) == name


@pytest.mark.parametrize(
    "a",
    [
        pl.Series(values=[1, 2], name=None),
        pd_Series([1, 2]),  # We dont set name in Series => it is None.
    ],
)
def test_array_name_none(a):
    """Test that array_name gives default if name is None."""
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")
    assert array_name(a, default="default") == "default"
    assert array_name(None, default="default")


def test_array_name_empty_string():
    """Test that array_name gives default if name is empty string."""
    a = pl.Series(name="", values=[1, 2])
    assert array_name(a, default="default") == "default"
    assert not array_name(a, default="")  # test for empty string ""


@pytest.mark.parametrize(
    "a",
    [
        (0, 1, 99),
        [0, 1, 99],
        [[0, 1], [2, 99]],
        np.array([0, 1, 99]),
        np.array([[0, 1], [2, 99]]),
        pa_array([0, 1, 99]),
        pa_table({"0": [0, 1], "1": [2, 99]}),
        pd_Series([0, 1, 99]),
        pd_DataFrame({"0": [0, 1], "1": [2, 99]}),
        pl.Series([0, 1, 99]),
        pl.DataFrame({"0": [0, 1], "1": [2, 99]}),
    ],
)
def test_get_array_min_max(a):
    """Test that get_array_min_max does its job"""
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")
    a_min, a_max = get_array_min_max(a)
    assert (a_min, a_max) == (0, 99)


def test_get_sorted_array_names():
    """Test that get_sorted_array_names does its job."""
    y_pred = [1]
    names, indices = get_sorted_array_names(y_pred)
    assert names == [""]
    assert indices == [0]

    y_pred = [[1, 10], [2, 20]]
    names, indices = get_sorted_array_names(y_pred)
    assert names == ["0", "1"]
    assert indices == [0, 1]

    y_pred = pl.Series(values=[0, 1], name="model_1")
    names, indices = get_sorted_array_names(y_pred)
    assert names == ["model_1"]
    assert indices == [0]

    y_pred = pl.DataFrame(
        {
            "model_2": [0, 1],
            "model_3": [1, 1],
            "model_1": [2, 1],
            "": [2, 2],
        }
    )
    names, indices = get_sorted_array_names(y_pred)
    assert names == ["model_2", "model_3", "model_1", "3"]
    assert indices == [3, 2, 0, 1]


def test_safe_index_rows_raises():
    """Test that safe_index_rows raises errors."""
    msg = "Only integer indices are allowed for indexing rows."
    with pytest.raises(ValueError, match=msg):
        safe_index_rows(np.arange(2), "asd")


@pytest.mark.parametrize(
    "a",
    [
        (-99, 1, 22),
        [-99, 1, 22],
        [[-99, 0], [1, 1], [22, 2]],
        np.array([-99, 1, 22]),
        np.array([[-99, 0], [1, 1], [22, 2]]),
        pa_array([-99, 1, 22]),
        pa_table({"0": [-99, 1, 22], "1": [0, 1, 2]}),
        pd_Series([-99, 1, 22]),
        pd_DataFrame({"0": [-99, 1, 22], "1": [0, 1, 2]}),
        pl.Series([-99, 1, 22]),
        pl.DataFrame({"0": [-99, 1, 22], "1": [0, 1, 2]}),
    ],
)
def test_safe_index_rows(a):
    """Test that safe_index_rows does its job."""
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")

    a_sub = safe_index_rows(a, [0, 2])

    if is_pyarrow_table(a):
        # Because pyarrow has not to_numpy() and older versions of pyarrow
        # give the transposed with np.asarray(a)
        a = a.to_pandas()
        a_sub = a_sub.to_pandas()
    a = np.asarray(a)
    a_sub = np.asarray(a_sub)

    if length_of_second_dimension(a) >= 1:
        assert_array_equal(a_sub, np.array([[-99, 0], [22, 2]]))
    else:
        assert_array_equal(a_sub, np.array([-99, 22]))
