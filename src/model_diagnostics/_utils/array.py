import copy
import sys
import warnings
from importlib.metadata import version
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import polars as pl
from packaging.version import Version, parse

AL_or_polars = Union[npt.ArrayLike, pl.Series]


def length_of_first_dimension(a: npt.ArrayLike) -> int:
    """Return length of first dimension."""
    if hasattr(a, "shape"):
        if len(a.shape) < 1:
            msg = "Array-like object has zero length first dimension."
            raise ValueError(msg)
        else:
            return a.shape[0]
    elif hasattr(a, "length") and callable(a.length):
        return a.length()
    elif hasattr(a, "__len__"):
        return len(a)  # type: ignore
    else:
        msg = "Unable to determine array-like object's length of first dimension."
        raise ValueError(msg)


def length_of_second_dimension(a: npt.ArrayLike) -> int:
    """Return length of second dimension."""
    if not hasattr(a, "shape"):
        a = np.asarray(a)

    dim = len(a.shape)
    if dim < 2:
        return 0
    elif dim == 2:
        return a.shape[1]
    else:
        msg = "Array-like has more than 2 dimensions."
        raise ValueError(msg)


def get_second_dimension(a: npt.ArrayLike, i: int) -> npt.ArrayLike:
    """Get i-th column of a, e.g. a[:, i]."""
    if hasattr(a, "iloc"):
        # pandas
        return a.iloc[:, i]
    elif hasattr(a, "column") and callable(a.column):
        # pyarrow
        return a.column(i)  # a[i] would also work
    elif isinstance(a, (list, tuple)):
        return np.array([row[i] for row in a])
    else:
        # numpy or polars
        return a[:, i]  # type: ignore


def validate_same_first_dimension(a: AL_or_polars, b: AL_or_polars) -> bool:
    """Validate that 2 array-like have the same length of the first dimension."""
    if length_of_first_dimension(a) != length_of_first_dimension(b):
        msg = (
            "The two array-like objects don't have the same length of their first "
            "dimension."
        )
        raise ValueError(msg)
    else:
        return True


def validate_2_arrays(
    a: npt.ArrayLike, b: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Validate 2 arrays.

    Both arrays are checked to have same dimensions and shapes.
    They are returned as numpy arrays.

    Returns
    -------
    a : ndarray
        Input as an ndarray
    b : ndarray
        Input as an ndarray
    """
    # Note: If the input is a pyarrow array, np.asarray produces a read-only ndarray.
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != b.ndim:
        msg = f"Arrays must have the same dimension, got {a.ndim=} and {b.ndim=}."
        raise ValueError(msg)
    for i in range(a.ndim):
        if a.shape[i] != b.shape[i]:
            msg = f"Arrays must have the same shape, got {a.shape=} and {b.shape=}."
            raise ValueError(msg)
    return a, b


def array_name(a: Optional[npt.ArrayLike], default: str = "") -> str:
    """Extract name from array if it exists."""
    if a is None:
        name = default
    elif hasattr(a, "name"):
        # pandas and polars Series
        name = a.name
    elif hasattr(a, "_name"):
        # pyarrow Array / ChunkedArray
        name = a._name  # noqa: SLF001
    else:
        name = default

    if name is None or not name:  # not name is same as name == ""
        # The name attribute could be None, at least for pandas.Series, or "".
        name = default

    return name


def get_array_min_max(a: npt.ArrayLike):
    """Get min and max over all elements of ArrayLike.

    Returns
    -------
    a_min :
        The minimum value of a.
    a_max :
        The maximum value of a.
    """
    if hasattr(a, "max") and hasattr(a, "min"):
        a_min, a_max = a.min(), a.max()
        if hasattr(a_min, "to_numpy"):
            # Polars and pandas dataframes return min/max per column and have different
            # semantics of a second call a_min.min() wrt the axis argument. Therefor we
            # simply convert to numpy.
            a_min, a_max = a_min.to_numpy().min(), a_max.to_numpy().max()
    else:
        a_min, a_max = np.min(a), np.max(a)
    return a_min, a_max


def get_sorted_array_names(y_pred: Union[npt.ArrayLike, pl.Series, pl.DataFrame]):
    """Get names of an array and sorted indices.

    Returns
    -------
    pred_names : list
        The (column) names of the predictions.
    sorted_indices : list
        A list of indices such that `[pred_names[i] for i in sorted_indices]`
        is a sorted list.
    """
    n_pred = length_of_second_dimension(y_pred)
    if n_pred == 0:
        pred_names = [array_name(y_pred, default="")]
    else:
        pred_names = []
        for i in range(n_pred):
            x = get_second_dimension(y_pred, i)
            pred_names.append(array_name(x, default=str(i)))

    if n_pred >= 2:
        # https://stackoverflow.com/questions/6422700
        sorted_indices = sorted(range(len(pred_names)), key=pred_names.__getitem__)
    else:
        sorted_indices = [0]

    return pred_names, sorted_indices


def is_pandas_series(x):
    """Return True if the x is a pandas Series."""
    try:
        pd = sys.modules["pandas"]
    except KeyError:
        return False
    return isinstance(x, pd.Series)


def is_pandas_df(x):
    """Return True if the x is a pandas DataFrame."""
    try:
        pd = sys.modules["pandas"]
    except KeyError:
        return False
    return isinstance(x, pd.DataFrame)


def is_pyarrow_array(x):
    """Return True if the x is a pyarrow Array or ChunkedArray."""
    try:
        pa = sys.modules["pyarrow"]
    except KeyError:
        return False
    return isinstance(x, (pa.Array, pa.ChunkedArray))


def is_pyarrow_table(x):
    """Return True if the x is a pyarrow Table or RecordBatch."""
    try:
        pa = sys.modules["pyarrow"]
    except KeyError:
        return False
    return isinstance(x, (pa.Table, pa.RecordBatch))


def safe_assign_column(x, values, column_index):
    """Safely assign values array to a column of an array_like x.

    Parameters
    ----------
    x : array-like
        Array to be modified. It is expected to be 2-dimensional.

    values : ndarray
        The values to be assigned to `x`.

    column_index : int
        Index of the column / second dimension.

    Returns
    -------
    x : Modified `x` with the new assign column.
    """
    if isinstance(x, list):
        # Multiple rows may point to the same underlying object, e.g. a result of
        # repeated indices like safe_index_rows(x, [0, 0, 1, 1]). Therefore, we must
        # be careful, i.e. (shallow) copy the rows.
        if hasattr(x[0], "copy"):

            def copy_element(x):
                return x.copy()
        elif hasattr(x[0], "clone"):

            def copy_element(x):
                return x.clone()
        else:

            def copy_element(x):
                return copy.copy(x)

        try:
            row = copy_element(x[0])
            row[column_index] = values[0]
        except Exception as e:
            e.add_note("Unable to set item in safe_assign_column of a list object.")
            raise
        if row[column_index] != values[0]:
            msg = "Elements of the list can't be assigned new vlues."
            raise ValueError(msg)

        for i in range(len(x)):
            row = copy_element(x[i])
            row[column_index] = values[i]
            x[i] = row
    elif is_pandas_df(x):
        try:
            # Avoid deprecation warning of pandas by handling dtype explicitly.
            #   Setting an item of incompatible dtype is deprecated and will raise in a
            #   future error of pandas.
            # Also, assigning with a different index makes troubles.
            pd = sys.modules["pandas"]
            dtype = x.dtypes.iloc[column_index]
            if isinstance(values, pl.Series) and isinstance(
                values.dtype, pl.Categorical
            ):
                # FIXME: pyarrow not installed
                pd_values = pd.Series(
                    data=values.cast(pl.Utf8).to_numpy(),
                    dtype=dtype,
                )
            else:
                pd_values = pd.Series(
                    data=values.to_pandas()
                    if isinstance(values, pl.Series)
                    else values,
                    dtype=dtype,
                )
            if parse(version("pandas")) < Version("2.0.0"):
                # FIXME: pandas >= 2.0 (<2.0 means 1.5.*)
                with warnings.catch_warnings():
                    msg = (
                        r"In a future version, `df.iloc\[:, i\] = newvals` will "
                        r"attempt to set the values inplace instead of always "
                        r"setting a new array. To retain the old behavior, use either "
                        r"`df\[df.columns\[i\]\] = newvals` or, if columns are "
                        r"non-unique, `df.isetitem\(i, newvals\)`"
                    )
                    warnings.filterwarnings(
                        "ignore", category=DeprecationWarning, message=msg
                    )
                    if not x.index.is_unique:
                        # Pandas might error with:
                        #   cannot reindex on an axis with duplicate labels
                        # Try reindexing ourselves.
                        x = x.reset_index()
                    if not pd_values.index.is_unique:
                        pd_values = pd_values.reset_index()
                    x.iloc[:, column_index] = pd_values
            else:
                x.iloc[:, column_index] = pd_values
        except Exception as e:
            # FIXME: pyarrow version XXX
            # Older pyarrow versions of AttributeError do not have a 'add_note' method.
            args = e.args
            msg = (
                args[0]
                + "\nThe problem might be fixable with newer versions of pandas, polars"
                " or pyarrow."
            )
            raise type(e)(msg, *args[1:]) from e
    elif is_pyarrow_table(x):
        x = x.set_column(column_index, x.column_names[column_index], [values])
    elif isinstance(x, pl.DataFrame):
        cname = x.columns[column_index]
        dtype = x.get_column(cname).dtype
        x = x.with_columns(pl.Series(values, dtype=dtype).alias(cname))
    else:  # numpy array or other array-like
        x[:, column_index] = values
    return x


def safe_index_rows(x, indices):
    """Safely index rows (first dimention) of an array-like x.

    Parameters
    ----------
    x : array-like
        Array-like to be indexed on its first dimension.
    indices : array-like of integers

    Returns
    -------
    subset
        Subset of x on first dimension. This may be a view.
    """
    index = np.asarray(indices)
    if index.dtype.kind not in ("i", "u"):
        msg = "Only integer indices are allowed for indexing rows."
        raise ValueError(msg)

    if is_pyarrow_table(x) or is_pyarrow_array(x):
        return x.take(indices)
    elif hasattr(x, "iloc"):
        # using take() instead of iloc[] ensures the return value is a "proper"
        # copy that will not raise SettingWithCopyWarning
        return x.take(indices, axis=0)
    elif isinstance(x, (list, tuple)):
        return [x[idx] for idx in indices]
    else:
        # numpy, polars
        return x[indices]
