from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import polars as pl

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
    elif hasattr(a, "column"):
        # pyarrow
        return a.column(i)  # a[i] would also work
    elif isinstance(a, (list, tuple)):
        return np.asarray(a)[:, i]
    else:
        # numpy or polars
        return a[:, i]  # type: ignore


def validate_2_arrays(
    a: npt.ArrayLike, b: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Validate 2 arrays.

    Both arrays are checked to have same dimensions and shapes and returned as numpy
    arrays.

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
