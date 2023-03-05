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
        # numpy
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

    if name is None:
        # The name attribute could be None, at least for pandas.Series.
        name = default

    return name
