from typing import Optional

import numpy as np
import numpy.typing as npt


def length_of_first_dimension(a: npt.ArrayLike) -> int:
    """Return length of first dimension."""
    if hasattr(a, "shape"):
        if len(a.shape) < 1:
            raise ValueError("Array-like object has zero length first dimension.")
        else:
            return a.shape[0]
    elif hasattr(a, "length") and callable(a.length):
        return a.length()
    elif hasattr(a, "__len__"):
        return len(a)  # type: ignore
    else:
        raise ValueError(
            "Unable to determine array-like object's length of first dimension."
        )


def validate_same_first_dimension(a: npt.ArrayLike, b: npt.ArrayLike) -> bool:
    """Validate that 2 array-like have the same length of the first dimension."""
    if length_of_first_dimension(a) != length_of_first_dimension(b):
        raise ValueError(
            "The two array-like objects don't have the same length of their first "
            "dimension."
        )
    else:
        return True


def length_of_second_dimension(a: npt.ArrayLike) -> int:
    """Return length of first dimension."""
    if not hasattr(a, "shape"):
        a = np.asarray(a)

    dim = len(a.shape)
    if dim < 2:
        return 0
    elif dim == 2:
        return a.shape[1]
    else:
        raise ValueError("Array-like has more than 2 dimensions.")


def get_second_dimension(a: npt.ArrayLike, i: int) -> npt.ArrayLike:
    """Get i-th column of a, e.g. a[:, i]."""
    if hasattr(a, "iloc"):
        # pandas
        return a.iloc[:, i]
    elif hasattr(a, "column"):
        # pyarrow
        return a.column(i)  # a[i] would also work
    else:
        # numpy
        return a[:, i]  # type: ignore


def validate_2_arrays(
    a: npt.ArrayLike, b: npt.ArrayLike
) -> tuple[np.ndarray, np.ndarray]:
    """Validate 2 arrays."""
    # Note: If the input is an pyarrow array, np.asarray produces a read-only ndarray.
    a = np.asarray(a)
    b = np.asarray(b)
    if a.ndim != b.ndim:
        raise ValueError(
            f"Arrays must have the same dimension, got {a.ndim=} and {b.ndim=}."
        )
    for i in range(a.ndim):
        if a.shape[i] != b.shape[i]:
            raise ValueError(
                f"Arrays must have the same shape, got {a.shape=} and {b.shape=}."
            )
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
        name = a._name
    else:
        name = default

    if name is None:
        # The name attribute could be None, at least for pandas.Series.
        name = default

    return name
