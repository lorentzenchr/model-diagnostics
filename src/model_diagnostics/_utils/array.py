from typing import Optional

import numpy as np
import numpy.typing as npt


def _length_of_first_dimension(a: npt.ArrayLike) -> int:
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
    if _length_of_first_dimension(a) != _length_of_first_dimension(b):
        raise ValueError(
            "The two array-like objects don't have the same length of their first "
            "dimension."
        )
    else:
        return True


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
