import pytest

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pyarrow as pa

from model_diagnostics._utils.array_validation import validate_2_arrays


@pytest.mark.parametrize("a",
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
