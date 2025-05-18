import numpy as np
import polars as pl
import pytest

from polars.testing import assert_frame_equal

from model_diagnostics._utils.test_helper import (
    SkipContainer,
    pa_table,
    pd_DataFrame,
)
from model_diagnostics._utils.array import get_second_dimension
from model_diagnostics.xai import compute_permutation_importance

@pytest.mark.parametrize(
    "X", 
    [
        pd_DataFrame({"a": [10, 5, 0], "b": [0, 1, 2]}),
        pl.DataFrame({"a": [10, 5, 0], "b": [0, 1, 2]}),
        pa_table({"a": [10, 5, 0], "b": [0, 1, 2]}),
    ]
)  
@pytest.mark.parametrize("n_repeats", [1, 2])
@pytest.mark.parametrize("weights", [None, [1, 2, 3]])
def test_permutation_importance_consistent_across_types(X, n_repeats, weights):
    if isinstance(X, SkipContainer):
        pytest.skip("Module for data container not imported.")

    y = np.array([9, 6, 1])

    def predict(X):
        return np.array(get_second_dimension(X, 0))
    
    # Calculate reference result using numpy
    X_numpy = np.array([[10, 0], [5, 1], [0, 2]])
    reference = compute_permutation_importance(
        predict,
         X=X_numpy,
         y=y,
         features={"a": 0, "b": 1},  # specify feature names
         n_repeats=n_repeats,
         weights=weights,
         rng=0,
    )

    result = compute_permutation_importance(
        predict,
        X=X,
        y=y,
        n_repeats=n_repeats,
        weights=weights,
        rng=0,
    )

    assert_frame_equal(result, reference)
       