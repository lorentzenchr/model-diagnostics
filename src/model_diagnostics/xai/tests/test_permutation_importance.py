import numpy as np
import polars as pl
import pytest
from numpy.testing import assert_array_equal
from polars.testing import assert_frame_equal
from sklearn.ensemble import RandomForestRegressor

from model_diagnostics._utils.array import (
    get_second_dimension,
    length_of_first_dimension,
)
from model_diagnostics._utils.test_helper import (
    SkipContainer,
    pa_table,
    pd_DataFrame,
)
from model_diagnostics.scoring import PoissonDeviance, SquaredError
from model_diagnostics.xai import compute_permutation_importance


@pytest.mark.parametrize(
    "X",
    [
        pd_DataFrame({"a": [10, 5, 0], "b": [0, 1, 2]}),
        pl.DataFrame({"a": [10, 5, 0], "b": [0, 1, 2]}),
        pa_table({"a": [10, 5, 0], "b": [0, 1, 2]}),
    ],
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


@pytest.mark.parametrize(
    ("a", "original"),
    [
        (
            np.array([[1, 2, 3, 4], [1, 2, 3, 4]]).T,
            np.array([[1, 2, 3, 4], [1, 2, 3, 4]]).T,
        ),
        (
            pa_table({"a": [0, 1, 2, 3], "b": [1, 1, 2, 2]}),
            pa_table({"a": [0, 1, 2, 3], "b": [1, 1, 2, 2]}),
        ),
        (
            pd_DataFrame({"a": [0, 1, 2, 3], "b": [1, 1, 2, 2]}),
            pd_DataFrame({"a": [0, 1, 2, 3], "b": [1, 1, 2, 2]}),
        ),
        (
            pl.DataFrame({"a": [0, 1, 2, 3], "b": [1, 1, 2, 2]}),
            pl.DataFrame({"a": [0, 1, 2, 3], "b": [1, 1, 2, 2]}),
        ),
    ],
)
@pytest.mark.parametrize("n_max", [3, None])
@pytest.mark.parametrize("n_repeat", [1, 2])
@pytest.mark.parametrize("weights", [None, [1.0, 1.0, 2.0, 2.0]])
def test_no_side_effects(a, original, n_max, n_repeat, weights):
    """Test that calculate_permutation_importance() does not modify input.

    For simplicity, we only use numerical data.
    """
    if isinstance(a, SkipContainer):
        pytest.skip("Module for data container not imported.")

    y = np.arange(length_of_first_dimension(a))
    rf = RandomForestRegressor(n_estimators=10, random_state=0)
    rf.fit(a, y)
    _ = compute_permutation_importance(
        rf.predict,
        X=a,
        y=y,
        weights=weights,
        n_repeats=n_repeat,
        n_max=n_max,
        rng=0,
    )
    assert_array_equal(a, original)


@pytest.mark.parametrize(
    "scorer",
    [PoissonDeviance(), SquaredError()],
)
@pytest.mark.parametrize("n_repeats", [1, 2])
@pytest.mark.parametrize("n_max", [None, 8])
@pytest.mark.parametrize("weights", [None, [1.0] * 10])
@pytest.mark.parametrize("method", ["difference", "ratio"])
def test_permutation_importance_finds_important_feature(
    scorer, n_repeats, n_max, weights, method
):
    X = pl.DataFrame(
        {
            "a": np.array([0, 1] * 5),
            "b": np.linspace(0.1, 0.9, num=10),  # important feature
            "c": np.zeros(10),
        }
    )

    y = X["a"] + X["b"] + X["c"]

    def predict(x):
        return x["b"]

    result = compute_permutation_importance(
        predict,
        X=X,
        y=y,
        weights=weights,
        scoring_function=scorer,
        n_repeats=n_repeats,
        n_max=n_max,
        method=method,
        rng=0,
    )
    assert result["feature"][0] == "b"
    assert result["importance"][0] > 0.0 + (method == "ratio")
    assert result["importance"][1] == pytest.approx(0.0 + (method == "ratio"))


def test_compute_permutation_importance_raises_errors():
    X = pl.DataFrame(
        {
            "a": np.array([0, 1] * 5),
            "b": np.linspace(0.1, 0.9, num=10),  # important feature
            "c": np.zeros(10),
        }
    )

    y = X["a"] + X["b"] + X["c"]

    def predict(x):
        return x["b"]

    # n_repeats
    msg = "Argument n_repeats must be >= 1, got 0"
    with pytest.raises(ValueError, match=msg):
        compute_permutation_importance(predict, X=X, y=y, n_repeats=0)

    # method
    msg = "Unknown normalization method: invalid_method"
    with pytest.raises(ValueError, match=msg):
        compute_permutation_importance(predict, X=X, y=y, method="invalid_method")

    # scoring_direction
    msg = "Argument scoring_direction must be 'smaller' or 'greater', got .*"
    with pytest.raises(ValueError, match=msg):
        compute_permutation_importance(predict, X=X, y=y, scoring_direction="larger")
