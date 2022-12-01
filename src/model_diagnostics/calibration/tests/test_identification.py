from functools import partial

import numpy as np
import pyarrow as pa
import pytest
from scipy.optimize import root_scalar
from scipy.stats import ttest_1samp

from model_diagnostics.calibration import compute_bias, identification_function


# Note: expectile might arrive with scipy 1.10.
# Backport of scipy.stats.expectile
# https://github.com/scipy/scipy/pull/17039
def expectile(a, alpha=0.5, *, dtype=None, weights=None):  # pragma: no cover
    r"""Compute the expectile along the specified axis."""
    if alpha < 0 or alpha > 1:
        raise ValueError("The expectile level alpha must be in the range [0, 1].")
    a = np.asarray(a, dtype=dtype)

    if weights is not None:
        weights = np.asarray(weights, dtype=dtype)

    # This is the empirical equivalent of Eq. (13) with identification
    # function from Table 9 (omitting a factor of 2) in
    # Gneiting, T. (2009). "Making and Evaluating Point Forecasts".
    # Journal of the American Statistical Association, 106, 746 - 762.
    # https://doi.org/10.48550/arXiv.0912.0902
    def first_order(t):
        return alpha * np.average(np.fmax(0, a - t), weights=weights) - (
            1 - alpha
        ) * np.average(np.fmax(0, t - a), weights=weights)

    if alpha >= 0.5:
        x0 = np.average(a, weights=weights)
        x1 = np.amax(a)
    else:
        x1 = np.average(a, weights=weights)
        x0 = np.amin(a)

    if x0 == x1:
        # a has a single unique element
        return x0

    # Note that the expectile is the unique solution, so no worries about
    # finding a wrong root.
    res = root_scalar(first_order, x0=x0, x1=x1)
    return res.root


@pytest.mark.parametrize(
    "functional, est",
    [
        ("mean", np.mean),
        ("median", np.median),
        ("expectile", expectile),
        ("quantile", np.quantile),
    ],
)
@pytest.mark.parametrize("level", [0.01, 0.4, 0.4999, 0.5, 0.50001, 0.8, 0.999])
def test_identification_function_equal_input(functional, est, level):
    rng = np.random.default_rng(112358132134)
    n = 1000
    if functional == "quantile":
        est = partial(est, q=level)
    elif functional == "expectile":
        est = partial(est, alpha=level)
    y_obs = rng.normal(loc=-2, scale=2, size=n) + rng.normal(loc=1, scale=0.1, size=n)
    y_pred = np.ones_like(y_obs) * est(y_obs)
    id_func = identification_function(y_obs, y_pred, functional=functional, level=level)
    assert np.mean(id_func) == pytest.approx(0, abs=1e-1 / n)


@pytest.mark.parametrize(
    "functional, level, msg",
    [
        ("no good functional", 0.5, "Argument functional must be one of"),
        ("quantile", 1.1, "Argument level must fulfil 0 <= level <= 1"),
    ],
)
def test_identification_function_raises(functional, level, msg):
    y_obs, y_pred = np.arange(5), np.arange(5)
    with pytest.raises(ValueError, match=msg):
        identification_function(y_obs, y_pred, functional=functional, level=level)


@pytest.mark.parametrize(
    "feature, f_result",
    [
        (
            pa.DictionaryArray.from_arrays([0, 0, 1, 1], ["b", "a"]),
            pa.DictionaryArray.from_arrays([0, 1], ["b", "a"]),
        ),
        (
            pa.array([0.1, 0.1, 0.9, 0.9]),
            pa.array([0.1, 0.9]),
        ),
    ],
)
def test_compute_bias(feature, f_result):
    """Test compute_bias on simple data."""
    df = pa.table(
        {
            "y_obs": [0, 1, 2, 4],
            "y_pred": [1, 1, 2, 2],
            "feature": feature,
        }
    )
    df_bias = compute_bias(
        y_obs=df.column("y_obs"),
        y_pred=df.column("y_pred"),
        feature=df.column("feature"),
    )
    df_expected = pa.table(
        {
            "feature": f_result,
            "bias_mean": [0.5, -1],
            "bias_count": [2, 2],
            "bias_stddev": np.sqrt([0.25 + 0.25, 1 + 1]),
            "p_value": [ttest_1samp([1, 0], 0).pvalue, ttest_1samp([0, -2], 0).pvalue],
        }
    )
    print(f"{df_bias=}")
    print(f"{df_expected=}")
    assert df_bias.equals(df_expected)
