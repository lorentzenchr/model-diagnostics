from functools import partial

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest
from polars.testing import assert_frame_equal
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
    "feature, f_grouped",
    [
        (
            pa.DictionaryArray.from_arrays([0, 0, 1, 1, 1], ["b", "a"]),
            pa.DictionaryArray.from_arrays(np.array([0, 1], dtype=np.int8), ["b", "a"]),
        ),
        (
            pa.array(["a", "a", "b", "b", "b"]),
            pa.array(["a", "b"]),
        ),
        (
            pa.array([0.1, 0.1, 0.9, 0.9, 0.9]),
            pa.array([0.1, 0.9]),
        ),
    ],
)
def test_compute_bias(feature, f_grouped):
    """Test compute_bias on simple data."""
    with pl.StringCache():
        # The string cache is needed to avoid the following error message:
        # exceptions.ComputeError: Cannot compare categoricals originating from
        # different sources. Consider setting a global string cache.
        df = pa.table(
            {
                "y_obs": [0, 1, 2, 4, 3],
                "y_pred": [1, 1, 2, 2, 2],
                "feature": feature,
            }
        )
        df_bias = compute_bias(
            y_obs=df.column("y_obs"),
            y_pred=df.column("y_pred"),
            feature=df.column("feature"),
        )
        df_expected = pl.DataFrame(
            {
                "feature": f_grouped,
                "bias_mean": [0.5, -1],
                "bias_count": pl.Series(values=[2, 3], dtype=pl.UInt32),
                "bias_stderr": np.sqrt([(0.25 + 0.25), (1 + 1 + 0) / 2])
                / np.sqrt([2, 3]),
                "p_value": [
                    ttest_1samp([1, 0], 0).pvalue,
                    ttest_1samp([0, -2, -1], 0).pvalue,
                ],
            }
        )
        assert_frame_equal(df_bias, df_expected, check_exact=False)


def test_compute_bias_feature_none():
    """Test compute_bias for feature = None."""
    df = pa.table(
        {
            "y_obs": [0, 1, 2, 4, 3],
            "y_pred": [1, 1, 2, 2, 2],
        }
    )
    df_bias = compute_bias(
        y_obs=df.column("y_obs"),
        y_pred=df.column("y_pred"),
        feature=None,
    )
    df_expected = pl.DataFrame(
        {
            "bias_mean": [-0.4],  # (1 + 0 + 0 - 2 - 1) / 5
            "bias_count": [5],
            "bias_stderr": [np.std([1, 0, 0, -2, -1], ddof=1) / np.sqrt(5)],
            "p_value": [
                ttest_1samp([1, 0, 0, -2, -1], 0).pvalue,
            ],
        }
    )
    assert_frame_equal(df_bias, df_expected)


def test_compute_bias_numerical_feature():
    """Test compute_bias for a numerical feature."""
    n_obs = 100
    n_bins = 10
    n_steps = n_obs // n_bins
    df = pl.DataFrame(
        {
            "y_obs": 2 * np.linspace(-0.5, 0.5, num=n_obs, endpoint=False),
            "y_pred": np.linspace(0, 1, num=n_obs, endpoint=False),
            "feature": np.linspace(0, 1, num=n_obs, endpoint=False),
        }
    )
    df_bias = compute_bias(
        y_obs=df.get_column("y_obs"),
        y_pred=df.get_column("y_pred"),
        feature=df.get_column("feature"),
        n_bins=n_bins,
    )
    bias = (df.get_column("y_pred") - df.get_column("y_obs")).to_numpy()
    df_expected = pl.DataFrame(
        {
            "feature": 0.045 + 0.1 * np.arange(10),
            "bias_mean": 0.955 - 0.1 * np.arange(10),
            "bias_count": n_steps * np.ones(n_bins, dtype=np.uint32),
            "bias_stderr": [
                np.std(bias[n : n + n_steps], ddof=1) / np.sqrt(n_steps)
                for n in range(0, n_obs, n_steps)
            ],
            "p_value": [
                ttest_1samp(bias[n : n + n_steps], 0).pvalue
                for n in range(0, n_obs, n_steps)
            ],
        }
    )
    assert_frame_equal(df_bias, df_expected, check_exact=False)


@pytest.mark.parametrize("feature_type", ["cat", "num", "string"])
def test_compute_bias_multiple_predictions(feature_type):
    """test compute_bias for multiple predictions."""
    with pl.StringCache():
        n_obs = 10
        y_obs = np.ones(n_obs)
        y_obs[: 10 // 2] = 2
        y_pred = pd.DataFrame(
            {"model_1": np.ones(n_obs), "model_2": 3 * np.ones(n_obs)}
        )
        if feature_type == "cat":
            feature = pd.Series(
                y_obs.astype("=U8"), dtype="category", name="nice_feature"
            )
        elif feature_type == "string":
            feature = pd.Series(y_obs.astype("=U8"), name="nice_feature")
        else:
            feature = pd.Series(y_obs, name="nice_feature")
        df_bias = compute_bias(
            y_obs=y_obs,
            y_pred=y_pred,
            feature=feature,
        )
        f_expected = [1.0, 2, 1, 2]
        df_expected = pl.DataFrame(
            {
                "model": ["model_1", "model_1", "model_2", "model_2"],
                "nice_feature": f_expected,
                "bias_mean": [0.0, -1, 2, 1],
                "bias_count": np.array([5] * 4, dtype=np.uint32),
                "bias_stderr": [0.0] * 4,
                "p_value": [np.nan] * 4,
            }
        )

        if feature_type == "cat":
            df_expected = df_expected.replace(
                "nice_feature",
                df_expected["nice_feature"].cast(pl.Utf8).cast(pl.Categorical),
            )
        elif feature_type == "string":
            df_expected = df_expected.replace(
                "nice_feature", df_expected["nice_feature"].cast(pl.Utf8)
            )
        assert_frame_equal(df_bias, df_expected, check_exact=False)

        # Same for pure numpy input.
        feature_np = feature.to_numpy()
        if feature_type == "cat":
            # convert object to pd.Categorical
            feature_np = pd.Series(feature_np, dtype="category")
        elif feature_type == "string":
            feature_np = feature_np.astype("=U8")  # to_numpy gives dtype=object
        df_bias = compute_bias(
            y_obs=y_obs,
            y_pred=y_pred.to_numpy(),
            feature=feature_np,
        )
        df_expected = df_expected.replace("model", pl.Series(["0", "0", "1", "1"]))
        df_expected = df_expected.rename({"nice_feature": "feature"})
        assert_frame_equal(df_bias, df_expected, check_exact=False)

        # Model and feature name clash.
        feature = feature.rename("model")
        df_bias = compute_bias(
            y_obs=y_obs,
            y_pred=y_pred,
            feature=feature,
        )
        df_expected = df_expected.rename({"model": "model_", "feature": "model"})
        df_expected = df_expected.replace(
            "model_", pl.Series(["model_1", "model_1", "model_2", "model_2"])
        )
        print(f"left\n{df_bias}")
        print(f"right\n{df_expected}")
        assert_frame_equal(df_bias, df_expected, check_exact=False)
