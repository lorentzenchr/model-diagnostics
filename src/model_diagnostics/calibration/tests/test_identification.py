from functools import partial

import numpy as np
import polars as pl
import pytest
from packaging.version import Version
from polars.testing import assert_frame_equal, assert_series_equal
from scipy.special import stdtr
from scipy.stats import expectile, ttest_1samp

from model_diagnostics._utils.test_helper import (
    SkipContainer,
    pa_array,
    pa_DictionaryArray_from_arrays,
    pd_DataFrame,
    pd_Series,
)
from model_diagnostics.calibration import compute_bias, identification_function
from model_diagnostics.calibration.identification import compute_marginal


@pytest.mark.parametrize(
    ("functional", "est"),
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
    ("functional", "level", "msg"),
    [
        ("no good functional", 0.5, "Argument functional must be one of"),
        ("quantile", 0, "Argument level must fulfil 0 < level < 1"),
        ("quantile", 1, "Argument level must fulfil 0 < level < 1"),
        ("quantile", 1.1, "Argument level must fulfil 0 < level < 1"),
    ],
)
def test_identification_function_raises(functional, level, msg):
    y_obs, y_pred = np.arange(5), np.arange(5)
    with pytest.raises(ValueError, match=msg):
        identification_function(y_obs, y_pred, functional=functional, level=level)


@pytest.mark.parametrize(
    ("feature", "f_grouped"),
    [
        (
            pl.Series(["a", "a", "b", "b", "b"]),
            pl.Series(["a", "b"]),
        ),
        (
            pl.Series([0.1, 0.1, 0.9, 0.9, 0.9]),
            pl.Series([0.1, 0.9]),
        ),
        (
            pl.Series([None, np.nan, 1.0, 1, 1]),
            pl.Series([None, 1.0]),
        ),
        (
            pl.Series(["a", "a", None, None, None]),
            pl.Series(["a", None]),
        ),
        (
            pa_array(["a", "a", "b", "b", "b"]),
            pa_array(["a", "b"]),
        ),
        (
            pa_array([0.1, 0.1, 0.9, 0.9, 0.9]),
            pa_array([0.1, 0.9]),
        ),
        (
            pa_array([None, np.nan, 1.0, 1, 1]),
            pa_array([None, 1.0]),
        ),
        (
            pa_array(["a", "a", None, None, None]),
            pa_array(["a", None]),
        ),
        (
            pa_DictionaryArray_from_arrays([0, 0, 1, 1, 1], ["b", "a"]),
            pa_DictionaryArray_from_arrays(np.array([0, 1], dtype=np.int8), ["b", "a"]),
        ),
    ],
)
def test_compute_bias(feature, f_grouped):
    """Test compute_bias on simple data."""
    if isinstance(feature, SkipContainer):
        pytest.skip("Module for data container not imported.")
    with pl.StringCache():
        # The string cache is needed to avoid the following error message:
        # exceptions.ComputeError: Cannot compare categoricals originating from
        # different sources. Consider setting a global string cache.
        df = pl.DataFrame(
            {
                "y_obs": [0, 1, 2, 4, 3],
                "y_pred": [1, 1, 2, 2, 2],
                "feature": feature,
            }
        )
        # V = [1, 0, 0, -2, -1]
        df_bias = compute_bias(
            y_obs=df.get_column("y_obs"),
            y_pred=df.get_column("y_pred"),
            feature=df.get_column("feature"),
        )
        df_expected = pl.DataFrame(
            {
                "feature": f_grouped,
                "bias_mean": [0.5, -1],
                "bias_count": pl.Series(values=[2, 3], dtype=pl.UInt32),
                "bias_weights": pl.Series(values=[2, 3], dtype=pl.Float64),
                "bias_stderr": np.sqrt([(0.25 + 0.25), (1 + 1 + 0) / 2])
                / np.sqrt([2, 3]),
                "p_value": [
                    ttest_1samp([1, 0], 0).pvalue,
                    ttest_1samp([0, -2, -1], 0).pvalue,
                ],
            }
        ).sort("feature")
        assert_frame_equal(df_bias, df_expected, check_exact=False)

        # Same with weights.
        feature = pl.Series(values=feature).gather([0, 4, 4]).alias("feature")
        df_bias = compute_bias(
            # y_obs=[0.5, (1 * 2 + 0.5 * 4) / 1.5, (0.5 * 4 + 3) / 1.5]
            y_obs=[0.5, 8 / 3, 10 / 3],
            y_pred=[1, 2, 2],
            weights=[2, 1.5, 1.5],
            feature=feature,
        )
        # V = [0.5, -2/3, -4/3]
        df_expected = pl.DataFrame(
            {
                "feature": f_grouped,
                "bias_mean": [0.5, -1],
                "bias_count": pl.Series(values=[1, 2], dtype=pl.UInt32),
                "bias_weights": pl.Series(values=[2, 3], dtype=pl.Float64),
                # For feature[0], the variance is 0 because there is only one
                # observation. For feature[4], a direct calculation gives:
                # SE = sqrt((1.5 * 1/9 + 1.5 * 1/9) / 3 / (2-1)) = sqrt(1 / (3 * 3 * 1))
                #    = sqrt(1/9) = 1/3
                "bias_stderr": [0.0, 1 / 3],
                "p_value": [np.nan, 2 * stdtr(2 - 1, -np.abs(-1 / (1 / 3)))],
            }
        ).sort("feature")
        assert_frame_equal(df_bias, df_expected, check_exact=False)


def test_compute_bias_feature_none():
    """Test compute_bias for feature = None."""
    df = pl.DataFrame(
        {
            "y_obs": [0, 1, 2, 4, 3],
            "y_pred": [1, 1, 2, 2, 2],
        }
    )
    # V = [1, 0, 0, -2, -1]
    df_bias = compute_bias(
        y_obs=df.get_column("y_obs"),
        y_pred=df.get_column("y_pred"),
        feature=None,
    )
    df_expected = pl.DataFrame(
        {
            "bias_mean": [-0.4],  # (1 + 0 + 0 - 2 - 1) / 5
            "bias_count": pl.Series(values=[5], dtype=pl.UInt32),
            "bias_weights": [5.0],
            "bias_stderr": [np.std([1, 0, 0, -2, -1], ddof=1) / np.sqrt(5)],
            "p_value": [
                ttest_1samp([1, 0, 0, -2, -1], 0).pvalue,
            ],
        }
    )
    assert_frame_equal(df_bias, df_expected)

    # Same with weights.
    df_bias = compute_bias(
        # y_obs=[0.5, (1 * 2 + 0.5 * 4) / 1.5, (0.5 * 4 + 3) / 1.5]
        y_obs=[0.5, 8 / 3, 10 / 3],
        y_pred=[1, 2, 2],
        weights=[2, 1.5, 1.5],
        feature=None,
    )
    # V = [0.5, -2/3, -4/3]
    df_expected = df_expected.with_columns(
        pl.Series(values=[3], dtype=pl.UInt32).alias("bias_count"),
        # SE = sqrt((2 * 0.9**2 + 1.5 * 0.8**2/9 + 1.5 * 2.8**2/9) / 5 / (3-1))
        #    = sqrt(91 / 300)
        pl.Series(values=[np.sqrt(91 / 300)]).alias("bias_stderr"),
        pl.Series(values=[2 * stdtr(3 - 1, -np.abs(-0.4 / np.sqrt(91 / 300)))]).alias(
            "p_value"
        ),
    )
    assert_frame_equal(df_bias, df_expected, check_exact=False)


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
        bin_method="quantile",
    )
    bias = (df.get_column("y_pred") - df.get_column("y_obs")).to_numpy()
    df_expected = pl.DataFrame(
        {
            "feature": 0.045 + 0.1 * np.arange(10),
            "bias_mean": 0.955 - 0.1 * np.arange(10),
            "bias_count": n_steps * np.ones(n_bins, dtype=np.uint32),
            "bias_weights": [n_obs / n_bins] * n_bins,
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


# Filter warnings for method "stone".
@pytest.mark.filterwarnings(
    "ignore:.*The number of bins estimated may be suboptimal.*:RuntimeWarning"
)
@pytest.mark.parametrize(
    ("bin_method", "n_bins"),
    [
        ("quantile", 2),
        ("quantile", 10),
        ("uniform", 2),
        ("uniform", 10),
        ("auto", 10),
        ("fd", 10),
        ("doane", 10),
        ("scott", 10),
        ("stone", 10),
        ("rice", 10),
        ("sturges", 10),
        ("sqrt", 10),
    ],
)
def test_compute_bias_n_bins_numerical_feature(bin_method, n_bins):
    """Test compute_bias returns right number of bins for a numerical feature."""
    n_obs = 10
    y_obs = np.linspace(-1, 1, num=n_obs, endpoint=False)
    y_pred = y_obs**2
    feature = [4, 4, 4, 4, 3, 3, 3, 2, 2, 11]
    df_bias = compute_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        feature=feature,
        n_bins=n_bins,
        bin_method=bin_method,
    )
    if bin_method in ("quantile", "uniform"):
        assert df_bias.shape[0] == np.min([n_bins, 4])
    else:
        assert 2 <= df_bias.shape[0] <= 10
    assert df_bias["bias_count"].sum() == n_obs


@pytest.mark.parametrize("feature_type", ["cat", "cat_physical", "enum", "string"])
def test_compute_bias_n_bins_string_like_feature(feature_type):
    """Test compute_bias returns right number of bins and sorted for string feature."""
    if feature_type == "cat":
        dtype = pl.Categorical
    elif feature_type == "cat_physical":
        dtype = pl.Categorical(ordering="physical")
    elif feature_type == "enum":
        dtype = pl.Enum(categories=["b", "a", "c"])
    else:
        dtype = pl.Utf8

    with pl.StringCache():
        n_bins = 3
        n_obs = 6
        y_obs = np.arange(n_obs)
        y_pred = pl.Series("model", np.arange(n_obs) + 0.5)
        feature = pl.Series("feature", ["a", "a", None, "b", "b", "c"], dtype=dtype)

        if feature_type == "enum":
            expected_feature = pl.Series(
                values=[None, "b", "other 2"], dtype=pl.Enum(["b", "a", "c", "other 2"])
            )
        else:
            expected_feature = pl.Series(values=[None, "a", "other 2"], dtype=dtype)

        df_expected = pl.DataFrame(
            {
                "feature": expected_feature,
                "bias_mean": 0.5,
                "bias_count": pl.Series([1, 2, 3], dtype=pl.UInt32),
            }
        )
        for _i in range(10):
            # The default args in polars group_by(..., maintain_order=False) returns
            # non-deterministic ordering which compute_bias should take care of such
            # that compute_bias is deterministic.
            df = compute_bias(
                y_obs=y_obs, y_pred=y_pred, feature=feature, n_bins=n_bins
            )
            assert_frame_equal(
                df.select(["feature", "bias_mean", "bias_count"]), df_expected
            )


@pytest.mark.parametrize("feature_type", ["cat", "num", "string"])
def test_compute_bias_multiple_predictions(feature_type):
    """Test compute_bias for multiple predictions."""
    with pl.StringCache():
        n_obs = 10
        y_obs = np.ones(n_obs)
        y_obs[: 10 // 2] = 2
        y_pred = pl.DataFrame(
            {"model_1": np.ones(n_obs), "model_2": 3 * np.ones(n_obs)}
        )
        if feature_type == "cat":
            feature = pd_Series(
                y_obs.astype("=U8"), dtype="category", name="nice_feature"
            )
        elif feature_type == "string":
            feature = pd_Series(y_obs.astype("=U8"), name="nice_feature")
        else:
            feature = pl.Series(values=y_obs, name="nice_feature")

        if isinstance(feature, SkipContainer):
            pytest.skip("Module for data container not imported.")

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
                "bias_count": pl.Series(values=[5] * 4, dtype=pl.UInt32),
                "bias_weights": [5.0] * 4,
                "bias_stderr": [0.0] * 4,
                "p_value": [0.0] * 4,
            }
        )

        if feature_type == "cat":
            df_expected = df_expected.with_columns(
                df_expected["nice_feature"]
                .cast(pl.Utf8)
                .cast(pl.Categorical)
                .alias("nice_feature"),
            )
        elif feature_type == "string":
            df_expected = df_expected.with_columns(
                df_expected["nice_feature"].cast(pl.Utf8).alias("nice_feature")
            )
        assert_frame_equal(df_bias, df_expected, check_exact=False)

        # Same for pure numpy input.
        feature_np = feature.to_numpy()
        if feature_type == "cat":
            # convert object to pd.Categorical
            feature_np = pd_Series(feature_np, dtype="category")
        elif feature_type == "string":
            feature_np = feature_np.astype("=U8")  # to_numpy gives dtype=object
        df_bias = compute_bias(
            y_obs=y_obs,
            y_pred=y_pred.to_numpy(),
            feature=feature_np,
        )
        df_expected = df_expected.with_columns(
            pl.Series(["0", "0", "1", "1"]).alias("model")
        )
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
        df_expected = df_expected.with_columns(
            pl.Series(["model_1", "model_1", "model_2", "model_2"]).alias("model_")
        )
        assert_frame_equal(df_bias, df_expected, check_exact=False)


@pytest.mark.parametrize("feature", [None, [1, 1]])
def test_compute_bias_model_name_order(feature):
    """Test that the order of the colum model is not sorted."""
    y_obs = [0, 0]
    y_pred = pl.DataFrame(
        {
            "model_3": [1, 1],
            "model_1": [0, 1],
            "model_2": [0, 1],
        }
    )
    df_bias = compute_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        feature=feature,
    )
    assert df_bias.get_column("model").to_list() == ["model_3", "model_1", "model_2"]


def test_compute_bias_many_sparse_feature_values():
    """Test that compute_bias returns same values for high cardinality feature."""
    n_obs = 100
    y_obs = np.arange(n_obs)
    y_pred = pl.DataFrame(
        {
            "model_1": np.arange(n_obs) + 0.5,
            "model_2": (y_obs - 5) ** 2,
            "model_3": (y_obs - 3) ** 2,
        }
    )
    rng = np.random.default_rng(42)
    feature = rng.integers(low=0, high=n_obs // 2, size=n_obs).astype(str)

    df = compute_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        feature=feature,
        n_bins=10,
    )

    assert set(df.filter(pl.col("model") == "model_1")["feature"]) == set(
        df.filter(pl.col("model") == "model_2")["feature"]
    )
    assert set(df.filter(pl.col("model") == "model_1")["feature"]) == set(
        df.filter(pl.col("model") == "model_3")["feature"]
    )


def test_compute_bias_keeps_null_values():
    """Test that compute_bias keeps Null values."""
    n_obs = 10
    y_obs = np.arange(n_obs)
    y_pred = (y_obs - 5) ** 2
    feature = pl.Series([np.nan, None, 1, 1, 1, 2, 2, 2, 2, 2.0])

    df_bias = compute_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        feature=feature,
        n_bins=2,
        bin_method="quantile",
    )
    assert_series_equal(df_bias["feature"], pl.Series("feature", [None, 1.625]))
    assert_series_equal(
        df_bias["bias_count"], pl.Series("bias_count", [2, 8], dtype=pl.UInt32)
    )
    assert df_bias["bias_count"].sum() == n_obs

    df_bias = compute_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        feature=feature,
        n_bins=4,
        bin_method="quantile",
    )
    assert_series_equal(df_bias["feature"], pl.Series("feature", [None, 1.0, 2.0]))
    assert_series_equal(
        df_bias["bias_count"], pl.Series("bias_count", [2, 3, 5], dtype=pl.UInt32)
    )
    assert df_bias["bias_count"].sum() == n_obs


def test_compute_bias_raises_weights_shape():
    y_obs, y_pred = np.arange(5), np.arange(5)
    weights = np.arange(5)[:, None]
    msg = "The array weights must be 1-dimensional, got weights.ndim=2."
    with pytest.raises(ValueError, match=msg):
        compute_bias(y_obs, y_pred, weights=weights)


def test_compute_bias_raises_bin_method():
    y_obs, y_pred, feature = np.arange(5), np.arange(5), np.arange(5)
    msg = "Parameter bin_method must be one of .*quantile"
    with pytest.raises(ValueError, match=msg):
        compute_bias(y_obs, y_pred, feature, n_bins=5, bin_method=None)


@pytest.mark.parametrize(
    "list2array",
    [lambda x: x, np.asarray, pa_array, pd_Series, pl.Series],
)
def test_compute_bias_1d_array_like(list2array):
    """Test that compute_bias workds for 1d array-likes."""
    y_obs = list2array([0, 1, 2, 3])
    y_pred = list2array([-1, 1, 0, 2])
    feature = list2array([0, 1, 0, 1])
    weights = list2array([1, 1, 1, 1])
    if isinstance(y_pred, SkipContainer):
        pytest.skip("Module for data container not imported.")
    df_bias = compute_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        weights=weights,
        feature=feature,
        bin_method="quantile",
    )
    df_expected = pl.DataFrame(
        {
            "feature": [0.0, 1.0],
            "bias_mean": [-1.5, -0.5],
            "bias_count": pl.Series(values=[2, 2], dtype=pl.UInt32),
            # For unknown reasons, on Windos this can be int32 instead of int64.
            "bias_weights": pl.Series([2, 2], dtype=df_bias["bias_weights"].dtype),
            "bias_stderr": [0.5, 0.5],
            "p_value": [0.20483276469913345, 0.5],
        }
    )
    assert_frame_equal(df_bias, df_expected, check_exact=False)


@pytest.mark.parametrize(
    ("feature", "f_grouped", "bin_edges"),
    [
        (
            pl.Series(["a", "a", "b", "b", "b"]),
            pl.Series(["a", "b"]),
            None,
        ),
        (
            pl.Series([0.1, 0.1, 0.9, 0.9, 0.9]),
            pl.Series([0.1, 0.9]),
            pl.Series([[0.1, 0, 0.1], [0.1, 0, 0.9]], dtype=pl.Array(pl.Float64, 3)),
        ),
        (
            pl.Series([None, np.nan, 1.0, 1, 1]),
            pl.Series([None, 1.0]),
            pl.Series([[None] * 3, [1.0, 0, 1.0]], dtype=pl.Array(pl.Float64, 3)),
        ),
        (
            pl.Series(["a", "a", None, None, None]),
            pl.Series(["a", None]),
            None,
        ),
        (
            pa_array(["a", "a", "b", "b", "b"]),
            pa_array(["a", "b"]),
            None,
        ),
        (
            pa_array([0.1, 0.1, 0.9, 0.9, 0.9]),
            pa_array([0.1, 0.9]),
            pl.Series([[0.1, 0, 0.1], [0.1, 0, 0.9]], dtype=pl.Array(pl.Float64, 3)),
        ),
        (
            pa_array([None, np.nan, 1.0, 1, 1]),
            pa_array([None, 1.0]),
            pl.Series([[None] * 3, [1.0, 0, 1.0]], dtype=pl.Array(pl.Float64, 3)),
        ),
        (
            pa_array(["a", "a", None, None, None]),
            pa_array(["a", None]),
            None,
        ),
        (
            pa_DictionaryArray_from_arrays([0, 0, 1, 1, 1], ["b", "a"]),
            pa_DictionaryArray_from_arrays(np.array([0, 1], dtype=np.int8), ["b", "a"]),
            None,
        ),
    ],
)
def test_compute_marginal(feature, f_grouped, bin_edges):
    """Test compute_marginal on simple data."""
    if isinstance(feature, SkipContainer):
        pytest.skip("Module for data container not imported.")
    with pl.StringCache():
        # The string cache is needed to avoid the following error message:
        # exceptions.ComputeError: Cannot compare categoricals originating from
        # different sources. Consider setting a global string cache.
        df = pl.DataFrame(
            {
                "y_obs": [0, 1, 2, 4, 3],
                "y_pred": [1, 1, 2, 2, 2],
                "feature": feature,
            }
        )
        df_marginal = compute_marginal(
            y_obs=df.get_column("y_obs"),
            y_pred=df.get_column("y_pred"),
            X=df.select("feature"),
            feature_name="feature",
            bin_method="quantile",
        )
        df_expected = pl.DataFrame(
            {
                "feature": f_grouped,
                "y_obs_mean": [0.5, 3],
                "y_pred_mean": [1.0, 2.0],
                "y_obs_stderr": np.sqrt([0.25 + 0.25, (1 + 1 + 0) / 2])
                / np.sqrt([2, 3]),
                "y_pred_stderr": [0.0, 0.0],
                "count": pl.Series(values=[2, 3], dtype=pl.UInt32),
                "weights": pl.Series(values=[2, 3], dtype=pl.Float64),
            }
        )
        if bin_edges is not None:
            df_expected = df_expected.hstack([bin_edges.alias("bin_edges")])

            if Version("1.11.0") <= Version(pl.__version__) <= Version("1.12"):
                # See https://github.com/pola-rs/polars/issues/19482
                pytest.skip()

        df_expected = df_expected.sort("feature")
        assert_frame_equal(df_marginal, df_expected, check_exact=False)

        # Same with weights.
        feature = pl.Series(values=feature).gather([0, 4, 4]).alias("feature")
        df_marginal = compute_marginal(
            # y_obs=[0.5, (1 * 2 + 0.5 * 4) / 1.5, (0.5 * 4 + 3) / 1.5]
            y_obs=[0.5, 8 / 3, 10 / 3],
            y_pred=[1, 2, 2],
            weights=[2, 1.5, 1.5],
            X=pl.DataFrame({"feature": feature}),
            feature_name="feature",
            bin_method="quantile",
        )
        df_expected = pl.DataFrame(
            {
                "feature": f_grouped,
                "y_obs_mean": [0.5, 3.0],
                "y_pred_mean": [1.0, 2.0],
                "y_obs_stderr": [0.0, np.sqrt((1 / 9 + 1 / 9) / 1 / 2)],
                "y_pred_stderr": [0.0, 0.0],
                "count": pl.Series(values=[1, 2], dtype=pl.UInt32),
                "weights": pl.Series(values=[2, 3], dtype=pl.Float64),
            }
        )
        if bin_edges is not None:
            df_expected = df_expected.hstack([bin_edges.alias("bin_edges")])
        df_expected = df_expected.sort("feature")
        assert_frame_equal(df_marginal, df_expected, check_exact=False)


def test_compute_marginal_feature_none():
    """Test compute_marginal for feature = None."""
    df = pl.DataFrame(
        {
            "y_obs": [0, 1, 2, 4, 3],
            "y_pred": [1, 1, 2, 2, 2],
        }
    )
    df_marginal = compute_marginal(
        y_obs=df.get_column("y_obs"),
        y_pred=df.get_column("y_pred"),
        X=None,
        feature_name=None,
    )
    df_expected = pl.DataFrame(
        {
            "y_obs_mean": [2.0],
            "y_pred_mean": [8 / 5],
            "y_obs_stderr": [np.sqrt(2.5 / 5)],
            "y_pred_stderr": [np.sqrt((9 + 9 + 4 + 4 + 4) / 25 / 4 / 5)],
            "count": pl.Series(values=[5], dtype=pl.UInt32),
            "weights": [5.0],
        }
    )
    assert_frame_equal(df_marginal, df_expected)

    # Same with weights.
    df_marginal = compute_marginal(
        # y_obs=[0.5, (1 * 2 + 0.5 * 4) / 1.5, (0.5 * 4 + 3) / 1.5]
        y_obs=[0.5, 8 / 3, 10 / 3],
        y_pred=[1, 2, 2],
        weights=[2, 1.5, 1.5],
        X=None,
        feature_name=None,
    )
    df_expected = df_expected.with_columns(
        pl.Series(values=[3], dtype=pl.UInt32).alias("count"),
        pl.Series(values=[np.sqrt((9 / 2 + 2 / 3 + 8 / 3) / 5 / 2)]).alias(
            "y_obs_stderr"
        ),
        pl.Series(values=[np.sqrt(1.2 / 5 / 2)]).alias("y_pred_stderr"),
    )
    assert_frame_equal(df_marginal, df_expected, check_exact=False)


# Filter warnings for method "stone".
@pytest.mark.filterwarnings(
    "ignore:.*The number of bins estimated may be suboptimal.*:RuntimeWarning"
)
@pytest.mark.parametrize(
    "bin_method",
    [
        "quantile",
        "uniform",
        "auto",
        "fd",
        "doane",
        "scott",
        "stone",
        "rice",
        "sturges",
        "sqrt",
    ],
)
def test_compute_marginal_numerical_feature(bin_method):
    """Test compute_marginal for a numerical feature."""
    n_obs = 100
    n_bins = 10
    n_steps = n_obs // n_bins
    df = pl.DataFrame(
        {
            "y_obs": (y_obs := 2 * np.linspace(-0.5, 0.5, num=n_obs, endpoint=False)),
            "y_pred": (y_pred := np.linspace(0, 1, num=n_obs, endpoint=False)),
            "feature": np.linspace(0, 1, num=n_obs, endpoint=False),
        }
    )
    df_marginal = compute_marginal(
        y_obs=df.get_column("y_obs"),
        y_pred=df.get_column("y_pred"),
        X=df.select("feature"),
        feature_name="feature",
        n_bins=n_bins,
        bin_method=bin_method,
    )
    f_std = np.linspace(0, 1, 10, endpoint=False).std(ddof=0) / n_bins
    df_expected = pl.DataFrame(
        {
            "feature": 0.045 + 0.1 * np.arange(10),
            "y_obs_mean": -0.91 + 0.2 * np.arange(10),
            "y_pred_mean": 0.045 + 0.1 * np.arange(10),
            "y_obs_stderr": [
                np.std(y_obs[n : n + n_steps], ddof=1) / np.sqrt(n_steps)
                for n in range(0, n_obs, n_steps)
            ],
            "y_pred_stderr": [
                np.std(y_pred[n : n + n_steps], ddof=1) / np.sqrt(n_steps)
                for n in range(0, n_obs, n_steps)
            ],
            "count": n_steps * np.ones(n_bins, dtype=np.uint32),
            "weights": [n_obs / n_bins] * n_bins,
            "bin_edges": pl.Series(
                # min, std, max
                [
                    [0, f_std, 0.09],
                    *[
                        [x * 0.1 - 0.01, f_std, (x + 1) * 0.1 - 0.01]
                        for x in range(1, 10)
                    ],
                ]
                if bin_method == "quantile"
                else [[0.099 * x, f_std, 0.099 * (x + 1)] for x in range(10)],
                dtype=pl.Array(pl.Float64, 3),
            ),
        }
    )

    if bin_method in ("quantile", "uniform", "rice", "sqrt"):
        assert_frame_equal(df_marginal, df_expected, check_exact=False)
    else:
        # Method "stone" produces only 1 bin.
        assert 1 <= df_marginal.shape[0] < n_obs
        assert_series_equal(
            df_marginal["feature"], df_marginal["y_pred_mean"], check_names=False
        )


@pytest.mark.parametrize("bin_method", ["quantile", "uniform"])
@pytest.mark.parametrize("n_bins", [2, 10])
def test_compute_marginal_n_bins_numerical_feature(bin_method, n_bins):
    """Test compute_marginal returns right number of bins for a numerical feature."""
    n_obs = 10
    y_obs = np.linspace(-1, 1, num=n_obs, endpoint=False)
    y_pred = y_obs**2
    feature = [4, 4, 4, 4, 3, 3, 3, 2, 2, 11]
    df_marginal = compute_marginal(
        y_obs=y_obs,
        y_pred=y_pred,
        X=pl.DataFrame({"col": feature}),
        feature_name="col",
        n_bins=n_bins,
        bin_method=bin_method,
    )
    assert df_marginal.shape[0] == np.min([n_bins, 4])
    assert df_marginal["count"].sum() == n_obs


@pytest.mark.parametrize("feature_type", ["cat", "cat_physical", "enum", "string"])
def test_compute_marginal_n_bins_string_like_feature(feature_type):
    """Test compute_marginal returns right number of bins and sorted string feature."""
    if feature_type == "cat":
        dtype = pl.Categorical
    elif feature_type == "cat_physical":
        dtype = pl.Categorical(ordering="physical")
    elif feature_type == "enum":
        dtype = pl.Enum(categories=["b", "a", "c"])
    else:
        dtype = pl.Utf8

    with pl.StringCache():
        n_bins = 3
        n_obs = 6
        y_obs = np.array([1, 1, 0, 3, 1, 2])
        y_pred = np.zeros(n_obs)
        feature = pl.Series("feature", ["a", "a", None, "b", "b", "c"], dtype=dtype)

        if feature_type == "enum":
            feature_expected = pl.Series(
                [None, "b", "other 2"], dtype=pl.Enum(["b", "a", "c", "other 2"])
            )
            y_obs_mean = [0.0, 2, 4 / 3]
        else:
            feature_expected = pl.Series([None, "a", "other 2"], dtype=dtype)
            y_obs_mean = [0.0, 1, 2]

        df_expected = pl.DataFrame(
            {
                "feature": feature_expected,
                "y_obs_mean": y_obs_mean,
                "y_pred_mean": 0.0,
                "count": pl.Series([1, 2, 3], dtype=pl.UInt32),
            }
        )
        for _i in range(10):
            # The default args in polars group_by(..., maintain_order=False) returns
            # non-deterministic ordering which compute_bias should take care of such
            # that compute_bias is deterministic.
            df = compute_marginal(
                y_obs=y_obs,
                y_pred=y_pred,
                X=pl.DataFrame({"feature": feature}),
                feature_name=0,
                n_bins=n_bins,
            )
            assert_frame_equal(
                df.select(["feature", "y_obs_mean", "y_pred_mean", "count"]),
                df_expected,
            )


@pytest.mark.parametrize("feature_type", ["cat", "num", "string"])
def test_compute_marginal_multiple_predictions(feature_type):
    """Test compute_marginal for multiple predictions."""
    with pl.StringCache():
        n_obs = 10
        y_obs = np.ones(n_obs)
        y_obs[: 10 // 2] = 2
        y_pred = pl.DataFrame(
            {"model_1": np.ones(n_obs), "model_2": 3 * np.ones(n_obs)}
        )
        if feature_type == "cat":
            feature = pd_DataFrame(
                pd_Series(y_obs.astype("=U8"), dtype="category", name="nice_feature")
            )
        elif feature_type == "string":
            feature = pd_DataFrame(pd_Series(y_obs.astype("=U8"), name="nice_feature"))
        else:
            feature = pl.Series(values=y_obs, name="nice_feature").to_frame()

        if isinstance(feature, SkipContainer):
            pytest.skip("Module for data container not imported.")

        df_marginal = compute_marginal(
            y_obs=y_obs,
            y_pred=y_pred,
            X=feature,
            feature_name="nice_feature",
        ).drop(["y_obs_stderr", "y_pred_stderr", "weights"])
        if "bin_edges" in df_marginal.columns:
            df_marginal = df_marginal.drop("bin_edges")
        f_expected = [1.0, 2, 1, 2]
        df_expected = pl.DataFrame(
            {
                "model": ["model_1", "model_1", "model_2", "model_2"],
                "nice_feature": f_expected,
                "y_obs_mean": [1.0, 2, 1, 2],
                "y_pred_mean": [1.0, 1.0, 3, 3],
                "count": pl.Series(values=[5] * 4, dtype=pl.UInt32),
            }
        )

        if feature_type == "cat":
            df_expected = df_expected.with_columns(
                df_expected["nice_feature"]
                .cast(pl.Utf8)
                .cast(pl.Categorical)
                .alias("nice_feature"),
            )
        elif feature_type == "string":
            df_expected = df_expected.with_columns(
                df_expected["nice_feature"].cast(pl.Utf8).alias("nice_feature")
            )
        assert_frame_equal(df_marginal, df_expected, check_exact=False)

        # Same for pure numpy input.
        feature_np = feature.to_numpy()
        if feature_type == "cat":
            # convert object to pd.Categorical
            feature_np = pd_DataFrame(pd_Series(feature_np.squeeze(), dtype="category"))
        elif feature_type == "string":
            feature_np = feature_np.astype("=U8")  # to_numpy gives dtype=object
        df_marginal = compute_marginal(
            y_obs=y_obs,
            y_pred=y_pred.to_numpy(),
            X=feature_np,
            feature_name=0,
        ).drop(["y_obs_stderr", "y_pred_stderr", "weights"])
        if "bin_edges" in df_marginal.columns:
            df_marginal = df_marginal.drop("bin_edges")
        df_expected = df_expected.with_columns(
            pl.Series(["0", "0", "1", "1"]).alias("model")
        )
        df_expected = df_expected.rename({"nice_feature": "feature 0"})
        assert_frame_equal(df_marginal, df_expected, check_exact=False)

        # Model and feature name clash.
        if isinstance(feature, pl.DataFrame):
            feature = feature.rename({"nice_feature": "model"})
        else:
            feature = feature.rename(columns={"nice_feature": "model"})
        df_marginal = compute_marginal(
            y_obs=y_obs,
            y_pred=y_pred,
            X=feature,
            feature_name="model",
        ).drop(["y_obs_stderr", "y_pred_stderr", "weights"])
        if "bin_edges" in df_marginal.columns:
            df_marginal = df_marginal.drop("bin_edges")
        df_expected = df_expected.rename({"model": "model_", "feature 0": "model"})
        df_expected = df_expected.with_columns(
            pl.Series(["model_1", "model_1", "model_2", "model_2"]).alias("model_")
        )
        assert_frame_equal(df_marginal, df_expected, check_exact=False)


# TODO: We could also add test for compute_marginal like
# - test_compute_bias_keeps_null_values
# - test_compute_bias_warning_for_n_bins
# - test_compute_bias_raises_weights_shape
# - test_compute_bias_1d_array_like


@pytest.mark.parametrize("weights", [None, True])
def test_compute_marginal_with_partial_dependence(weights):
    """Test partial_dependence values in compute_marginal."""
    n_obs = 20
    n_bins = 5
    X = pl.DataFrame(
        {
            "a": np.arange(n_obs) % n_bins,
            "b": -(np.arange(n_obs) % n_bins),
        }
    )
    if weights is not None:
        weights = np.ones(n_obs)

    def predict(X):
        a = X.get_column("a")
        b = X.get_column("b")
        return a + b - 2 * a * b

    df = compute_marginal(
        y_obs=np.zeros(n_obs),
        y_pred=predict(X),
        X=X,
        feature_name="a",
        predict_function=predict,
        weights=weights,
        n_bins=100,
        bin_method="uniform",
        n_max=n_obs,
        rng=123,
    ).select(["a", "y_obs_mean", "y_pred_mean", "partial_dependence"])

    df_expected = pl.DataFrame(
        {
            "a": pl.Series(np.arange(n_bins), dtype=pl.Float64),
            "y_obs_mean": 0.0,
            "y_pred_mean": 2.0 * np.arange(n_bins) ** 2,
            "partial_dependence": 5.0 * np.arange(n_bins) - 2,
        }
    )
    assert_frame_equal(df, df_expected)


@pytest.mark.parametrize("with_null", [False, True])
def test_compute_marginal_with_partial_dependence_on_strings(with_null):
    """Test partial_dependence in compute_marginal with a other n value."""
    n_obs = 20
    n_bins = 3
    X = pl.DataFrame(
        {
            "a": np.arange(n_obs) % 5,
            "b": ["a", "b", "c", "d", None if with_null else "e"] * 4,
        }
    )

    # The Python function ord throws an error if "other 3" is passed to it.
    def predict(X):
        a = X.get_column("a")
        b = (
            X.get_column("b")
            .map_elements(lambda x: ord(x) - ord("a"), return_dtype=pl.Int32)
            .fill_null(ord("e") - ord("a"))
        )

        return a + b - 2 * a * b

    df = compute_marginal(
        y_obs=np.zeros(n_obs),
        y_pred=predict(X),
        X=X,
        feature_name="b",
        predict_function=predict,
        n_bins=n_bins,
        bin_method="uniform",
        n_max=n_obs,
        rng=123,
    ).select(["b", "y_obs_mean", "partial_dependence"])

    df_expected = pl.DataFrame(
        {
            "b": [None, "a", "other 3"] if with_null else ["a", "b", "other 3"],
            "y_obs_mean": 0.0,
            "partial_dependence": [-10.0, 2, None] if with_null else [2.0, -1, None],
        }
    )
    assert_frame_equal(df, df_expected)


def test_compute_marginal_raises_X_None():
    """Test that compute_marginal raises if X=None but feature_name not None."""
    n_obs = 5
    msg = "X must be a data container like*."
    with pytest.raises(ValueError, match=msg):
        compute_marginal(
            y_obs=np.zeros(n_obs),
            y_pred=np.ones(n_obs),
            X=None,
            feature_name="a",
        )
