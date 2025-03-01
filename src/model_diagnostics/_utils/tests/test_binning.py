from importlib.metadata import version

import numpy as np
import polars as pl
import pytest
from packaging.version import Version, parse
from polars.testing import assert_series_equal

from model_diagnostics._utils.binning import bin_feature


def test_bin_feature_raises():
    feature = np.arange(5)
    msg = "Parameter bin_method must be one of .*quantile"
    with pytest.raises(ValueError, match=msg):
        bin_feature(
            feature=feature,
            feature_name=None,
            n_obs=feature.shape[0],
            bin_method="wrong",
        )


@pytest.mark.parametrize("with_null", [False, True])
@pytest.mark.parametrize("feature_type", ["str", "cat", "enum"])
@pytest.mark.parametrize("n_bins", [3, 5])
@pl.StringCache()
def test_binning_strings_categorical(with_null, feature_type, n_bins):
    """Test that binning works for strings, categoricals, enums."""
    feature = pl.Series(name="my_feature", values=["a", "b", "c"] * 3 + ["e"])
    if with_null:
        feature[0] = None
        assert feature.has_nulls()
        rest_name = "other 3"
        f_binned_expected = pl.Series(
            name="bin", values=["other 3", "b", "other 3"] * 3 + ["other 3"]
        )
        f_binned_expected[0] = None
    else:
        rest_name = "other 2"
        f_binned_expected = pl.Series(
            name="bin", values=["a", "b", "other 2"] * 3 + ["other 2"]
        )
    if feature_type == "cat":
        feature = feature.cast(pl.Categorical)
        f_binned_expected = f_binned_expected.cast(pl.Categorical)
    elif feature_type == "enum":
        feature = feature.cast(pl.Enum(["b", "a", "c", "e"]))
        f_binned_expected = f_binned_expected.cast(
            pl.Enum(["b", "a", "c", "e", rest_name])
        )
    n_obs = len(feature)

    if n_bins > 4:
        f_binned_expected = pl.Series(name="bin", values=feature)

    feature, n_bins_, f_binned = bin_feature(
        feature=feature,
        feature_name=feature.name,
        n_obs=n_obs,
        n_bins=n_bins,
        bin_method="uniform",
    )

    assert isinstance(feature, pl.Series)
    if n_bins > 3:
        assert n_bins_ == 4 + with_null
    else:
        assert n_bins_ == n_bins
    assert_series_equal(f_binned["bin"], f_binned_expected)


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
@pytest.mark.parametrize("with_null", [False, True])
@pytest.mark.parametrize(
    ("feature_type", "with_inf"),
    [
        (pl.Float64, False),
        (pl.Float32, True),
        (pl.Int32, False),
        (pl.UInt16, False),
    ],
)
def test_binning_numerical(bin_method, with_inf, with_null, feature_type):
    """Test that binning works for numerical features."""
    n_bins = 5
    feature = pl.Series(name="my_feature", values=np.arange(100), dtype=feature_type)
    if with_null:
        feature[10] = None  # somewhere in the middle
        assert feature.has_nulls()
    if with_inf:
        feature[11] = np.inf  # somewhere in the middle
        feature[12] = -np.inf  # somewhere in the middle
    n_obs = len(feature)

    feature, n_bins_, f_binned = bin_feature(
        feature=feature,
        feature_name=feature.name,
        n_obs=n_obs,
        n_bins=n_bins,
        bin_method=bin_method,
    )

    if bin_method in ("quantile", "uniform"):
        assert n_bins_ == n_bins
    assert isinstance(feature, pl.Series)
    assert isinstance(f_binned, pl.DataFrame)
    assert f_binned.schema == pl.Schema(
        {"bin": feature_type, "bin_edges": pl.Array(pl.Float64, shape=(2,))}
    )
    assert f_binned["bin"].max() + 1 + with_null == f_binned["bin"].n_unique()
    if bin_method in ("quantile", "uniform"):
        assert f_binned["bin"].n_unique() == n_bins

    if parse(version("polars")) >= Version("1.20.0"):
        # It might already work for earlier polars versions.
        df = f_binned.unique().sort("bin")
        if with_null:
            assert df[0, 0] is None
            assert df[0, 1] is None
            assert df[1, 0] == 0.0
            assert df[1, 1][0] == (-np.inf if with_inf else 0)
            assert df[1, 1][1] > 0  # no zero-length interval
        else:
            assert df[0, 0] == 0.0
            assert df[0, 1][0] == (-np.inf if with_inf else 0)
            assert df[0, 1][1] > 0  # no zero-length interval
        if bin_method in ("quantile", "uniform"):
            assert df[-1, 0] == n_bins - 1 - with_null
        assert df[-1, 1][0] < 99  # no zero-length interval
        assert df[-1, 1][1] == np.inf if with_inf else 99


def test_binning_auto():
    """Test auto bin method"""
    n_bins = 5
    feature = pl.Series(name="my_feature", values=[0, 1, 1, 2])
    n_obs = len(feature)

    feature, n_bins_, f_binned = bin_feature(
        feature=feature,
        feature_name=feature.name,
        n_obs=n_obs,
        n_bins=n_bins,
        bin_method="auto",
    )
    if parse(version("numpy")) >= Version("2.1.0"):
        # https://numpy.org/doc/stable/release/2.1.0-notes.html#histogram-auto-binning-now-returns-bin-sizes-1-for-integer-input-data
        assert n_bins_ == 2
    else:
        assert n_bins_ == 4


def test_binning_strings_with_rest():
    """Test what happens if 'other n' is already taken."""
    n_bins = 3
    feature = pl.Series(name="my_feature", values=["a", "other 2"] * 5 + ["c", "d"] * 2)
    f_binned_expected = pl.Series(
        name="bin", values=["a", "other 2"] * 5 + ["_other 2"] * 4
    )
    n_obs = len(feature)

    feature, n_bins_, f_binned = bin_feature(
        feature=feature,
        feature_name=feature.name,
        n_obs=n_obs,
        n_bins=n_bins,
    )

    assert_series_equal(f_binned["bin"], f_binned_expected)
