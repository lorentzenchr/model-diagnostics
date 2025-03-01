import sys
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import polars as pl

from model_diagnostics._utils.array import (
    array_name,
    is_pandas_series,
    length_of_first_dimension,
)


def _format_integer(x):
    """Nicely round and format large integers."""
    # https://stackoverflow.com/a/45846841
    x = float(f"{x:.3g}")
    magnitude = 0
    while abs(x) >= 1000 and magnitude < 4:
        magnitude += 1
        x /= 1000.0
    return "{}{}".format(
        f"{x:f}".rstrip("0").rstrip("."), ["", "k", "M", "G", "T"][magnitude]
    )


def bin_feature(
    feature: Optional[Union[npt.ArrayLike, pl.Series]],
    feature_name: Optional[Union[int, str]],
    n_obs: int,
    n_bins: int = 10,
    bin_method: str = "sturges",
):
    """Helper function to bin features of different dtypes.

    Best call this function inside a `with pl.StringCache()` context manager.

    Parameters
    ----------
    feature : array-like of shape (n_obs)
        Some feature column.
    feature_name : int, str or None
        Name of the feature.
    n_obs : int
        The expected length of the first dimention of feature.
    n_bins : int
        The number of bins, at least 2. For numerical features, `n_bins` only applies
        when `bin_method` is set to `"quantile"` or `"uniform"`.
        For string-like and categorical features, the most frequent values are taken.
        Ties are dealt with by taking the first value in natural sorting order.
        The remaining values are merged into `"other n"` with `n` indicating the unique
        count.

        I present, null values are always included in the output, accounting for one
        bin. NaN values are treated as null values.
    bin_method : str
        The method to use for finding bin edges (boundaries). Options are:

        - `"quantile"`
        - `"uniform"`
        - `"auto"`
        - `"fd"`
        - `"doane"`
        - `"scott"`
        - `"stone"`
        - `"rice"`
        - `"sturges"`
        - `"sqrt"`

    Returns
    -------
    feature : pl.Series or None
        The polars.Series version of the feature.
    n_bins : int
        Effective number of bins.
    f_binned : pl.DataFrame or None
        The binned/digitized version of the feature.
        For numerical features, columns are:

        - `bin`: The bin number.
          Bin `i` is assigned if `bin_edges[i] < feature <= bin_edges[i]`
        - `bin_edges`: edges/thresholds of the bins.

        For other features, columns are:

        - `bin`: The binned version of it, i.e. the many too small values are put
          together as `"other n"` where `n` is the number of unique values it contains.
    """
    is_categorical = False
    is_enum = False
    is_string = False
    f_binned = None

    valid_bin_methods = (
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
    )
    if bin_method not in valid_bin_methods:
        msg = (
            f"Parameter bin_method must be one of {valid_bin_methods};"
            f" got {bin_method}."
        )
        raise ValueError(msg)
    if n_bins < 2:
        msg = f"Parameter n_bins must be at least 2, got {n_bins}."
        raise ValueError(msg)

    default = f"feature {feature_name}" if isinstance(feature_name, int) else "feature"
    feature_name = array_name(feature, default=default)
    # The following statement, i.e. possibly the creation of a pl.Categorical,
    # MUST be under the StringCache context manager!
    try:
        feature = pl.Series(name=feature_name, values=feature)
    except ImportError:
        # FIXME: pyarrow not installed
        # For non numpy-backed columns, pyarrow is needed. Here we handle the case
        # where pyarrow is not installed and such a pandas extention array is
        # passed, e.g. with CategoricalDtype.
        if is_pandas_series(feature):
            pandas = sys.modules["pandas"]
            is_pandas_categorical = isinstance(
                feature.dtype,  # type: ignore
                pandas.CategoricalDtype,
            )
            feature = pl.from_dataframe(
                feature.to_frame(name=feature_name)  # type: ignore
            )[:, 0]
            if is_pandas_categorical and isinstance(feature.dtype, pl.Enum):
                # Pandas categoricals usually get mapped to polars categoricals.
                # But this code path gives pl.Enum.
                feature = feature.cast(pl.Categorical)
        else:
            raise  # re-raises the ImportError
    if length_of_first_dimension(feature) != n_obs:
        msg = (
            f"The feature array {feature_name} does not have length {n_obs} of its"
            " first dimension."
        )
        raise ValueError(msg)
    if feature.dtype == pl.Categorical:
        is_categorical = True
    elif feature.dtype == pl.Enum:
        is_enum = True
    elif feature.dtype in [pl.Utf8, pl.Object]:
        # We could convert strings to categoricals.
        is_string = True
    elif feature.dtype.is_float():
        # We treat NaN as Null values, numpy will see a Null as a NaN.
        feature = feature.fill_nan(None)
    else:
        # integers
        pass

    # If we have Null values, we should reserve one bin for it and reduce
    # the effective number of bins by 1.
    n_bins_ef = max(1, n_bins - feature.has_nulls())

    if is_categorical or is_enum or is_string:
        # For categorical and string features, knowing the frequency table in
        # advance makes life easier in order to make results consistent.
        # Consider (no null values)
        #     feature  count
        #         "a"      3
        #         "b"      2
        #         "c"      2
        #         "d"      1
        # with n_bins = 3. As we want the effective number of bins to be at most
        # n_bins, we want, in the above case, only "a" and "b" in the final result. All
        # the others a put into the second bin and called "other 2" because it comprises
        # 2 unique features values (c, d). Ties are dealt with by sorting.

        # value_counts(sort=True) sorts ties by first occurence, we want
        # alphanumerical sorting order.
        value_counts = (
            feature.drop_nulls()
            .value_counts()
            .sort(by=["count", feature_name], descending=[True, False])
        )

        if n_bins_ef >= value_counts.shape[0]:
            # This also covers the case of only null values.
            n_bins_ef = value_counts.shape[0]
            f_binned = pl.DataFrame({"bin": feature})
        else:
            # We keep the n_bins_ef - 1 most frequent values. Ties are resolved by
            # taking the first one of the sorted values (most often alpha-numerical).
            if feature.has_nulls():
                # To ease adding the null value, we take one value more.
                keep_values = value_counts[feature_name].head(n_bins_ef)
                if is_categorical:
                    # FIXME: Workaround for https://github.com/pola-rs/polars/issues/21175
                    keep_values = keep_values.cast(pl.String)
                    keep_values[-1] = None
                    keep_values = keep_values.cast(pl.Categorical)
                else:
                    keep_values[-1] = None
            else:
                keep_values = value_counts[feature_name].head(n_bins_ef - 1)
            # Number of feature values to put into one bin, called "other n",
            # n = n_remaining.
            n_remaining = value_counts.shape[0] - (n_bins_ef - 1)
            remaining_name = "other " + _format_integer(n_remaining)
            while remaining_name in keep_values:
                remaining_name = "_" + remaining_name
            return_dtype = feature.dtype
            if is_enum:
                return_dtype = pl.Enum(
                    pl.concat([feature.dtype.categories, pl.Series([remaining_name])])
                )
            f_binned = feature.replace_strict(
                old=keep_values,
                new=keep_values,
                default=remaining_name,
                return_dtype=return_dtype,
            )
            f_binned = pl.DataFrame({"bin": f_binned})
    else:
        # Binning a numerical feature
        # We will need min and max anyway.
        feature_min, feature_max = feature.min(), feature.max()
        if feature_min == -np.inf:
            finite_min = feature.filter(feature > -np.inf).min()
        else:
            finite_min = feature_min
        if feature_max == np.inf:
            finite_max = feature.filter(feature < np.inf).max()
        else:
            finite_max = feature_max
        f_range = finite_max - finite_min

        if bin_method == "quantile":
            # We use method="inverted_cdf" instead of the default "linear" because
            # "linear" produces as many unique values as before.
            q = np.nanquantile(
                feature,
                # Improved rounding errors by using integers and dividing at the
                # end as opposed to np.linspace with 1/n_bins step size.
                q=np.arange(1, n_bins_ef) / n_bins_ef,
                method="inverted_cdf",
            )
            bin_edges = np.unique(q)  # Some quantiles might be the same.
        elif bin_method == "uniform":
            bin_edges = finite_min + f_range * np.arange(1, n_bins_ef) / n_bins_ef
        else:
            # numpy histogram bin methods
            a = feature.filter(feature.is_finite() & feature.is_not_null())
            bin_edges = np.histogram_bin_edges(a, bins=bin_method)[1:-1]
            n_bins_ef = bin_edges.shape[0] + 1
        # We want: bins[i-1] < x <= bins[i]
        f_binned = np.digitize(feature, bins=bin_edges, right=True)
        # The full bin edges also include min and max of the feature.
        if bin_edges.size == 0:
            bin_edges = np.r_[feature_min, feature_max]
        else:
            bin_edges = np.r_[feature_min, bin_edges, feature_max]
        # This is quite a hack with numpy strides and views. We want to accomplish
        # bin_edges = [[value0, value1], [value1, value2], [value2, value3], ..]
        bin_edges = np.lib.stride_tricks.as_strided(
            bin_edges, (bin_edges.shape[0] - 1, 2), bin_edges.strides * 2
        )
        # Back to the binned feature.
        # Now, we insert Null values again at the original places.
        f_binned = (
            pl.LazyFrame(
                [
                    feature,
                    pl.Series("__f_binned", f_binned, dtype=feature.dtype),
                    pl.Series(
                        "__bin_edges",
                        bin_edges[f_binned],
                        dtype=pl.Array(pl.Float64, 2),
                    ),
                ]
            )
            .select(
                pl.when(pl.col(feature_name).is_null())
                .then(None)
                .otherwise(pl.col("__f_binned"))
                .alias("bin"),
                pl.when(pl.col(feature_name).is_null())
                .then(None)
                .otherwise(pl.col("__bin_edges"))
                .alias("bin_edges"),
            )
            .collect()
        )
    return feature, n_bins_ef + feature.has_nulls(), f_binned
