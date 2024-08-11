import sys
import warnings
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import polars as pl

from model_diagnostics._utils.array import (
    array_name,
    is_pandas_series,
    validate_same_first_dimension,
)


def bin_feature(
    feature: Optional[Union[npt.ArrayLike, pl.Series]],
    feature_name: Optional[Union[int, str]],
    y_obs: npt.ArrayLike,
    n_bins: int = 10,
    bin_method: str = "quantile",
):
    """Helper function to bin features of different dtypes.

    Best call this function inside a `with pl.StringCache()` context manager.

    Parameters
    ----------
    feature : array-like of shape (n_obs) or None
        Some feature column.
    feature_name : int, str or None
        Name of the feature.
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        Only needed for validation of dimensions of feature.
    n_bins : int
        The number of bins for numerical features and the maximal number of (most
        frequent) categories shown for categorical features. Due to ties, the effective
        number of bins might be smaller than `n_bins`. Null values are always included
        in the output, accounting for one bin. NaN values are treated as null values.
    bin_method : str
        The method to use for finding bin edges (boundaries). Options are:

        - "quantile"
        - "uniform"

    Returns
    -------
    feature : pl.Series or None
        The polars.Series version of the feature.
    feature_name : str
        The name of the feature.
    is_categorical : bool
        True if feature is categorical or enum.
    is_string : bool
        True if feature is a string type.
    n_bins : int
        Effective number of bins.
    f_binned : pl.DataFrame or None
        For a numerical feature the binned/digitized version of it.
        Columns are:

        - `bin`: The bin number.
        - `bin_edges`: edges/thresholds of the bins.
    """
    is_categorical = False
    is_string = False
    f_binned = None

    if bin_method not in ("quantile", "uniform"):
        msg = (
            "Parameter bin_method must be either 'quantile' or ''uniform';"
            f" got {bin_method}."
        )
        raise ValueError(msg)

    if feature is None:
        # TODO: Remove this branch.
        feature_name = None
    else:
        if isinstance(feature_name, int):
            default = f"feature {feature_name}"
        else:
            default = "feature"
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
                is_pandas_categorical = pandas.api.types.is_categorical_dtype(feature)
                # dataframe = feature.to_frame(name="0").__dataframe__()
                # feature = pl.from_dataframe(dataframe)[:, 0]
                feature = pl.from_dataframe(feature.to_frame(name="0"))[:, 0]  # type: ignore
                if is_pandas_categorical and isinstance(feature.dtype, pl.Enum):
                    # Pandas categoricals usually get mapped to polars categoricals.
                    # But this code path gives pl.Enum.
                    feature = feature.cast(pl.Categorical)
            else:
                raise  # re-raises the ImportError
        validate_same_first_dimension(y_obs, feature)
        if feature.dtype in [pl.Categorical, pl.Enum]:
            is_categorical = True
        elif feature.dtype in [pl.Utf8, pl.Object]:
            # We could convert strings to categoricals.
            is_string = True
        elif feature.dtype.is_float():
            # We treat NaN as Null values, numpy will see a Null as a NaN.
            feature = feature.fill_nan(None)
        else:
            # integers
            pass

        if is_categorical or is_string:
            # For categorical and string features, knowing the frequency table in
            # advance makes life easier in order to make results consistent.
            # Consider
            #     feature  count
            #         "a"      3
            #         "b"      2
            #         "c"      2
            #         "d"      1
            # with n_bins = 2. As we want the effective number of bins to be at
            # most n_bins, we want, in the above case, only "a" in the final
            # result. Therefore, we need to internally decrease n_bins to 1.
            if feature.null_count() == 0:
                value_count = feature.value_counts(sort=True)
                n_bins_ef = n_bins
            else:
                value_count = feature.drop_nulls().value_counts(sort=True)
                n_bins_ef = n_bins - 1

            if n_bins_ef >= value_count.shape[0]:
                n_bins = value_count.shape[0]
            else:
                count_name = "count"
                n = value_count[count_name][n_bins_ef]
                n_bins_tmp = value_count.filter(pl.col(count_name) >= n).shape[0]
                if n_bins_tmp > n_bins_ef:
                    n_bins = value_count.filter(pl.col(count_name) > n).shape[0]
                else:
                    n_bins = n_bins_tmp

            if feature.null_count() >= 1:
                n_bins += 1

            if n_bins == 0:
                msg = (
                    "Due to ties, the effective number of bins is 0. "
                    f"Consider to increase n_bins>={n_bins_tmp}."
                )
                warnings.warn(msg, UserWarning, stacklevel=2)
        else:
            # Binning
            # If we have Null values, we should reserve one bin for it and reduce
            # the effective number of bins by 1.
            n_bins_ef = max(1, n_bins - (feature.null_count() >= 1))
            # We will need min and max anyway.
            feature_min, feature_max = feature.min(), feature.max()
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
            else:
                # Uniform
                f_range = feature_max - feature_min
                bin_edges = feature_min + f_range * np.arange(1, n_bins_ef) / n_bins_ef
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
    return feature, feature_name, is_categorical, is_string, n_bins, f_binned
