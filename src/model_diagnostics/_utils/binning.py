import warnings
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
import polars as pl
from packaging.version import Version

from model_diagnostics import polars_version
from model_diagnostics._utils.array import array_name, validate_same_first_dimension


def bin_feature(
    feature: Optional[Union[npt.ArrayLike, pl.Series]],
    y_obs: npt.ArrayLike,
    n_bins: int = 10,
):
    """Helper function to bin features of different dtypes.

    Best call this function inside a `with pl.StringCache()` context manager.

    Parameters
    ----------
    feature : array-like of shape (n_obs) or None
        Some feature column.
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        Only needed for validation of dimensions of feature.
    n_bins : int
        The number of bins for numerical features and the maximal number of (most
        frequent) categories shown for categorical features. Due to ties, the effective
        number of bins might be smaller than `n_bins`. Null values are always included
        in the output, accounting for one bin. NaN values are treated as null values.

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
    n_bins: int
        Effective number of bins.
    f_binned : pl.Series or None
        For a numerical feature the binned/digitized version of it.
    """
    is_categorical = False
    is_string = False
    f_binned = None

    if feature is None:
        feature_name = None
    else:
        feature_name = array_name(feature, default="feature")
        # The following statement, i.e. possibly the creation of a pl.Categorical,
        # MUST be under the StringCache context manager!
        feature = pl.Series(name=feature_name, values=feature)
        validate_same_first_dimension(y_obs, feature)
        if (feature.dtype == pl.Categorical) or (
            polars_version >= Version("0.20.0") and feature.dtype == pl.Enum
        ):
            # FIXME: polars >= 0.20.0
            is_categorical = True
        elif feature.dtype in [pl.Utf8, pl.Object]:
            # We could convert strings to categoricals.
            is_string = True
        # FIXME: polars >= 0.19.14
        # Then, just use Series.dtype.is_float()
        elif (hasattr(feature.dtype, "is_float") and feature.dtype.is_float()) or (
            not hasattr(feature.dtype, "is_float") and feature.is_float()
        ):
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
                # FIXME: polars >= 0.20
                if polars_version >= Version("0.20.0"):
                    count_name = "count"
                else:
                    count_name = "counts"
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
            # We use method="inverted_cdf" (same as "lower") instead of the
            # default "linear" because "linear" produces as many unique values
            # as before.
            # If we have Null values, we should reserve one bin for it and reduce
            # the effective number of bins by 1.
            n_bins_ef = max(1, n_bins - (feature.null_count() >= 1))
            q = np.nanquantile(
                feature,
                # Improved rounding errors by using integers and dividing at the
                # end as opposed to np.linspace with 1/n_bins step size.
                q=np.arange(1, n_bins_ef) / n_bins_ef,
                method="inverted_cdf",
            )
            q = np.unique(q)  # Some quantiles might be the same.
            # We want: bins[i-1] < x <= bins[i]
            f_binned = np.digitize(feature, bins=q, right=True)
            # Now, we insert Null values again at the original places.
            f_binned = (
                pl.LazyFrame(
                    [
                        pl.Series("__f_binned", f_binned, dtype=feature.dtype),
                        feature,
                    ]
                )
                .select(
                    pl.when(pl.col(feature_name).is_null())
                    .then(None)
                    .otherwise(pl.col("__f_binned"))
                    .alias("bin")
                )
                .collect()
                .get_column("bin")
            )
    return feature, feature_name, is_categorical, is_string, n_bins, f_binned
