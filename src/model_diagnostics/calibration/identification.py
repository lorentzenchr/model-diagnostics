from itertools import chain
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import special

from model_diagnostics._utils.array import (
    get_second_dimension,
    get_sorted_array_names,
    length_of_first_dimension,
    length_of_second_dimension,
    validate_2_arrays,
    validate_same_first_dimension,
)
from model_diagnostics._utils.binning import bin_feature
from model_diagnostics._utils.partial_dependence import compute_partial_dependence


def identification_function(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    *,
    functional: str = "mean",
    level: float = 0.5,
) -> np.ndarray:
    r"""Canonical identification function.

    Identification functions act as generalised residuals. See [Notes](#notes) for
    further details.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs)
        Predicted values of the `functional`, e.g. the conditional expectation of
        the response, `E(Y|X)`.
    functional : str
        The functional that is induced by the identification function `V`. Options are:

        - `"mean"`. Argument `level` is neglected.
        - `"median"`. Argument `level` is neglected.
        - `"expectile"`
        - `"quantile"`
    level : float
        The level of the expectile of quantile. (Often called \(\alpha\).)
        It must be `0 < level < 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.

    Returns
    -------
    V : ndarray of shape (n_obs)
        Values of the identification function.

    Notes
    -----
    [](){#notes}
    The function \(V(y, z)\) for observation \(y=y_{pred}\) and prediction
    \(z=y_{pred}\) is a strict identification function for the functional \(T\), or
    induces the functional \(T\) as:

    \[
    \mathbb{E}[V(Y, z)] = 0\quad \Leftrightarrow\quad z\in T(F) \quad \forall
    \text{ distributions } F
    \in \mathcal{F}
    \]

    for some class of distributions \(\mathcal{F}\). Implemented examples of the
    functional \(T\) are mean, median, expectiles and quantiles.

    | functional | strict identification function \(V(y, z)\)           |
    | ---------- | ---------------------------------------------------- |
    | mean       | \(z - y\)                                            |
    | median     | \(\mathbf{1}\{z \ge y\} - \frac{1}{2}\)              |
    | expectile  | \(2 \mid\mathbf{1}\{z \ge y\} - \alpha\mid (z - y)\) |
    | quantile   | \(\mathbf{1}\{z \ge y\} - \alpha\)                   |

    For `level` \(\alpha\).

    References
    ----------
    `[Gneiting2011]`

    :   T. Gneiting.
        "Making and Evaluating Point Forecasts". (2011)
        [doi:10.1198/jasa.2011.r10138](https://doi.org/10.1198/jasa.2011.r10138)
        [arxiv:0912.0902](https://arxiv.org/abs/0912.0902)

    Examples
    --------
    >>> identification_function(y_obs=[0, 0, 1, 1], y_pred=[-1, 1, 1 , 2])
    array([-1,  1,  0,  1])
    """
    y_o: np.ndarray
    y_p: np.ndarray
    y_o, y_p = validate_2_arrays(y_obs, y_pred)

    if functional in ("expectile", "quantile") and (level <= 0 or level >= 1):
        msg = f"Argument level must fulfil 0 < level < 1, got {level}."
        raise ValueError(msg)

    if functional == "mean":
        return y_p - y_o
    elif functional == "median":
        return np.greater_equal(y_p, y_o) - 0.5
    elif functional == "expectile":
        return 2 * np.abs(np.greater_equal(y_p, y_o) - level) * (y_p - y_o)
    elif functional == "quantile":
        return np.greater_equal(y_p, y_o) - level
    else:
        allowed_functionals = ("mean", "median", "expectile", "quantile")
        msg = (
            f"Argument functional must be one of {allowed_functionals}, got "
            f"{functional}."
        )
        raise ValueError(msg)


def compute_bias(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    feature: Optional[Union[npt.ArrayLike, pl.Series]] = None,
    weights: Optional[npt.ArrayLike] = None,
    *,
    functional: str = "mean",
    level: float = 0.5,
    n_bins: int = 10,
    bin_method: str = "quantile",
):
    r"""Compute generalised bias conditional on a feature.

    This function computes and aggregates the generalised bias, i.e. the values of the
    canonical identification function, versus (grouped by) a feature.
    This is a good way to assess whether a model is conditionally calibrated or not.
    Well calibrated models have bias terms around zero.
    For the mean functional, the generalised bias is the negative residual
    `y_pred - y_obs`.
    See [Notes](#notes) for further details.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs) or (n_obs, n_models)
        Predicted values, e.g. for the conditional expectation of the response,
        `E(Y|X)`.
    feature : array-like of shape (n_obs) or None
        Some feature column.
    weights : array-like of shape (n_obs) or None
        Case weights. If given, the bias is calculated as weighted average of the
        identification function with these weights.
        Note that the standard errors and p-values in the output are based on the
        assumption that the variance of the bias is inverse proportional to the
        weights. See the Notes section for details.
    functional : str
        The functional that is induced by the identification function `V`. Options are:

        - `"mean"`. Argument `level` is neglected.
        - `"median"`. Argument `level` is neglected.
        - `"expectile"`
        - `"quantile"`
    level : float
        The level of the expectile of quantile. (Often called \(\alpha\).)
        It must be `0 < level < 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.
    n_bins : int
        The number of bins for numerical features and the maximal number of (most
        frequent) categories shown for categorical features. Due to ties, the effective
        number of bins might be smaller than `n_bins`. Null values are always included
        in the output, accounting for one bin. NaN values are treated as null values.
    bin_method : str
        The method to use for finding bin edges (boundaries). Options are:

        - `"quantile"`
        - `"uniform"`

    Returns
    -------
    df : polars.DataFrame
        The result table contains at least the columns:

        - `bias_mean`: Mean of the bias
        - `bias_cout`: Number of data rows
        - `bias_weights`: Sum of weights
        - `bias_stderr`: Standard error, i.e. standard deviation of `bias_mean`
        - `p_value`: p-value of the 2-sided t-test with null hypothesis:
          `bias_mean = 0`

        If `feautre ` is not None, then there is also the column:

        - `feature_name`: The actual name of the feature with the (binned) feature
          values.

    Notes
    -----
    [](){#notes}
    A model \(m(X)\) is conditionally calibrated iff
    \(\mathbb{E}(V(m(X), Y)|X)=0\) almost surely with canonical identification
    function \(V\).
    The empirical version, given some data, reads
    \(\bar{V} = \frac{1}{n}\sum_i \phi(x_i) V(m(x_i), y_i)\) with a test function
    \(\phi(x_i)\) that projects on the specified feature.
    For a feature with only two distinct values `"a"` and `"b"`, this becomes
    \(\bar{V} = \frac{1}{n_a}\sum_{i \text{ with }x_i=a} V(m(a), y_i)\) with
    \(n_a=\sum_{i \text{ with }x_i=a}\) and similar for `"b"`.
    With case weights, this reads
    \(\bar{V} = \frac{1}{\sum_i w_i}\sum_i w_i \phi(x_i) V(m(x_i), y_i)\).
    This generalises the classical residual (up to a minus sign) for target functionals
    other than the mean. See `[FLM2022]`.

    The standard error for \(\bar{V}\) is calculated in the standard way as
    \(\mathrm{SE} = \sqrt{\operatorname{Var}(\bar{V})} = \frac{\sigma}{\sqrt{n}}\) and
    the standard variance estimator for \(\sigma^2 = \operatorname{Var}(\phi(x_i)
    V(m(x_i), y_i))\) with Bessel correction, i.e. division by \(n-1\) instead of
    \(n\).

    With case weights, the variance estimator becomes \(\operatorname{Var}(\bar{V})
    = \frac{1}{n-1} \frac{1}{\sum_i w_i} \sum_i w_i (V(m(x_i), y_i) - \bar{V})^2\) with
    the implied relation \(\operatorname{Var}(V(m(x_i), y_i)) \sim \frac{1}{w_i} \).
    If your weights are for repeated observations, so-called frequency weights, then
    the above estimate is conservative because it uses \(n - 1\) instead
    of \((\sum_i w_i) - 1\).

    References
    ----------
    `[FLM2022]`

    :   T. Fissler, C. Lorentzen, and M. Mayer.
        "Model Comparison and Calibration Assessment". (2022)
        [arxiv:2202.12780](https://arxiv.org/abs/2202.12780).

    Examples
    --------
    >>> compute_bias(y_obs=[0, 0, 1, 1], y_pred=[-1, 1, 1 , 2])
    shape: (1, 5)
    ┌───────────┬────────────┬──────────────┬─────────────┬──────────┐
    │ bias_mean ┆ bias_count ┆ bias_weights ┆ bias_stderr ┆ p_value  │
    │ ---       ┆ ---        ┆ ---          ┆ ---         ┆ ---      │
    │ f64       ┆ u32        ┆ f64          ┆ f64         ┆ f64      │
    ╞═══════════╪════════════╪══════════════╪═════════════╪══════════╡
    │ 0.25      ┆ 4          ┆ 4.0          ┆ 0.478714    ┆ 0.637618 │
    └───────────┴────────────┴──────────────┴─────────────┴──────────┘
    >>> compute_bias(y_obs=[0, 0, 1, 1], y_pred=[-1, 1, 1 , 2],
    ... feature=["a", "a", "b", "b"])
    shape: (2, 6)
    ┌─────────┬───────────┬────────────┬──────────────┬─────────────┬─────────┐
    │ feature ┆ bias_mean ┆ bias_count ┆ bias_weights ┆ bias_stderr ┆ p_value │
    │ ---     ┆ ---       ┆ ---        ┆ ---          ┆ ---         ┆ ---     │
    │ str     ┆ f64       ┆ u32        ┆ f64          ┆ f64         ┆ f64     │
    ╞═════════╪═══════════╪════════════╪══════════════╪═════════════╪═════════╡
    │ a       ┆ 0.0       ┆ 2          ┆ 2.0          ┆ 1.0         ┆ 1.0     │
    │ b       ┆ 0.5       ┆ 2          ┆ 2.0          ┆ 0.5         ┆ 0.5     │
    └─────────┴───────────┴────────────┴──────────────┴─────────────┴─────────┘
    """
    validate_same_first_dimension(y_obs, y_pred)
    n_pred = length_of_second_dimension(y_pred)
    pred_names, _ = get_sorted_array_names(y_pred)

    if weights is not None:
        validate_same_first_dimension(weights, y_obs)
        w = np.asarray(weights)
        if w.ndim > 1:
            msg = f"The array weights must be 1-dimensional, got weights.ndim={w.ndim}."
            raise ValueError(msg)
    else:
        w = np.ones_like(y_obs, dtype=float)

    n_obs = length_of_first_dimension(y_pred)
    df_list = []
    with pl.StringCache():
        feature_name = None
        if feature is not None:
            feature, n_bins, f_binned = bin_feature(
                feature=feature,
                feature_name=None,
                n_obs=n_obs,
                n_bins=n_bins,
                bin_method=bin_method,
            )
            feature_name = feature.name
            is_cat_or_string = feature.dtype in [
                pl.Categorical,
                pl.Enum,
                pl.Utf8,
                pl.Object,
            ]

        for i in range(len(pred_names)):
            # Loop over columns of y_pred.
            x = y_pred if n_pred == 0 else get_second_dimension(y_pred, i)

            bias = identification_function(
                y_obs=y_obs,
                y_pred=x,
                functional=functional,
                level=level,
            )

            if feature is None:
                bias_mean = np.average(bias, weights=w)
                bias_weights = np.sum(w)
                bias_count = bias.shape[0]
                # Note: with Bessel correction
                bias_stddev = np.average((bias - bias_mean) ** 2, weights=w) / np.amax(
                    [1, bias_count - 1]
                )
                df = pl.DataFrame(
                    {
                        "bias_mean": [bias_mean],
                        "bias_count": pl.Series([bias_count], dtype=pl.UInt32),
                        "bias_weights": [bias_weights],
                        "bias_stderr": [np.sqrt(bias_stddev)],
                    }
                )
            else:
                df = pl.DataFrame(
                    {
                        "y_obs": y_obs,
                        "y_pred": x,
                        feature_name: feature,
                        "bias": bias,
                        "weights": w,
                    }
                )

                agg_list = [
                    pl.col("bias_mean").first(),
                    pl.count("bias").alias("bias_count"),
                    pl.col("weights").sum().alias("bias_weights"),
                    (
                        (
                            pl.col("weights")
                            * ((pl.col("bias") - pl.col("bias_mean")) ** 2)
                        ).sum()
                        / pl.col("weights").sum()
                    ).alias("variance"),
                ]

                if is_cat_or_string:
                    groupby_name = feature_name
                else:
                    df = df.hstack([f_binned.get_column("bin")])
                    groupby_name = "bin"
                    agg_list.append(pl.col(feature_name).mean())

                df = (
                    df.lazy()
                    .select(
                        pl.all(),
                        (
                            (pl.col("weights") * pl.col("bias"))
                            .sum()
                            .over(groupby_name)
                            / pl.col("weights").sum().over(groupby_name)
                        ).alias("bias_mean"),
                    )
                    .group_by(groupby_name)
                    .agg(agg_list)
                    .with_columns(
                        [
                            pl.when(pl.col("bias_count") > 1)
                            .then(pl.col("variance") / (pl.col("bias_count") - 1))
                            .otherwise(pl.col("variance"))
                            .sqrt()
                            .alias("bias_stderr"),
                        ]
                    )
                    # With sort and head alone, we could lose the null value, but we
                    # want to keep it.
                    # .sort("bias_count", descending=True)
                    # .head(n_bins)
                    .with_columns(
                        pl.when(pl.col(feature_name).is_null())
                        .then(pl.max("bias_count") + 1)
                        .otherwise(pl.col("bias_count"))
                        .alias("__priority")
                    )
                    .sort("__priority", descending=True)
                    .head(n_bins)
                )
                # FIXME: When n_bins=0, the result should be an empty dataframe
                # (0 rows and some columns). For some unknown reason as of
                # polars 0.20.20, the following sort neglects the head(0) statement.
                # Therefore, we place an explicit collect here. This should not be
                # needed!
                if n_bins == 0 or feature.null_count() >= 1:
                    df = df.collect().lazy()
                df = df.sort(feature_name, descending=False)

                df = df.select(
                    [
                        pl.col(feature_name),
                        pl.col("bias_mean"),
                        pl.col("bias_count"),
                        pl.col("bias_weights"),
                        pl.col("bias_stderr"),
                    ]
                ).collect()

            # Add column with p-value of 2-sided t-test.
            # We explicitly convert "to_numpy", because otherwise we get:
            #   RuntimeWarning: A builtin ctypes object gave a PEP3118 format string
            #   that does not match its itemsize, so a best-guess will be made of the
            #   data type. Newer versions of python may behave correctly.
            stderr_ = df.get_column("bias_stderr")
            p_value = np.full_like(stderr_, fill_value=np.nan)
            n = df.get_column("bias_count")
            p_value[np.asarray((n > 1) & (stderr_ == 0), dtype=bool)] = 0
            mask = stderr_ > 0
            x = df.get_column("bias_mean").filter(mask).to_numpy()
            n = df.get_column("bias_count").filter(mask).to_numpy()
            stderr = stderr_.filter(mask).to_numpy()
            # t-statistic t (-|t| and factor of 2 because of 2-sided test)
            p_value[np.asarray(mask, dtype=bool)] = 2 * special.stdtr(
                n - 1,  # degrees of freedom
                -np.abs(x / stderr),
            )
            df = df.with_columns(pl.Series("p_value", p_value))

            # Add column "model".
            if n_pred > 0:
                model_col_name = "model_" if feature_name == "model" else "model"
                df = df.with_columns(
                    pl.Series(model_col_name, [pred_names[i]] * df.shape[0])
                )

            # Select the columns in the correct order.
            col_selection = []
            if n_pred > 0:
                col_selection.append(model_col_name)
            if feature_name is not None and feature_name in df.columns:
                col_selection.append(feature_name)
            col_selection += [
                "bias_mean",
                "bias_count",
                "bias_weights",
                "bias_stderr",
                "p_value",
            ]
            df_list.append(df.select(col_selection))

        df = pl.concat(df_list)
    return df


def compute_marginal(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    X: Optional[npt.ArrayLike] = None,
    feature_name: Optional[Union[str, int]] = None,
    predict_function: Optional[Callable] = None,
    weights: Optional[npt.ArrayLike] = None,
    *,
    n_bins: int = 10,
    bin_method: str = "uniform",
    n_max: int = 1000,
    rng: Optional[Union[np.random.Generator, int]] = None,
):
    r"""Compute the marginal expectation conditional on a single feature.

    This function computes the (weighted) average of observed response and predictions
    conditional on a given feature.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs) or (n_obs, n_models)
        Predicted values, e.g. for the conditional expectation of the response,
        `E(Y|X)`.
    X : array-like of shape (n_obs, n_features) or None
        The dataframe or array of features to be passed to the model predict function.
    feature_name : int, str or None
        Column name (str) or index (int) of feature in `X`. If None, the total marginal
        is computed.
    predict_function : callable or None
        A callable to get prediction, i.e. `predict_function(X)`. Used to compute
        partial dependence. If `None`, partial dependence is omitted.
    weights : array-like of shape (n_obs) or None
        Case weights. If given, the bias is calculated as weighted average of the
        identification function with these weights.
    n_bins : int
        The number of bins for numerical features and the maximal number of (most
        frequent) categories shown for categorical features. Due to ties, the effective
        number of bins might be smaller than `n_bins`. Null values are always included
        in the output, accounting for one bin. NaN values are treated as null values.
    bin_method : str
        The method to use for finding bin edges (boundaries). Options are:

        - `"quantile"`
        - `"uniform"`
    n_max : int or None
        Used only for partial dependence computation. The number of rows to subsample
        from X. This speeds up computation, in particular for slow predict functions.
    rng : np.random.Generator, int or None
        Used only for partial dependence computation. The random number generator used
        for subsampling of `n_max` rows. The input is internally wrapped by
        `np.random.default_rng(rng)`.

    Returns
    -------
    df : polars.DataFrame
        The result table contains at least the columns:

        - `y_obs_mean`: Mean of `y_obs`
        - `y_pred_mean`: Mean of `y_pred`
        - `y_obs_stderr`: Standard error, i.e. standard deviation of `y_obs_mean`
        - `y_pred_stderr`: Standard error, i.e. standard deviation of `y_pred_mean`
        - `count`: Number of data rows
        - `weights`: Sum of weights

        If `feature ` is not None, then there is also the column:

        - `feature_name`: The actual name of the feature with the (binned) feature
          values.

        If `feature` is numerical, one also has:

        - `bin_edges`: The edges and standard deviation of the bins, i.e.
          (min, std, max).

    Notes
    -----
    The marginal values are computed as an estimation of:

    - `y_obs`: \(\mathbb{E}(Y|feature)\)
    - `y_pred`: \(\mathbb{E}(m(X)|feature)\)

    with \(feature\) the column specified by `feature_name`.
    Computationally that is more or less a group-by-aggregate operation on a dataset.

    The standard error for both are calculated in the standard way as
    \(\mathrm{SE} = \sqrt{\operatorname{Var}(\bar{Y})} = \frac{\sigma}{\sqrt{n}}\) and
    the standard variance estimator for \(\sigma^2\) with Bessel correction, i.e.
    division by \(n-1\) instead of \(n\).

    With case weights, the variance estimator becomes \(\operatorname{Var}(\bar{Y})
    = \frac{1}{n-1} \frac{1}{\sum_i w_i} \sum_i w_i (y_i - \bar{y})^2\) with
    the implied relation \(\operatorname{Var}(y_i) \sim \frac{1}{w_i} \).
    If your weights are for repeated observations, so-called frequency weights, then
    the above estimate is conservative because it uses \(n - 1\) instead
    of \((\sum_i w_i) - 1\).

    Examples
    --------
    >>> compute_marginal(y_obs=[0, 0, 1, 1], y_pred=[-1, 1, 1, 2])
    shape: (1, 6)
    ┌────────────┬─────────────┬──────────────┬───────────────┬───────┬─────────┐
    │ y_obs_mean ┆ y_pred_mean ┆ y_obs_stderr ┆ y_pred_stderr ┆ count ┆ weights │
    │ ---        ┆ ---         ┆ ---          ┆ ---           ┆ ---   ┆ ---     │
    │ f64        ┆ f64         ┆ f64          ┆ f64           ┆ u32   ┆ f64     │
    ╞════════════╪═════════════╪══════════════╪═══════════════╪═══════╪═════════╡
    │ 0.5        ┆ 0.75        ┆ 0.288675     ┆ 0.629153      ┆ 4     ┆ 4.0     │
    └────────────┴─────────────┴──────────────┴───────────────┴───────┴─────────┘
    >>> import polars as pl
    >>> from sklearn.linear_model import Ridge
    >>> pl.Config.set_tbl_width_chars(84)  # doctest: +ELLIPSIS
    <class 'polars.config.Config'>
    >>> y_obs, X =[0, 0, 1, 1], [[0, 1], [1, 1], [1, 2], [2, 2]]
    >>> m = Ridge().fit(X, y_obs)
    >>> compute_marginal(y_obs=y_obs, y_pred=m.predict(X), X=X, feature_name=0,
    ... predict_function=m.predict)
    shape: (3, 9)
    ┌──────────┬─────────┬─────────┬─────────┬───┬───────┬─────────┬─────────┬─────────┐
    │ feature  ┆ y_obs_m ┆ y_pred_ ┆ y_obs_s ┆ … ┆ count ┆ weights ┆ bin_edg ┆ partial │
    │ 0        ┆ ean     ┆ mean    ┆ tderr   ┆   ┆ ---   ┆ ---     ┆ es      ┆ _depend │
    │ ---      ┆ ---     ┆ ---     ┆ ---     ┆   ┆ u32   ┆ f64     ┆ ---     ┆ ence    │
    │ f64      ┆ f64     ┆ f64     ┆ f64     ┆   ┆       ┆         ┆ array[f ┆ ---     │
    │          ┆         ┆         ┆         ┆   ┆       ┆         ┆ 64, 3]  ┆ f64     │
    ╞══════════╪═════════╪═════════╪═════════╪═══╪═══════╪═════════╪═════════╪═════════╡
    │ 0.0      ┆ 0.0     ┆ 0.1     ┆ 0.0     ┆ … ┆ 1     ┆ 1.0     ┆ [0.0,   ┆ 0.3     │
    │          ┆         ┆         ┆         ┆   ┆       ┆         ┆ 0.0,    ┆         │
    │          ┆         ┆         ┆         ┆   ┆       ┆         ┆ 0.2]    ┆         │
    │ 1.0      ┆ 0.5     ┆ 0.5     ┆ 0.5     ┆ … ┆ 2     ┆ 2.0     ┆ [0.8,   ┆ 0.5     │
    │          ┆         ┆         ┆         ┆   ┆       ┆         ┆ 0.0,    ┆         │
    │          ┆         ┆         ┆         ┆   ┆       ┆         ┆ 1.0]    ┆         │
    │ 2.0      ┆ 1.0     ┆ 0.9     ┆ 0.0     ┆ … ┆ 1     ┆ 1.0     ┆ [1.8,   ┆ 0.7     │
    │          ┆         ┆         ┆         ┆   ┆       ┆         ┆ 0.0,    ┆         │
    │          ┆         ┆         ┆         ┆   ┆       ┆         ┆ 2.0]    ┆         │
    └──────────┴─────────┴─────────┴─────────┴───┴───────┴─────────┴─────────┴─────────┘
    """
    validate_same_first_dimension(y_obs, y_pred)
    n_pred = length_of_second_dimension(y_pred)
    pred_names, _ = get_sorted_array_names(y_pred)
    y_obs = np.asarray(y_obs)

    if weights is not None:
        validate_same_first_dimension(weights, y_obs)
        w = np.asarray(weights)
        if w.ndim > 1:
            msg = f"The array weights must be 1-dimensional, got weights.ndim={w.ndim}."
            raise ValueError(msg)
    else:
        w = np.ones_like(y_obs, dtype=float)

    if feature_name is None:
        # X is completely ignored.
        feature_input = feature = None
    elif X is None:
        msg = (
            "X must be a data container like a (polars) dataframe or an (numpy) array."
        )
        raise ValueError(msg)
    elif not isinstance(feature_name, (int, str)):
        msg = f"The argument 'feature_name' must be an int or str; got {feature_name}"
        raise ValueError(msg)
    elif isinstance(feature_name, int):
        feature_index = feature_name
        feature_input = get_second_dimension(X, feature_name)
    else:
        X_names, _ = get_sorted_array_names(X)
        feature_index = X_names.index(feature_name)
        feature_input = get_second_dimension(X, feature_index)

    n_obs = length_of_first_dimension(y_pred)
    df_list = []
    with pl.StringCache():
        if feature_input is not None:
            feature, n_bins, f_binned = bin_feature(
                feature=feature_input,
                feature_name=feature_name,
                n_obs=n_obs,
                n_bins=n_bins,
                bin_method=bin_method,
            )
            feature_name = feature.name
            is_cat_or_string = feature.dtype in [
                pl.Categorical,
                pl.Enum,
                pl.Utf8,
                pl.Object,
            ]

        for i in range(len(pred_names)):
            # Loop over columns of y_pred.
            x = np.asarray(y_pred if n_pred == 0 else get_second_dimension(y_pred, i))

            if feature is None:
                y_obs_mean = np.average(y_obs, weights=w)
                y_pred_mean = np.average(x, weights=w)
                weights_sum = np.sum(w)
                count = y_obs.shape[0]
                # Note: with Bessel correction
                y_obs_stddev = np.average(
                    (y_obs - y_obs_mean) ** 2, weights=w
                ) / np.amax([1, count - 1])
                y_pred_stddev = np.average((x - y_pred_mean) ** 2, weights=w) / np.amax(
                    [1, count - 1]
                )
                df = pl.DataFrame(
                    {
                        "y_obs_mean": [y_obs_mean],
                        "y_pred_mean": [y_pred_mean],
                        "count": pl.Series([count], dtype=pl.UInt32),
                        "weights": [weights_sum],
                        "y_obs_stderr": [np.sqrt(y_obs_stddev)],
                        "y_pred_stderr": [np.sqrt(y_pred_stddev)],
                    }
                )
            else:
                df = pl.DataFrame(
                    {
                        "y_obs": y_obs,
                        "y_pred": x,
                        feature_name: feature,
                        "weights": w,
                    }
                )

                agg_list = [
                    pl.count("y_obs").alias("count"),
                    pl.col("weights").sum().alias("weights_sum"),
                    *chain.from_iterable(
                        [
                            pl.col(c + "_mean").first(),
                            (
                                (
                                    pl.col("weights")
                                    * ((pl.col(c) - pl.col(c + "_mean")) ** 2)
                                ).sum()
                                / pl.col("weights").sum()
                            ).alias(c + "_variance"),
                        ]
                        for c in ["y_obs", "y_pred"]
                    ),
                ]

                if is_cat_or_string:
                    groupby_name = feature_name
                else:
                    # We also add the bin edges.
                    df = df.hstack(
                        [f_binned.get_column("bin"), f_binned.get_column("bin_edges")]
                    )
                    groupby_name = "bin"
                    agg_list += [
                        pl.col(feature_name).mean(),
                        pl.col(feature_name).std(ddof=0).alias("__feature_std"),
                        pl.col("bin_edges").first(),
                    ]

                df = (
                    df.lazy()
                    .select(
                        pl.all(),
                        (
                            (pl.col("weights") * pl.col("y_obs"))
                            .sum()
                            .over(groupby_name)
                            / pl.col("weights").sum().over(groupby_name)
                        ).alias("y_obs_mean"),
                        (
                            (pl.col("weights") * pl.col("y_pred"))
                            .sum()
                            .over(groupby_name)
                            / pl.col("weights").sum().over(groupby_name)
                        ).alias("y_pred_mean"),
                    )
                    .group_by(groupby_name)
                    .agg(agg_list)
                    .with_columns(
                        [
                            pl.when(pl.col("count") > 1)
                            .then(pl.col(c + "_variance") / (pl.col("count") - 1))
                            .otherwise(pl.col(c + "_variance"))
                            .sqrt()
                            .alias(c + "_stderr")
                            for c in ("y_obs", "y_pred")
                        ]
                    )
                    # With sort and head alone, we could lose the null value, but we
                    # want to keep it.
                    # .sort("bias_count", descending=True)
                    # .head(n_bins)
                    .with_columns(
                        pl.when(pl.col(feature_name).is_null())
                        .then(pl.max("count") + 1)
                        .otherwise(pl.col("count"))
                        .alias("__priority")
                    )
                    .sort("__priority", descending=True)
                    .head(n_bins)
                )
                # FIXME: When n_bins=0, the result should be an empty dataframe
                # (0 rows and some columns). For some unknown reason as of
                # polars 0.20.20, the following sort neglects the head(0) statement.
                # Therefore, we place an explicit collect here. This should not be
                # needed!
                if n_bins == 0 or feature.null_count() >= 1:
                    df = df.collect().lazy()
                df = df.sort(feature_name, descending=False)

                df = df.select(
                    [
                        pl.col(feature_name),
                        pl.col("y_obs_mean"),
                        pl.col("y_pred_mean"),
                        pl.col("y_obs_stderr"),
                        pl.col("y_pred_stderr"),
                        pl.col("weights_sum").alias("weights"),
                        pl.col("count"),
                    ]
                    + (
                        []
                        if is_cat_or_string
                        else [pl.col("bin_edges"), pl.col("__feature_std")]
                    )
                ).collect()

                if not is_cat_or_string:
                    df = df.with_columns(
                        pl.col("bin_edges")
                        .arr.first()
                        .list.concat(pl.col("__feature_std"))
                        .list.concat(pl.col("bin_edges").arr.last())
                        .list.to_array(3)
                        .alias("bin_edges")
                    )

            # Add column "model".
            if n_pred > 0:
                model_col_name = "model_" if feature_name == "model" else "model"
                df = df.with_columns(
                    pl.Series(model_col_name, [pred_names[i]] * df.shape[0])
                )

            # Add partial dependence.
            with_pd = predict_function is not None and feature_name is not None
            if with_pd:
                pd_values = compute_partial_dependence(
                    pred_fun=predict_function,  # type: ignore
                    X=X,  # type: ignore
                    feature_index=feature_index,
                    grid=df.get_column(feature_name),
                    weights=weights,
                    n_max=n_max,
                    rng=rng,
                )
                df = df.with_columns(
                    pl.Series(name="partial_dependence", values=pd_values)
                )

            # Select the columns in the correct order.
            col_selection = []
            if n_pred > 0:
                col_selection.append(model_col_name)
            if feature_name is not None and feature_name in df.columns:
                col_selection.append(str(feature_name))
            col_selection += [
                "y_obs_mean",
                "y_pred_mean",
                "y_obs_stderr",
                "y_pred_stderr",
                "count",
                "weights",
            ]
            if feature_name in df.columns and not is_cat_or_string:
                col_selection += ["bin_edges"]
            if with_pd:
                col_selection += ["partial_dependence"]
            df_list.append(df.select(col_selection))

        df = pl.concat(df_list)
    return df
