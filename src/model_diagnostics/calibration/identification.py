from typing import Optional

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import special

from .._utils.array import (
    array_name,
    get_second_dimension,
    length_of_second_dimension,
    validate_2_arrays,
    validate_same_first_dimension,
)


def identification_function(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    *,
    functional: str = "mean",
    level: float = 0.5,
) -> np.ndarray:
    r"""Canonical identification function.

    Identification functions act as generalised residuals. See Notes for further
    details.

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
        It must be `0 <= level <= 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.

    Returns
    -------
    V : ndarray of shape (n_obs)
        Values of the identification function.

    Notes
    -----
    The function \(V(y, z)\) for observation \(y=y_{pred}\) and prediction
    \(z=y_{pred}\) is a strict identification function for the functional \(T\), or
    induces the functional \(T\) as:

    \[
    \mathbb{E}[V(Y, z)] = 0\quad \Leftrightarrow\quad z\in T(F) \quad \forall
    \text{ distributions } F
    \in \mathcal{F}
    \]

    for some class of distribtions \(\mathcal{F}\). Implemented examples of the
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

    Examples
    --------

    """
    y_o: np.ndarray
    y_p: np.ndarray
    y_o, y_p = validate_2_arrays(y_obs, y_pred)

    if functional in ("expectile", "quantile") and (level < 0 or level > 1):
        raise ValueError(f"Argument level must fulfil 0 <= level <= 1, got {level}.")

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
        raise ValueError(
            f"Argument functional must be one of {allowed_functionals}, got "
            f"{functional}."
        )


def compute_bias(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    feature: Optional[npt.ArrayLike] = None,
    weights: Optional[npt.ArrayLike] = None,
    *,
    functional: str = "mean",
    level: float = 0.5,
    n_bins: int = 10,
):
    r"""Compute generalised bias conditional on a feature.

    This function computes and aggregates the generalised bias, i.e. the values of the
    canonical identification function, versus (grouped by) a feature.
    This is a good way to assess whether a model is conditionally calibrated or not.
    Well calibrated models have bias terms around zero.
    For the mean functional, the generalised bias is the negative residual
    `y_pred - y_obs`.
    See Notes for further details.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs) or (n_obs, n_models)
        Predicted values of the conditional expectation of Y, `E(Y|X)`.
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
        It must be `0 <= level <= 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.
    n_bins : int
        The number of bins for numerical features and the maximal number of (most
        frequent) categories shown for categorical features.

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

    Notes
    -----
    A model \(m(X)\) is conditionally calibrated iff
    \(\mathbb{E}(V(m(X), Y)|X)=0\) almost surely with canonical identification
    function \(V\).
    The empirical version, given some data, reads
    \(\bar{V} = \frac{1}{n}\sum_i \phi(x_i) V(m(x_i), y_i)\) with a function
    \(\phi(x_i)\) that projects on the specified feature.
    For a feature with only two distinct values `"a"` and `"b"`, this becomes
    \(\bar{V} = \frac{1}{n_a}\sum_{i \text{ with }x_i=a} V(m(a), y_i)\) with
    \(n_a=\sum_{i \text{ with }x_i=a}\) and the same with `"b"`.
    With case weights, this reads
    \(\bar{V} = \frac{1}{\sum_i w_i}\sum_i w_i \phi(x_i) V(m(x_i), y_i)\).
    This generalises the classical residual (up to a minus sign) for target functionals
    other than the mean. See `[FLM2022]`.

    The standard error for \(\bar{V}\) is caluclated in the standard way as
    \(\mathrm{SE} = \sqrt{\operatorname{Var}(\bar{V})} = \frac{\sigma}{\sqrt{n}}\) and
    the standard variance estimator for \(\sigma^2 = \operatorname{Var}(\phi(x_i)
    V(m(x_i), y_i))\) with Bessel correction, i.e. division by \(n-1\) instead of
    \(n\).

    With case weights, the variance estimator becomes \(\operatorname{Var}(\bar{V})
    = \frac{1}{n-1} \frac{1}{\sum_i w_i} \sum_i w_i (V(m(x_i), y_i) - \bar{V})^2\) with
    the implied relation \(\operatorname{Var}(V(m(x_i), y_i)) \sim \frac{1}{w_i} \).
    If your weights are for repeated observations, so-called frequency weights, then
    the above estimate is on the conservative site because is used \(n - 1\) instead
    of \((\sum_i w_i) - 1\).

    References
    ----------
    `[FLM2022]`

    :   T. Fissler, C. Lorentzen, and M. Mayer.
        "Model Comparison and Calibration Assessment". (2022)
        [arxiv:https://arxiv.org/abs/2202.12780](https://arxiv.org/abs/2202.12780).
    """
    validate_same_first_dimension(y_obs, y_pred)
    if feature is not None:
        validate_same_first_dimension(y_obs, feature)
    n_pred = length_of_second_dimension(y_pred)
    if n_pred == 0:
        pred_names = [""]
    else:
        pred_names = []
        for i in range(n_pred):
            x = get_second_dimension(y_pred, i)
            pred_names.append(array_name(x, default=str(i)))

    if weights is not None:
        validate_same_first_dimension(weights, y_obs)
        w = np.asarray(weights)
        if w.ndim > 1:
            raise ValueError(
                "The array weights must be 1-dimensional, got weights.ndim={w.ndim}."
            )
    else:
        w = np.ones_like(y_obs)

    df_list = []
    with pl.StringCache():
        for i in range(len(pred_names)):
            # Loop over columns of y_pred.
            if n_pred == 0:
                x = y_pred
            else:
                x = get_second_dimension(y_pred, i)

            bias = identification_function(
                y_obs=y_obs,
                y_pred=x,
                functional=functional,
                level=level,
            )

            if feature is None:
                feature_name = None
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
                        "bias_count": [bias_count],
                        "bias_weights": [bias_weights],
                        "bias_stderr": [np.sqrt(bias_stddev)],
                    }
                )
            else:
                feature_name = array_name(feature, default="feature")
                df = pl.DataFrame(
                    {
                        "y_obs": y_obs,
                        "y_pred": x,
                        feature_name: feature,
                        "bias": bias,
                        "weights": w,
                    }
                )

                is_categorical = False
                is_numeric = False
                is_string = False
                if df.get_column(feature_name).dtype is pl.Categorical:
                    is_categorical = True
                elif df.get_column(feature_name).dtype in [pl.Utf8, pl.Object]:
                    # We could convert strings to categoricals.
                    is_string = True
                else:
                    is_numeric = True

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

                if is_categorical or is_string:
                    groupby_name = feature_name
                else:
                    # Binning
                    # We use method="inverted_cdf" (same as "lower") instead of the
                    # default "linear" because "linear" produces as many unique values
                    # as before.
                    q = np.quantile(
                        feature,
                        # q=np.linspace(0 + 1 / n_bins, 1 - 1 / n_bins, num=n_bins - 1),
                        # improved rounding errors
                        q=np.linspace(0 + 1, n_bins - 1, num=n_bins - 1) / n_bins,
                        method="inverted_cdf",
                    )  # type: ignore
                    q = np.unique(q)  # Some quantiles might be the same.
                    # bins[i-1] < x <= bins[i]
                    f_binned = np.digitize(feature, bins=q, right=True)  # type: ignore
                    df = df.hstack([pl.Series("bin", f_binned)])
                    groupby_name = "bin"
                    agg_list.append(pl.col(feature_name).mean())

                df = (
                    df.lazy()
                    .select(
                        [
                            pl.all(),
                            (
                                (pl.col("weights") * pl.col("bias"))
                                .sum()
                                .over(groupby_name)
                                / pl.col("weights").sum().over(groupby_name)
                            ).alias("bias_mean"),
                        ]
                    )
                    .groupby(groupby_name)
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
                    .sort("bias_count", reverse=True)
                    .head(n_bins)
                    .sort(feature_name)
                )
                if is_numeric:
                    df = df.drop("bin")

                df = df.select(
                    [
                        pl.col(feature_name),
                        pl.col("bias_mean"),
                        pl.col("bias_count"),
                        pl.col("bias_weights"),
                        pl.col("bias_stderr"),
                    ]
                ).collect()

                # if is_categorical:
                #     # Pyarrow does not yet support sorting dictionary type arrays,
                #     # see
                #     # https://issues.apache.org/jira/browse/ARROW-14314
                #     # We resort to pandas instead.
                #     import pyarrow as pa
                #     df = df.to_pandas().sort_values(feature_name)
                #     df = pa.Table.from_pandas(df)

            # Add column with p-value of 2-sided t-test.
            # We explicitly convert "to_numpy", because otherwise we get:
            #   RuntimeWarning: A builtin ctypes object gave a PEP3118 format string
            #   that does not match its itemsize, so a best-guess will be made of the
            #   data type. Newer versions of python may behave correctly.
            stderr_ = df.get_column("bias_stderr")
            p_value = np.full_like(stderr_, fill_value=np.nan)
            mask = stderr_ > 0
            x = df.get_column("bias_mean").filter(mask).to_numpy()
            n = df.get_column("bias_count").filter(mask).to_numpy()
            stderr = stderr_.filter(mask).to_numpy()
            # t-statistic t (-|t| and factor of 2 because of 2-sided test)
            p_value[mask] = 2 * special.stdtr(
                n - 1,  # degrees of freedom
                -np.abs(x / stderr),
            )
            df = df.with_columns(pl.Series("p_value", p_value))

            # Add column "model".
            if n_pred > 0:
                if feature_name == "model":
                    model_col_name = "model_"
                else:
                    model_col_name = "model"
                df = df.with_columns(
                    pl.Series(model_col_name, [pred_names[i]] * df.shape[0])
                )

            # Select the columns in the correct order.
            col_selection = []
            if n_pred > 0:
                col_selection.append(model_col_name)
            if feature_name is not None and feature_name in df.columns:
                col_selection.append(feature_name)
            col_selection += ["bias_mean", "bias_count", "bias_stderr", "p_value"]
            df_list.append(df.select(col_selection))

        df = pl.concat(df_list)
    return df
