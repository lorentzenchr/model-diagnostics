from functools import partial
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.stats import bootstrap
from sklearn.isotonic import IsotonicRegression

from model_diagnostics._utils._array import array_name

from .identification import (
    compute_bias,
    get_second_dimension,
    length_of_second_dimension,
)


def plot_reliability_diagram(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    weights: Optional[npt.ArrayLike] = None,
    *,
    n_bootstrap: Optional[str] = None,
    confidence_level: float = 0.9,
    diagram_type: str = "reliability",
    ax: Optional[mpl.axes.Axes] = None,
):
    r"""Plot a reliability diagram.

    A reliability diagram or calibration curve assess auto-calibration. It plots the
    conditional expectation given the predictions `E(y_obs|y_pred)` (y-axis) vs the
    predictions `y_pred` (x-axis).
    The conditional expectation is estimated via isotonic regression (PAV algorithm)
    of `y_obs` on `y_pred`.
    See Notes for further details.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs) or (n_obs, n_models)
        Predicted values of the conditional expectation of Y, `E(Y|X)`.
    weights : array-like of shape (n_obs) or None
        Case weights.
    n_bootstrap : int or None
        If not `None`, then `scipy.stats.bootstrap` with `n_resamples=n_bootstrap`
        is used to calculate confidence intervals at level `confidence_level`.
    confidence_level : float
        Confidence level for bootstrap uncertainty regions.
    diagram_type: str
        - `"reliability"`: Plot a reliability diagram.
        - `"bias"`: Plot roughly a 45 degree rotated reliability diagram. The resulting
          plot is similar to `plot_bias`, i.e. `y_pred - E(y_obs|y_pred)` vs `y_pred`.
    ax : matplotlib.axes.Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax

    Notes
    -----
    The expectation conditional on the predictions is \(E(Y|y_{pred})\). This object is
    estimated by the pool-adjacent violator (PAV) algorithm, which has very desirable
    properties:

        - It is non-parametric without any tuning parameter. Thus, the results are
          easily reproducible.
        - Optimal selection of bins
        - Statistical consistent estimator

    For details, refer to [Dimitriadis2021].

    References
    ----------
    `[Dimitriadis2021]`

    :   T. Dimitriadis, T. Gneiting, and A. I. Jordan.
        "Stable reliability diagrams for probabilistic classifiers".
        In: Proceedings of the National Academy of Sciences 118.8 (2021), e2016191118.
        [doi:10.1073/pnas.2016191118](https://doi.org/10.1073/pnas.2016191118).
    """
    if ax is None:
        ax = plt.gca()

    if diagram_type not in ("reliability", "bias"):
        msg = (
            "Parameter diagram_type must be either 'reliability', 'bias', "
            f"got {diagram_type}."
        )
        raise ValueError(msg)

    # diagonal line
    n_pred = length_of_second_dimension(y_pred)
    if n_pred > 0:
        y_pred_min, y_pred_max = np.inf, -np.inf
        for i in range(n_pred):
            y_pred_i = get_second_dimension(y_pred, i)
            if not hasattr(y_pred_i, "min"):
                y_pred_i = np.asarray(y_pred)
            y_pred_min = np.amin([y_pred_min, y_pred_i.min()])  # type: ignore
            y_pred_max = np.amax([y_pred_max, y_pred_i.max()])  # type: ignore
    else:
        if not (hasattr(y_pred, "min") and hasattr(y_pred, "max")):
            y_pred = np.asarray(y_pred)
        y_pred_min, y_pred_max = y_pred.min(), y_pred.max()

    if diagram_type == "reliability":
        ax.plot(
            [y_pred_min, y_pred_max],
            [y_pred_min, y_pred_max],
            color="k",
            linestyle="dotted",
        )
    else:
        ax.hlines(y=0, xmin=y_pred_min, xmax=y_pred_max, color="k", linestyle="dotted")

    def iso_statistic(y_obs, y_pred, weights=None, x_values=None):
        iso_b = IsotonicRegression(out_of_bounds="clip").fit(
            y_pred, y_obs, sample_weight=weights
        )
        return iso_b.predict(x_values)

    for i in range(np.maximum(1, n_pred)):
        y_pred_i = y_pred if n_pred == 0 else get_second_dimension(y_pred, i)

        iso = IsotonicRegression().fit(y_pred_i, y_obs, sample_weight=weights)

        # confidence intervals
        if n_bootstrap is not None:
            data: tuple[npt.ArrayLike, ...]
            data = (y_obs, y_pred_i) if weights is None else (y_obs, y_pred_i, weights)

            boot = bootstrap(
                data=data,
                statistic=partial(iso_statistic, x_values=iso.X_thresholds_),
                n_resamples=n_bootstrap,
                paired=True,
                confidence_level=confidence_level,
                # Note: method="bca" might result in
                # DegenerateDataWarning: The BCa confidence interval cannot be
                # calculated. This problem is known to occur when the distribution is
                # degenerate or the statistic is np.min.
                method="basic",
            )

        # confidence intervals
        if n_bootstrap is not None:
            if diagram_type == "reliability":
                ax.fill_between(
                    iso.X_thresholds_,
                    # We make the interval conservatively monotone increasing by
                    # applying np.maximum.accumulate etc.
                    -np.minimum.accumulate(-boot.confidence_interval.low),
                    np.maximum.accumulate(boot.confidence_interval.high),
                    alpha=0.1,
                )
            else:
                ax.fill_between(
                    iso.X_thresholds_,
                    iso.X_thresholds_
                    + np.minimum.accumulate(-boot.confidence_interval.low),
                    iso.X_thresholds_
                    - np.maximum.accumulate(boot.confidence_interval.high),
                    alpha=0.1,
                )

        # reliability curve
        if n_pred >= 2:
            label = array_name(y_pred_i, default="")
            if len(label) == 0:
                label = str(i)
        else:
            label = None
        if diagram_type == "reliability":
            ax.plot(iso.X_thresholds_, iso.y_thresholds_, label=label)
        else:
            ax.plot(
                iso.X_thresholds_, iso.X_thresholds_ - iso.y_thresholds_, label=label
            )

    if diagram_type == "reliability":
        ax.set(xlabel="prediction for E(Y|X)", ylabel="estimated E(Y|prediction)")
        title = "Reliability Diagram"
    else:
        ax.set(
            xlabel="prediction for E(Y|X)",
            ylabel="bias = prediction - estimated E(Y|prediction)",
        )
        title = "Bias Reliability Diagram"

    if n_pred >= 2:
        ax.set_title(title)
        ax.legend()
    else:
        y_pred_i = y_pred if n_pred == 0 else get_second_dimension(y_pred, i)
        if len(model_name := array_name(y_pred_i, default="")) > 0:
            ax.set_title(title + " " + model_name)
        else:
            ax.set_title(title)

    return ax


def plot_bias(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    feature: Optional[npt.ArrayLike] = None,
    weights: Optional[npt.ArrayLike] = None,
    *,
    functional: str = "mean",
    level: float = 0.5,
    n_bins: int = 10,
    with_errorbars: bool = True,
    ax=None,
):
    r"""Plot model bias conditional on a feature.

    This plots the generalised bias (residuals), i.e. the values of the canonical
    identification function, versus a feature. This is a good way to assess whether
    a model is conditionally calibrated or not. Well calibrated models have bias terms
    around zero.
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
    functional : str
        The functional that is induced by the identification function `V`. Options are:
        - `"mean"`. Argument `level` is neglected.
        - `"median"`. Argument `level` is neglected.
        - `"expectile"`
        - `"quantile"`
    weights : array-like of shape (n_obs) or None
        Case weights. If given, the bias is calculated as weighted average of the
        identification function with these weights.
        Note that the standard errors and p-values in the output are based on the
        assumption that the variance of the bias is inverse proportional to the
        weights. See the Notes section for details.
    level : float
        The level of the expectile of quantile. (Often called \(\alpha\).)
        It must be `0 <= level <= 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.
    n_bins : int
        The number of bins for numerical features and the maximal number of (most
        frequent) categories shown for categorical features.
    with_errorbars : bool
        Whether or not to plot error bars.
    ax : matplotlib.axes.Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax

    Notes
    -----
    A model \(m(X)\) is conditionally calibrated iff \(E(V(m(X), Y))=0\) a.s. The
    empirical version, given some data, reads \(\frac{1}{n}\sum_i V(m(x_i), y_i)\).
    This generali. See [FLM2022]`.

    References
    ----------
    `FLM2022`

    :   T. Fissler, C. Lorentzen, and M. Mayer.
        "Model Comparison and Calibration Assessment". (2022)
        [arxiv:2202.12780](https://arxiv.org/abs/2202.12780).
    """
    if ax is None:
        ax = plt.gca()

    df = compute_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        feature=feature,
        weights=weights,
        functional=functional,
        level=level,
        n_bins=n_bins,
    )

    if "model_" in df.columns:
        col_model = "model_"
    elif "model" in df.columns:
        col_model = "model"
    else:
        col_model = None

    if feature is None:
        # We treat the predictions from different models as a feature.
        feature_name = col_model
    else:
        feature_name = array_name(feature, default="feature")

    is_categorical = False
    is_string = False
    if df.get_column(feature_name).dtype is pl.Categorical:
        is_categorical = True
    elif df.get_column(feature_name).dtype in [pl.Utf8, pl.Object]:
        is_string = True

    # horizontal line at y=0
    if is_categorical or is_string:
        min_max = {"min": 0, "max": df[feature_name].n_unique() - 1}
    else:
        min_max = {"min": df[feature_name].min(), "max": df[feature_name].max()}
    ax.hlines(
        y=0, xmin=min_max["min"], xmax=min_max["max"], color="k", linestyle="dotted"
    )

    # bias plot
    if feature is None or col_model is None:
        model_names = [None]
    else:
        model_names = df[col_model].unique().sort(descending=False)
    with_label = feature is not None and len(model_names) >= 2

    for i, m in enumerate(model_names):
        filter_condition = True if m is None else pl.col(col_model) == m
        df_i = df.filter(filter_condition)
        label = m if with_label else None

        if df_i["bias_stderr"].null_count() > 0 or with_errorbars is False:
            ax.plot(df_i[feature_name], df_i["bias_mean"], "o-", label=label)
        else:
            # We x-shift a little for a better visual.
            span = min_max["max"] - min_max["min"]
            if is_categorical or is_string:
                x = np.arange(df_i[feature_name].shape[0])
            else:
                x = df_i[feature_name]
            x = x + (i - len(model_names) // 2) * span * 5e-3

            ax.errorbar(
                x,
                df_i["bias_mean"],
                yerr=df_i["bias_stderr"],
                fmt="o-",
                capsize=4,
                label=label,
            )

    if is_categorical or is_string:
        ax.set_xticks(x, df_i[feature_name])
        ax.set(xlabel=feature_name, ylabel="bias")
    elif feature_name is not None:
        ax.set(xlabel="binned " + feature_name, ylabel="bias")

    if feature is None:
        ax.set_title("Bias Plot")
    elif with_label:
        ax.set_title("Bias Plot")
        ax.legend()
    else:
        model_name = array_name(y_pred, default="")
        if model_name == "":
            ax.set_title("Bias Plot")
        else:
            ax.set_title("Bias Plot " + model_name)

    return ax
