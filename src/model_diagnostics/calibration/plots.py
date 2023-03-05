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
    get_sorted_array_names,
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

    if diagram_type == "reliability":
        if hasattr(y_pred, "max") and hasattr(y_pred, "min"):
            y_max, y_min = y_pred.max(), y_pred.min()
        else:
            y_max, y_min = np.max(y_pred), np.min(y_pred)
        ax.plot([y_min, y_max], [y_min, y_max], color="k", linestyle="dotted")
    else:
        # horizontal line at y=0
        ax.axhline(y=0, xmin=0, xmax=1, color="k", linestyle="dotted")

    n_pred = length_of_second_dimension(y_pred)
    pred_names, _ = get_sorted_array_names(y_pred)

    for i in range(len(pred_names)):
        y_pred_i = y_pred if n_pred == 0 else get_second_dimension(y_pred, i)

        iso = (
            IsotonicRegression()
            .set_output(transform="default")
            .fit(y_pred_i, y_obs, sample_weight=weights)
        )

        # confidence intervals
        if n_bootstrap is not None:
            data: tuple[npt.ArrayLike, ...]
            data = (y_obs, y_pred_i) if weights is None else (y_obs, y_pred_i, weights)

            def iso_statistic(y_obs, y_pred, weights=None):
                iso_b = (
                    IsotonicRegression(out_of_bounds="clip")
                    .set_output(transform="default")
                    .fit(y_pred, y_obs, sample_weight=weights)
                )
                return iso_b.predict(iso.X_thresholds_)

            boot = bootstrap(
                data=data,
                statistic=iso_statistic,
                n_resamples=n_bootstrap,
                paired=True,
                confidence_level=confidence_level,
                # Note: method="bca" might result in
                # DegenerateDataWarning: The BCa confidence interval cannot be
                # calculated. This problem is known to occur when the distribution is
                # degenerate or the statistic is np.min.
                method="basic",
            )

            # We make the interval conservatively monotone increasing by applying
            # np.maximum.accumulate etc.
            lower = -np.minimum.accumulate(-boot.confidence_interval.low)
            upper = np.maximum.accumulate(boot.confidence_interval.high)
            if diagram_type == "bias":
                lower = iso.X_thresholds_ - lower
                upper = iso.X_thresholds_ - upper
            ax.fill_between(iso.X_thresholds_, lower, upper, alpha=0.1)

        # reliability curve
        label = pred_names[i] if n_pred >= 2 else None

        y_plot = (
            iso.y_thresholds_
            if diagram_type == "reliability"
            else iso.X_thresholds_ - iso.y_thresholds_
        )
        ax.plot(iso.X_thresholds_, y_plot, label=label)

    if diagram_type == "reliability":
        ylabel = "estimated E(Y|prediction)"
        title = "Reliability Diagram"
    else:
        ylabel = "prediction - estimated E(Y|prediction)"
        title = "Bias Reliability Diagram"
    ax.set(xlabel="prediction for E(Y|X)", ylabel=ylabel)

    if n_pred >= 2:
        ax.set_title(title)
        ax.legend()
    else:
        y_pred_i = y_pred if n_pred == 0 else get_second_dimension(y_pred, i)
        if len(pred_names[0]) > 0:
            ax.set_title(title + " " + pred_names[0])
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
        frequent) categories shown for categorical features. Due to ties, the effective
        number of bins might be smaller than `n_bins`.
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

    n_x = df[feature_name].n_unique()

    # horizontal line at y=0
    ax.axhline(y=0, xmin=0, xmax=1, color="k", linestyle="dotted")

    # bias plot
    if feature is None or col_model is None:
        pred_names = [None]
    else:
        # pred_names = df[col_model].unique() this automatically sorts
        pred_names, _ = get_sorted_array_names(y_pred)
    n_models = len(pred_names)
    with_label = feature is not None and n_models >= 2

    for i, m in enumerate(pred_names):
        filter_condition = True if m is None else pl.col(col_model) == m
        df_i = df.filter(filter_condition)
        label = m if with_label else None

        if df_i["bias_stderr"].null_count() > 0 or with_errorbars is False:
            if is_string or is_categorical:
                ax.plot(df_i[feature_name], df_i["bias_mean"], "o", label=label)
            else:
                ax.plot(df_i[feature_name], df_i["bias_mean"], "o-", label=label)
        elif is_string or is_categorical:
            # We x-shift a little for a better visual.
            span = (n_x - 1) / n_x / n_models  # length for one cat value and one model
            x = np.arange(n_x)
            if n_models > 1:
                x = x + (i - n_models // 2) * span * 0.5
            ax.errorbar(
                x,
                df_i["bias_mean"],
                yerr=df_i["bias_stderr"],
                marker="o",
                linestyle="None",
                capsize=4,
                label=label,
            )
        else:
            lower = df_i["bias_mean"] - df_i["bias_stderr"]
            upper = df_i["bias_mean"] + df_i["bias_stderr"]
            ax.fill_between(df_i[feature_name], lower, upper, alpha=0.1)
            ax.plot(
                df_i[feature_name],
                df_i["bias_mean"],
                linestyle="solid",
                marker="o",
                label=label,
            )

    if is_categorical or is_string:
        ax.set_xticks(np.arange(n_x), df_i[feature_name])
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
