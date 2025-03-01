import collections
import warnings
from functools import partial
from typing import Callable, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import special
from scipy.stats import bootstrap
from sklearn.isotonic import IsotonicRegression as IsotonicRegression_skl

from model_diagnostics import get_config
from model_diagnostics._utils.array import (
    array_name,
    get_array_min_max,
    get_second_dimension,
    get_sorted_array_names,
    length_of_second_dimension,
)
from model_diagnostics._utils.isotonic import IsotonicRegression
from model_diagnostics._utils.plot_helper import get_plotly_color, is_plotly_figure

from .identification import compute_bias, compute_marginal


def plot_reliability_diagram(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    weights: Optional[npt.ArrayLike] = None,
    *,
    functional: str = "mean",
    level: float = 0.5,
    n_bootstrap: Optional[str] = None,
    confidence_level: float = 0.9,
    diagram_type: str = "reliability",
    ax: Optional[mpl.axes.Axes] = None,
):
    r"""Plot a reliability diagram.

    A reliability diagram or calibration curve assesses auto-calibration. It plots the
    conditional expectation given the predictions `E(y_obs|y_pred)` (y-axis) vs the
    predictions `y_pred` (x-axis).
    The conditional expectation is estimated via isotonic regression (PAV algorithm)
    of `y_obs` on `y_pred`.
    See [Notes](#notes) for further details.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs) or (n_obs, n_models)
        Predicted values, e.g. for the conditional expectation of the response,
        `E(Y|X)`.
    weights : array-like of shape (n_obs) or None
        Case weights.
    functional : str
        The functional that is induced by the identification function `V`. Options are:

        - `"mean"`. Argument `level` is neglected.
        - `"median"`. Argument `level` is neglected.
        - `"expectile"`
        - `"quantile"`

    level : float
        The level of the expectile or quantile. (Often called \(\alpha\).)
        It must be `0 <= level <= 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.
    n_bootstrap : int or None
        If not `None`, then `scipy.stats.bootstrap` with `n_resamples=n_bootstrap`
        is used to calculate confidence intervals at level `confidence_level`.
    confidence_level : float
        Confidence level for bootstrap uncertainty regions.
    diagram_type: str
        - `"reliability"`: Plot a reliability diagram.
        - `"bias"`: Plot roughly a 45 degree rotated reliability diagram. The resulting
          plot is similar to `plot_bias`, i.e. `y_pred - E(y_obs|y_pred)` vs `y_pred`.
    ax : matplotlib.axes.Axes or plotly Figure
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax :
        Either the matplotlib axes or the plotly figure. This is configurable by
        setting the `plot_backend` via
        [`model_diagnostics.set_config`][model_diagnostics.set_config] or
        [`model_diagnostics.config_context`][model_diagnostics.config_context].

    Notes
    -----
    [](){#notes}
    The expectation conditional on the predictions is \(E(Y|y_{pred})\). This object is
    estimated by the pool-adjacent violator (PAV) algorithm, which has very desirable
    properties:

        - It is non-parametric without any tuning parameter. Thus, the results are
          easily reproducible.
        - Optimal selection of bins
        - Statistical consistent estimator

    For details, refer to `[Dimitriadis2021]`.

    References
    ----------
    `[Dimitriadis2021]`

    :   T. Dimitriadis, T. Gneiting, and A. I. Jordan.
        "Stable reliability diagrams for probabilistic classifiers".
        In: Proceedings of the National Academy of Sciences 118.8 (2021), e2016191118.
        [doi:10.1073/pnas.2016191118](https://doi.org/10.1073/pnas.2016191118).
    """
    if ax is None:
        plot_backend = get_config()["plot_backend"]
        if plot_backend == "matplotlib":
            ax = plt.gca()
        else:
            import plotly.graph_objects as go

            fig = ax = go.Figure()
    elif isinstance(ax, mpl.axes.Axes):
        plot_backend = "matplotlib"
    elif is_plotly_figure(ax):
        import plotly.graph_objects as go

        plot_backend = "plotly"
        fig = ax
    else:
        msg = (
            "The ax argument must be None, a matplotlib Axes or a plotly Figure, "
            f"got {type(ax)}."
        )
        raise ValueError(msg)

    if diagram_type not in ("reliability", "bias"):
        msg = (
            "Parameter diagram_type must be either 'reliability', 'bias', "
            f"got {diagram_type}."
        )
        raise ValueError(msg)

    if (n_cols := length_of_second_dimension(y_obs)) > 0:
        if n_cols == 1:
            y_obs = get_second_dimension(y_obs, 0)
        else:
            msg = (
                f"Array-like y_obs has more than 2 dimensions, y_obs.shape[1]={n_cols}"
            )
            raise ValueError(msg)

    y_min, y_max = get_array_min_max(y_pred)
    if diagram_type == "reliability":
        if plot_backend == "matplotlib":
            ax.plot([y_min, y_max], [y_min, y_max], color="k", linestyle="dotted")
        else:
            fig.add_scatter(
                x=[y_min, y_max],
                y=[y_min, y_max],
                mode="lines",
                line={"color": "black", "dash": "dot"},
                showlegend=False,
            )
    elif plot_backend == "matplotlib":
        # horizontal line at y=0

        # The following plots in axis coordinates
        # ax.axhline(y=0, xmin=0, xmax=1, color="k", linestyle="dotted")
        # but we plot in data coordinates instead.
        ax.hlines(0, xmin=y_min, xmax=y_max, color="k", linestyle="dotted")
    else:
        # horizontal line at y=0
        fig.add_hline(y=0, line={"color": "black", "dash": "dot"}, showlegend=False)

    if n_bootstrap is not None:
        if functional == "mean":

            def iso_statistic(y_obs, y_pred, weights=None, x_values=None):
                iso_b = (
                    IsotonicRegression_skl(out_of_bounds="clip")
                    .set_output(transform="default")
                    .fit(y_pred, y_obs, sample_weight=weights)
                )
                return iso_b.predict(x_values)

        else:

            def iso_statistic(y_obs, y_pred, weights=None, x_values=None):
                iso_b = IsotonicRegression(functional=functional, level=level).fit(
                    y_pred, y_obs, sample_weight=weights
                )
                return iso_b.predict(x_values)

    n_pred = length_of_second_dimension(y_pred)
    pred_names, _ = get_sorted_array_names(y_pred)

    for i in range(len(pred_names)):
        y_pred_i = y_pred if n_pred == 0 else get_second_dimension(y_pred, i)

        if functional == "mean":
            iso = (
                IsotonicRegression_skl()
                .set_output(transform="default")
                .fit(y_pred_i, y_obs, sample_weight=weights)
            )
        else:
            iso = IsotonicRegression(functional=functional, level=level).fit(
                y_pred_i, y_obs, sample_weight=weights
            )

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

            # We make the interval conservatively monotone increasing by applying
            # np.maximum.accumulate etc.
            # Conservative here means smaller intervals such that it is more likely
            # for the prediction to be out of the intervals leading to the conclusion
            # of "not auto-calibrated".
            lower = np.maximum.accumulate(boot.confidence_interval.low)
            upper = np.minimum.accumulate(boot.confidence_interval.high[::-1])[::-1]
            if diagram_type == "bias":
                lower = iso.X_thresholds_ - lower
                upper = iso.X_thresholds_ - upper
            if plot_backend == "matplotlib":
                ax.fill_between(iso.X_thresholds_, lower, upper, alpha=0.1)
            else:
                # plotly has not equivalent of fill_between and needs a bit more coding
                color = get_plotly_color(i)
                fig.add_scatter(
                    x=np.r_[iso.X_thresholds_, iso.X_thresholds_[::-1]],
                    y=np.r_[lower, upper[::-1]],
                    fill="toself",
                    fillcolor=color,
                    hoverinfo="skip",
                    line={"color": color},
                    mode="lines",
                    opacity=0.1,
                    showlegend=False,
                )

        # reliability curve
        label = pred_names[i] if n_pred >= 2 else None

        y_plot = (
            iso.y_thresholds_
            if diagram_type == "reliability"
            else iso.X_thresholds_ - iso.y_thresholds_
        )
        if plot_backend == "matplotlib":
            ax.plot(iso.X_thresholds_, y_plot, label=label)
        else:
            fig.add_scatter(
                x=iso.X_thresholds_,
                y=y_plot,
                mode="lines",
                line={"color": get_plotly_color(i)},
                name=label,
            )

    xlabel_mapping = {
        "mean": "E(Y|X)",
        "median": "median(Y|X)",
        "expectile": f"{level}-expectile(Y|X)",
        "quantile": f"{level}-quantile(Y|X)",
    }
    ylabel_mapping = {
        "mean": "E(Y|prediction)",
        "median": "median(Y|prediction)",
        "expectile": f"{level}-expectile(Y|prediction)",
        "quantile": f"{level}-quantile(Y|prediction)",
    }
    xlabel = "prediction for " + xlabel_mapping[functional]
    if diagram_type == "reliability":
        ylabel = "estimated " + ylabel_mapping[functional]
        title = "Reliability Diagram"
    else:
        ylabel = "prediction - estimated " + ylabel_mapping[functional]
        title = "Bias Reliability Diagram"

    if n_pred <= 1 and len(pred_names[0]) > 0:
        title = title + " " + pred_names[0]

    if plot_backend == "matplotlib":
        if n_pred >= 2:
            ax.legend()
        ax.set_title(title)
        ax.set(xlabel=xlabel, ylabel=ylabel)
    else:
        if n_pred <= 1:
            fig.update_layout(showlegend=False)
        fig.update_layout(xaxis_title=xlabel, yaxis_title=ylabel, title=title)

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
    bin_method: str = "sturges",
    confidence_level: float = 0.9,
    ax: Optional[mpl.axes.Axes] = None,
):
    r"""Plot model bias conditional on a feature.

    This plots the generalised bias (residuals), i.e. the values of the canonical
    identification function, versus a feature. This is a good way to assess whether
    a model is conditionally calibrated or not. Well calibrated models have bias terms
    around zero.
    See [Notes](#notes) for further details.

    For numerical features, NaN are treated as Null values. Null values are always
    plotted as rightmost value on the x-axis and marked with a diamond instead of a
    dot.

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
        The level of the expectile or quantile. (Often called \(\alpha\).)
        It must be `0 <= level <= 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.
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
        The method for finding bin edges (boundaries). Options using `n_bins` are:

        - `"quantile"`
        - `"uniform"`

        Options automatically selecting the number of bins for numerical features
        thereby using uniform bins are same options as
        [numpy.histogram_bin_edges](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html):

        - `"auto"`
        Minimum bin width between the `"sturges"` and `"fd"` estimators. Provides good
        all-around performance.
        - `"fd"` (Freedman Diaconis Estimator)
        Robust (resilient to outliers) estimator that takes into account data
        variability and data size.
        - `"doane"`
        An improved version of Sturges' estimator that works better with non-normal
        datasets.
        - `"scott"`
        Less robust estimator that takes into account data variability and data size.
        - `"stone"`
        Estimator based on leave-one-out cross-validation estimate of the integrated
        squared error. Can be regarded as a generalization of Scott's rule.
        - `"rice"`
        Estimator does not take variability into account, only data size. Commonly
        overestimates number of bins required.
        - `"sturges"`
        R's default method, only accounts for data size. Only optimal for gaussian data
        and underestimates number of bins for large non-gaussian datasets.
        - `"sqrt"`
        Square root (of data size) estimator, used by Excel and other programs for its
        speed and simplicity.

    confidence_level : float
        Confidence level for error bars. If 0, no error bars are plotted. Value must
        fulfil `0 <= confidence_level < 1`.
    ax : matplotlib.axes.Axes or plotly Figure
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax :
        Either the matplotlib axes or the plotly figure. This is configurable by
        setting the `plot_backend` via
        [`model_diagnostics.set_config`][model_diagnostics.set_config] or
        [`model_diagnostics.config_context`][model_diagnostics.config_context].

    Notes
    -----
    [](){#notes}
    A model \(m(X)\) is conditionally calibrated iff \(E(V(m(X), Y))=0\) a.s. The
    empirical version, given some data, reads \(\frac{1}{n}\sum_i V(m(x_i), y_i)\).
    See `[FLM2022]`.

    References
    ----------
    `FLM2022`

    :   T. Fissler, C. Lorentzen, and M. Mayer.
        "Model Comparison and Calibration Assessment". (2022)
        [arxiv:2202.12780](https://arxiv.org/abs/2202.12780).
    """
    if not (0 <= confidence_level < 1):
        msg = (
            f"Argument confidence_level must fulfil 0 <= level < 1, got "
            f"{confidence_level}."
        )
        raise ValueError(msg)
    with_errorbars = confidence_level > 0

    if ax is None:
        plot_backend = get_config()["plot_backend"]
        if plot_backend == "matplotlib":
            ax = plt.gca()
        else:
            import plotly.graph_objects as go

            fig = ax = go.Figure()
    elif isinstance(ax, mpl.axes.Axes):
        plot_backend = "matplotlib"
    elif is_plotly_figure(ax):
        import plotly.graph_objects as go

        plot_backend = "plotly"
        fig = ax
    else:
        msg = (
            "The ax argument must be None, a matplotlib Axes or a plotly Figure, "
            f"got {type(ax)}."
        )
        raise ValueError(msg)

    df = compute_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        feature=feature,
        weights=weights,
        functional=functional,
        level=level,
        n_bins=n_bins,
        bin_method=bin_method,
    )

    if df["bias_stderr"].fill_nan(None).null_count() > 0 and with_errorbars:
        msg = (
            "Some values of 'bias_stderr' are null. Therefore no error bars are "
            "shown for that y_pred/model, despite the fact that confidence_level>0 "
            "was set to True."
        )
        warnings.warn(msg, UserWarning, stacklevel=2)

    if "model_" in df.columns:
        col_model = "model_"
    elif "model" in df.columns:
        col_model = "model"
    else:
        col_model = None

    if feature is None:
        # We treat the predictions from different models as a feature.
        feature_name = col_model
        feature_has_nulls = False
    else:
        feature_name = array_name(feature, default="feature")
        feature_has_nulls = df[feature_name].null_count() > 0

    is_categorical = False
    is_string = False
    feature_dtype = df.get_column(feature_name).dtype
    if feature_dtype in [pl.Categorical, pl.Enum]:
        is_categorical = True
    elif feature_dtype in [pl.Utf8, pl.Object]:
        is_string = True

    n_x = df[feature_name].n_unique()

    # horizontal line at y=0
    if plot_backend == "matplotlib":
        ax.axhline(y=0, xmin=0, xmax=1, color="k", linestyle="dotted")
    else:
        fig.add_hline(y=0, line={"color": "black", "dash": "dot"}, showlegend=False)

    # bias plot
    if feature is None or col_model is None:
        pred_names = [None]
    else:
        # pred_names = df[col_model].unique() this automatically sorts
        pred_names, _ = get_sorted_array_names(y_pred)
    n_models = len(pred_names)
    with_label = feature is not None and (n_models >= 2 or feature_has_nulls)

    if (is_string or is_categorical) and feature_has_nulls:
        # We want the Null values at the end and therefore sort again.
        df = df.sort(feature_name, descending=False, nulls_last=True)

    for i, m in enumerate(pred_names):
        filter_condition = True if m is None else pl.col(col_model) == m
        df_i = df.filter(filter_condition)
        label = m if with_label else None

        if df_i["bias_stderr"].null_count() > 0:
            with_errorbars_i = False
        else:
            with_errorbars_i = with_errorbars

        if with_errorbars_i:
            # We scale bias_stderr by the corresponding value of the t-distribution
            # to get our desired confidence level.
            n = df_i["bias_count"].to_numpy()
            conf_level_fct = special.stdtrit(
                np.maximum(n - 1, 1),  # degrees of freedom, if n=0 => bias_stderr=0.
                1 - (1 - confidence_level) / 2,
            )
            df_i = df_i.with_columns(
                [(pl.col("bias_stderr") * conf_level_fct).alias("bias_stderr")]
            )

        if is_string or is_categorical:
            df_ii = df_i.filter(pl.col(feature_name).is_not_null())
            # We x-shift a little for a better visual.
            span = (n_x - 1) / n_x / n_models  # length for one cat value and one model
            x = np.arange(n_x - feature_has_nulls)
            if n_models > 1:
                x = x + (i - n_models // 2) * span * 0.5
            if plot_backend == "matplotlib":
                ax.errorbar(
                    x,
                    df_ii["bias_mean"],
                    yerr=df_ii["bias_stderr"] if with_errorbars_i else None,
                    marker="o",
                    linestyle="None",
                    capsize=4,
                    label=label,
                )
            else:
                fig.add_scatter(
                    x=x,
                    y=df_ii["bias_mean"],
                    error_y={
                        "type": "data",  # value of error bar given in data coordinates
                        "array": df_ii["bias_stderr"] if with_errorbars_i else None,
                        "width": 4,
                        "visible": True,
                    },
                    marker={"color": get_plotly_color(i)},
                    mode="markers",
                    name=label,
                )
        else:
            if with_errorbars_i:
                lower = df_i["bias_mean"] - df_i["bias_stderr"]
                upper = df_i["bias_mean"] + df_i["bias_stderr"]
                if plot_backend == "matplotlib":
                    ax.fill_between(
                        df_i[feature_name],
                        lower,
                        upper,
                        alpha=0.1,
                    )
                else:
                    # plotly has no equivalent of fill_between and needs a bit more
                    # coding
                    color = get_plotly_color(i)
                    fig.add_scatter(
                        x=pl.concat([df_i[feature_name], df_i[::-1, feature_name]]),
                        y=pl.concat([lower, upper[::-1]]),
                        fill="toself",
                        fillcolor=color,
                        hoverinfo="skip",
                        line={"color": color},
                        mode="lines",
                        opacity=0.1,
                        showlegend=False,
                    )
            if plot_backend == "matplotlib":
                ax.plot(
                    df_i[feature_name],
                    df_i["bias_mean"],
                    linestyle="solid",
                    marker="o",
                    label=label,
                )
            else:
                fig.add_scatter(
                    x=df_i[feature_name],
                    y=df_i["bias_mean"],
                    marker_symbol="circle",
                    mode="lines+markers",
                    line={"color": get_plotly_color(i)},
                    name=label,
                )

        if feature_has_nulls:
            # Null values are plotted as diamonds as rightmost point.
            df_i_null = df_i.filter(pl.col(feature_name).is_null())

            if is_string or is_categorical:
                x_null = np.array([n_x - 1])
            else:
                x_min = df_i[feature_name].min()
                x_max = df_i[feature_name].max()
                if n_x == 1:
                    # df_i[feature_name] is the null value.
                    x_null, span = np.array([0]), 1
                elif n_x == 2:
                    x_null, span = np.array([2 * x_max]), 0.5 * x_max / n_models
                else:
                    x_null = np.array([x_max + (x_max - x_min) / n_x])
                    span = (x_null - x_max) / n_models

            if n_models > 1:
                x_null = x_null + (i - n_models // 2) * span * 0.5

            if plot_backend == "matplotlib":
                color = ax.get_lines()[-1].get_color()  # previous line color
                ax.errorbar(
                    x_null,
                    df_i_null["bias_mean"],
                    yerr=df_i_null["bias_stderr"] if with_errorbars_i else None,
                    marker="D",
                    linestyle="None",
                    capsize=4,
                    label=None,
                    color=color,
                )
            else:
                fig.add_scatter(
                    x=x_null,
                    y=df_i_null["bias_mean"],
                    error_y={
                        "type": "data",  # value of error bar given in data coordinates
                        "array": df_i_null["bias_stderr"] if with_errorbars_i else None,
                        "width": 4,
                        "visible": True,
                    },
                    marker={"color": get_plotly_color(i), "symbol": "diamond"},
                    mode="markers",
                    showlegend=False,
                )

    if is_categorical or is_string:
        if feature_has_nulls:
            # Without cast to pl.Uft8, the following error might occur:
            # exceptions.ComputeError: cannot combine categorical under a global string
            # cache with a non cached categorical
            tick_labels = df_i[feature_name].cast(pl.Utf8).fill_null("Null")
        else:
            tick_labels = df_i[feature_name]
        x_label = feature_name
        if plot_backend == "matplotlib":
            ax.set_xticks(np.arange(n_x), labels=tick_labels)
        else:
            fig.update_layout(
                xaxis={
                    "tickmode": "array",
                    "tickvals": np.arange(n_x),
                    "ticktext": tick_labels,
                }
            )
    elif feature_name is not None:
        x_label = "binned " + feature_name
    else:
        x_label = ""

    if feature is None:
        title = "Bias Plot"
    else:
        model_name = array_name(y_pred, default="")
        # test for empty string ""
        title = "Bias Plot" if not model_name else "Bias Plot " + model_name

    if plot_backend == "matplotlib":
        ax.set(xlabel=x_label, ylabel="bias", title=title)
    else:
        fig.update_layout(xaxis_title=x_label, yaxis_title="bias", title=title)

    if with_label and plot_backend == "matplotlib":
        if feature_has_nulls:
            # Add legend entry for diamonds as Null values.
            # Unfortunately, the Null value legend entry often appears first, but we
            # want it at the end.
            ax.scatter([], [], marker="D", color="grey", label="Null values")
            handles, labels = ax.get_legend_handles_labels()
            if (labels[-1] != "Null values") and "Null values" in labels:
                i = labels.index("Null values")
                # i can't be the last index
                labels = labels[:i] + labels[i + 1 :] + [labels[i]]
                handles = handles[:i] + handles[i + 1 :] + [handles[i]]
            ax.legend(handles=handles, labels=labels)
        else:
            ax.legend()
    elif with_label and feature_has_nulls:
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Null values",
            marker={"size": 7, "color": "grey", "symbol": "diamond"},
        )

    return ax


def plot_marginal(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    X: npt.ArrayLike,
    feature_name: Union[str, int],
    predict_function: Optional[Callable] = None,
    weights: Optional[npt.ArrayLike] = None,
    *,
    n_bins: int = 10,
    bin_method: str = "sturges",
    n_max: int = 1000,
    rng: Optional[Union[np.random.Generator, int]] = None,
    ax: Optional[mpl.axes.Axes] = None,
    show_lines: str = "numerical",
):
    """Plot marginal observed and predicted conditional on a feature.

    This plot provides a means to inspect a model per feature.
    The average of observed and predicted are plotted as well as a histogram of the
    feature.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs)
        Predicted values, e.g. for the conditional expectation of the response,
        `E(Y|X)`.
    X : array-like of shape (n_obs, n_features)
        The dataframe or array of features to be passed to the model predict function.
    feature_name : str or int
        Column name (str) or index (int) of feature in `X`.
    predict_function : callable or None
        A callable to get prediction, i.e. `predict_function(X)`. Used to compute
        partial dependence. If `None`, partial dependence is omitted.
    weights : array-like of shape (n_obs) or None
        Case weights. If given, the bias is calculated as weighted average of the
        identification function with these weights.
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
        The method for finding bin edges (boundaries). Options using `n_bins` are:

        - `"quantile"`
        - `"uniform"`

        Options automatically selecting the number of bins for numerical features
        thereby using uniform bins are same options as
        [numpy.histogram_bin_edges](https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html):

        - `"auto"`
        Minimum bin width between the `"sturges"` and `"fd"` estimators. Provides good
        all-around performance.
        - `"fd"` (Freedman Diaconis Estimator)
        Robust (resilient to outliers) estimator that takes into account data
        variability and data size.
        - `"doane"`
        An improved version of Sturges' estimator that works better with non-normal
        datasets.
        - `"scott"`
        Less robust estimator that takes into account data variability and data size.
        - `"stone"`
        Estimator based on leave-one-out cross-validation estimate of the integrated
        squared error. Can be regarded as a generalization of Scott's rule.
        - `"rice"`
        Estimator does not take variability into account, only data size. Commonly
        overestimates number of bins required.
        - `"sturges"`
        R's default method, only accounts for data size. Only optimal for gaussian data
        and underestimates number of bins for large non-gaussian datasets.
        - `"sqrt"`
        Square root (of data size) estimator, used by Excel and other programs for its
        speed and simplicity.

    n_max : int or None
        Used only for partial dependence computation. The number of rows to subsample
        from X. This speeds up computation, in particular for slow predict functions.
    rng : np.random.Generator, int or None
        Used only for partial dependence computation. The random number generator used
        for subsampling of `n_max` rows. The input is internally wrapped by
        `np.random.default_rng(rng)`.
    ax : matplotlib.axes.Axes or plotly Figure
        Axes object to draw the plot onto, otherwise uses the current Axes.
    show_lines : str
        Option for how to display mean values and partial dependence:

        - `"always"`: Always draw lines.
        - `"numerical"`: String and categorical features are drawn as points, numerical
          ones as lines.

    Returns
    -------
    ax :
        Either the matplotlib axes or the plotly figure. This is configurable by
        setting the `plot_backend` via
        [`model_diagnostics.set_config`][model_diagnostics.set_config] or
        [`model_diagnostics.config_context`][model_diagnostics.config_context].

    Examples
    -----
    If you wish to plot multiple features at once with subfigures, here is how to do it
    with matplotlib:

    ```py
    from math import ceil
    import matplotlib.pyplot as plt
    import numpy as np
    from model_diagnostics.calibration import plot_marginal

    # Replace by your own data and model.
    n_obs = 100
    y_obs = np.arange(n_obs)
    X = np.ones((n_obs, 2))
    X[:, 0] = np.sin(np.arange(n_obs))
    X[:, 1] = y_obs ** 2

    def model_predict(X):
        s = 0.5 * n_obs * np.sin(X)
        return s.sum(axis=1) + np.sqrt(X[:, 1])

    # Now the plotting.
    feature_list = [0, 1]
    n_rows, n_cols = ceil(len(feature_list) / 2), 2
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, sharey=True)
    for i, ax in enumerate(axs):
        plot_marginal(
            y_obs=y_obs,
            y_pred=model_predict(X),
            X=X,
            feature_name=feature_list[i],
            predict_function=model_predict,
            ax=ax,
        )
    fig.tight_layout()
    ```

    For plotly, use the helper function
    [`add_marginal_subplot`][model_diagnostics.calibration.plots.add_marginal_subplot]:

    ```py
    from math import ceil
    import numpy as np
    from model_diagnostics import config_context
    from plotly.subplots import make_subplots
    from model_diagnostics.calibration import add_marginal_subplot, plot_marginal

    # Replace by your own data and model.
    n_obs = 100
    y_obs = np.arange(n_obs)
    X = np.ones((n_obs, 2))
    X[:, 0] = np.sin(np.arange(n_obs))
    X[:, 1] = y_obs ** 2

    def model_predict(X):
        s = 0.5 * n_obs * np.sin(X)
        return s.sum(axis=1) + np.sqrt(X[:, 1])

    # Now the plotting.
    feature_list = [0, 1]
    n_rows, n_cols = ceil(len(feature_list) / 2), 2
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        vertical_spacing=0.3 / n_rows,  # equals default
        # subplot_titles=feature_list,  # maybe
        specs=[[{"secondary_y": True}] * n_cols] * n_rows,  # This is important!
    )
    for row in range(n_rows):
        for col in range(n_cols):
            i = n_cols * row + col
            with config_context(plot_backend="plotly"):
                subfig = plot_marginal(
                    y_obs=y_obs,
                    y_pred=model_predict(X),
                    X=X,
                    feature_name=feature_list[i],
                    predict_function=model_predict,
                )
            add_marginal_subplot(subfig, fig, row, col)
    fig.show()
    ```

    """
    if ax is None:
        plot_backend = get_config()["plot_backend"]
        if plot_backend == "matplotlib":
            ax = plt.gca()
        else:
            from plotly.subplots import make_subplots

            # fig = ax = go.Figure()
            fig = ax = make_subplots(specs=[[{"secondary_y": True}]])
    elif isinstance(ax, mpl.axes.Axes):
        plot_backend = "matplotlib"
    elif is_plotly_figure(ax):
        plot_backend = "plotly"
        fig = ax
        # Take care to mimick make_subplots for secondary y axis.
        # The following code is by comparing
        #   make_subplots(specs=[[{"secondary_y": True}]])
        # vs
        #   go.Figure()
        if not hasattr(fig.layout, "yaxis2"):
            fig.update_layout(
                xaxis={"anchor": "y", "domain": [0.0, 0.94]},
                yaxis={"anchor": "x", "domain": [0.0, 1.0]},
                yaxis2={"anchor": "x", "overlaying": "y", "side": "right"},
            )
            SubplotRef = collections.namedtuple(  # noqa: PYI024
                "SubplotRef", ("subplot_type", "layout_keys", "trace_kwargs")
            )
            fig._grid_ref = [  # noqa: SLF001
                [
                    (
                        SubplotRef(
                            subplot_type="xy",
                            layout_keys=("xaxis", "yaxis"),
                            trace_kwargs={"xaxis": "x", "yaxis": "y"},
                        ),
                        SubplotRef(
                            subplot_type="xy",
                            layout_keys=("xaxis", "yaxis2"),
                            trace_kwargs={"xaxis": "x", "yaxis": "y2"},
                        ),
                    )
                ]
            ]
            fig._grid_str = "This is the format of your plot grid:\n[ (1,1) x,y,y2 ]\n"  # noqa: SLF001
    else:
        msg = (
            "The ax argument must be None, a matplotlib Axes or a plotly Figure, "
            f"got {type(ax)}."
        )
        raise ValueError(msg)

    if show_lines not in ("always", "numerical"):
        msg = (
            f"The argument show_lines mut be 'always' or 'numerical'; got {show_lines}."
        )
        raise ValueError(msg)

    # estimator = getattr(predict_callable, "__self__", None)
    n_pred = length_of_second_dimension(y_pred)
    if n_pred > 1:
        msg = (
            f"Parameter y_pred has shape (n_obs, {n_pred}), but only "
            "(n_obs) and (n_obs, 1) are allowd."
        )
        raise ValueError(msg)

    df = compute_marginal(
        y_obs=y_obs,
        y_pred=y_pred,
        X=X,
        feature_name=feature_name,
        predict_function=predict_function,
        weights=weights,
        n_bins=n_bins,
        bin_method=bin_method,
        n_max=n_max,
        rng=rng,
    )
    feature_name = df.columns[0]

    feature_has_nulls = df[feature_name].null_count() > 0
    n_bins_eff = df.shape[0] - feature_has_nulls
    # If df contains the columns "bin_edges", it's a numerical feature.
    is_categorical = "bin_edges" not in df.columns

    n_x = df[feature_name].n_unique()

    # marginal plot
    if is_categorical and feature_has_nulls:
        # We want the Null values at the end and therefore sort again.
        df = df.sort(feature_name, descending=False, nulls_last=True)
    df_no_nulls = df.filter(pl.col(feature_name).is_not_null())

    # Numerical columns are sometimes better treated as categorical.
    num_as_cat = False
    if not is_categorical:
        bin_edges = df_no_nulls.get_column("bin_edges")
        num_as_cat = (
            # left bin edge = right bin edge
            (bin_edges.arr.first() == bin_edges.arr.last())
            # feature == left bin edge
            | (bin_edges.arr.first() == df_no_nulls.get_column(feature_name))
            # feature == right bin edge
            | (bin_edges.arr.last() == df_no_nulls.get_column(feature_name))
            # standard deviation of feature in bin == 0
            | (bin_edges.arr.get(1) == 0)
        ).all()

    # First the histogram of weights on secondary y-axis.
    # Other graph elements should appear on top of it. For plotly, we therefore need to
    # plot the histogram on the primary y-axis and put primary to the right and
    # secondary to the left. All other plotly graphs are put on the secondary yaxis.
    #
    # We x-shift a little for a better visual.
    x = (
        np.arange(n_x - feature_has_nulls)
        if is_categorical
        else df_no_nulls[feature_name]
    )
    if plot_backend == "matplotlib":
        ax2 = ax.twinx()
        if is_categorical or num_as_cat:
            ax2.bar(
                x=x,
                height=df_no_nulls["weights"] / df["weights"].sum(),
                color="lightgrey",
            )
        else:
            # We can't use
            #   ax2.hist(
            #       x=df_no_nulls[feature_name],
            #       weights=df_no_nulls["weights"] / df["weights"].sum(),
            #       bins=np.r_[bin_edges[0][0], bin_edges.arr.last()],  # n_bins_eff,
            #       color="lightgrey",
            #       edgecolor="grey",
            #       rwidth=0.8 if n_bins_eff <= 2 else None,
            #   )
            # because we might have empty bins.
            ax2.bar(
                x=0.5 * (bin_edges.arr.last() + bin_edges.arr.first()),
                height=df_no_nulls["weights"] / df["weights"].sum(),
                width=(bin_edges.arr.last() - bin_edges.arr.first())
                * (1 if n_bins_eff > 2 else 0.8),
                color="lightgrey",
                edgecolor="grey",
            )
        # https://stackoverflow.com/questions/30505616/how-to-arrange-plots-of-secondary-axis-to-be-below-plots-of-primary-axis-in-matp
        ax.set_zorder(ax2.get_zorder() + 1)
        ax.set_frame_on(False)
    else:
        if is_categorical or num_as_cat:
            # fig.add_histogram(
            #     x=x, # df_no_nulls[feature_name],
            #     y=df_no_nulls["weights"] / df["weights"].sum(),
            #     histfunc="sum",
            #     marker={"color": "lightgrey"},
            #     secondary_y=False,
            #     showlegend=False,
            # )
            fig.add_bar(
                x=x,
                y=df_no_nulls["weights"] / df["weights"].sum(),
                marker={"color": "lightgrey"},
                secondary_y=False,
                showlegend=False,
            )
        else:
            fig.add_bar(
                x=0.5 * (bin_edges.arr.last() + bin_edges.arr.first()),
                y=df_no_nulls["weights"] / df["weights"].sum(),
                width=bin_edges.arr.last() - bin_edges.arr.first(),
                marker={"color": "lightgrey", "line": {"width": 1.0, "color": "grey"}},
                secondary_y=False,
                showlegend=False,
            )
        fig.update_layout(yaxis_side="right", yaxis2_side="left")
        if n_bins_eff <= 2:
            fig.update_layout(bargap=0.2)

    if feature_has_nulls:
        df_null = df.filter(pl.col(feature_name).is_null())
        # Null values are plotted as rightmost point at x_null.
        if is_categorical:
            x_null = np.array([n_x - 1])
            # matplotlib default width = 0.8
            width = 0.8 if plot_backend == "matplotlib" else None
        else:
            x_min = df[feature_name].min()
            x_max = df[feature_name].max()
            if n_x == 1:
                # df[feature_name] is the null value.
                x_null = np.array([0])
            elif n_x == 2:
                x_null = np.array([2 * x_max])
            else:
                x_null = np.array([x_max + (x_max - x_min) / n_x])
            width = x_null - bin_edges.arr.last().max()
            if width is not None and width <= 0:
                width = (x_max - x_min) / n_x / 2.0

        # Null value histogram
        if plot_backend == "matplotlib":
            ax2.bar(
                x=x_null,
                height=df_null["weights"] / df["weights"].sum(),
                width=width,
                color="lightgrey",
            )
        else:
            fig.add_bar(
                x=x_null,
                y=df_null["weights"] / df["weights"].sum(),
                width=width,
                marker={"color": "lightgrey"},
                secondary_y=False,
                showlegend=False,
            )

    plot_items = ["y_obs_mean", "y_pred_mean"]
    if predict_function is not None:
        plot_items.append("partial_dependence")
    label_dict = {
        "y_obs_mean": "mean y_obs",
        "y_pred_mean": "mean y_pred",
        "partial_dependence": "partial dependence",
    }
    for i, m in enumerate(plot_items):
        label = label_dict[m]
        if plot_backend == "matplotlib":
            linestyle = "dashed" if m == "partial_dependence" else "solid"
        else:
            line = {
                "color": get_plotly_color(i),
                "dash": "dash" if m == "partial_dependence" else None,
            }
        if is_categorical:
            # We x-shift a little for a better visual.
            x = np.arange(n_x - feature_has_nulls)
            if plot_backend == "matplotlib":
                ax.plot(
                    x,
                    df_no_nulls[m],
                    marker="o",
                    linestyle="None" if show_lines == "numerical" else linestyle,
                    label=label,
                )
            else:
                fig.add_scatter(
                    x=x,
                    y=df_no_nulls[m],
                    marker={"color": get_plotly_color(i)},
                    mode="markers" if show_lines == "numerical" else "lines+markers",
                    line=None if show_lines == "numerical" else line,
                    name=label,
                    secondary_y=True,
                )
        elif plot_backend == "matplotlib":
            ax.plot(
                df[feature_name],
                df[m],
                linestyle=linestyle,
                marker="o",
                label=label,
            )
        else:
            fig.add_scatter(
                x=df[feature_name],
                y=df[m],
                marker_symbol="circle",
                mode="lines+markers",
                line=line,
                name=label,
                secondary_y=True,
            )

        if feature_has_nulls:
            # Null values are plotted as diamonds as rightmost point.
            if plot_backend == "matplotlib":
                color = ax.get_lines()[-1].get_color()  # previous line color
                ax.plot(
                    x_null,
                    df_null[m],
                    marker="D",
                    linestyle="None",
                    label=None,
                    color=color,
                )
            else:
                fig.add_scatter(
                    x=x_null,
                    y=df_null[m],
                    marker={"color": get_plotly_color(i), "symbol": "diamond"},
                    mode="markers",
                    secondary_y=True,
                    showlegend=False,
                )

    if is_categorical:
        if df[feature_name].null_count() > 0:
            # Without cast to pl.Uft8, the following error might occur:
            # exceptions.ComputeError: cannot combine categorical under a global string
            # cache with a non cached categorical
            tick_labels = df[feature_name].cast(pl.Utf8).fill_null("Null")
        else:
            tick_labels = df[feature_name]
        x_label = feature_name
        if plot_backend == "matplotlib":
            ax.set_xticks(np.arange(n_x), labels=tick_labels)
        else:
            fig.update_layout(
                xaxis={
                    "tickmode": "array",
                    "tickvals": np.arange(n_x),
                    "ticktext": tick_labels,
                }
            )
    elif feature_name is not None:
        x_label = "binned " + str(feature_name)
    else:
        x_label = ""

    model_name = array_name(y_pred, default="")
    # test for empty string ""
    title = "Marginal Plot" if not model_name else "Marginal Plot " + model_name

    if plot_backend == "matplotlib":
        ax.set(xlabel=x_label, ylabel="y", title=title)
    else:
        fig.update_layout(xaxis_title=x_label, yaxis2_title="y", title=title)
        fig["layout"]["yaxis"]["showgrid"] = False

    if plot_backend == "matplotlib":
        if feature_has_nulls:
            # Add legend entry for diamonds as Null values.
            # Unfortunately, the Null value legend entry often appears first, but we
            # want it at the end.
            ax.scatter([], [], marker="D", color="grey", label="Null values")
            handles, labels = ax.get_legend_handles_labels()
            if (labels[-1] != "Null values") and "Null values" in labels:
                i = labels.index("Null values")
                # i can't be the last index
                labels = labels[:i] + labels[i + 1 :] + [labels[i]]
                handles = handles[:i] + handles[i + 1 :] + [handles[i]]
            ax.legend(handles=handles, labels=labels)
        else:
            ax.legend()
    elif feature_has_nulls:
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            name="Null values",
            marker={"size": 7, "color": "grey", "symbol": "diamond"},
            secondary_y=True,
        )

    return ax


def add_marginal_subplot(subfig, fig, row: int, col: int):
    """Add a plotly subplot from plot_marginal to a multi-plot figure.

    This auxiliary function is accompanies
    [`plot_marginal`][model_diagnostics.calibration.plot_marginal] in order to ease
    plotting with subfigures with the plotly backend.

    For it to work, you must call plotly's `make_subplots` with the `specs` argument
    and set the appropriate number of `{"secondary_y": True}` in a list of lists.
    ```py hl_lines="7"
    from plotly.subplots import make_subplots

    n_rows, n_cols = ...
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"secondary_y": True}] * n_cols] * n_rows,  # This is important!
    )
    ```
    The reason is that `plot_marginal` uses a secondary yaxis (and swapped sides with
    the primary yaxis).

    Parameters
    ----------
    subfig : plotly Figure
        The subfigure which is added to `fig`.
    fig : plotly Figure
        The multi-plot figure to which `subfig` is added at positions `row` and `col`.
    row : int
        The (0-based) row index of `fig` at which `subfig` is added.
    col : int
        The (0-based) column index of `fig` at which `subfig` is added.

    Returns
    -------
    fig
        The plotly figure `fig`.
    """
    # It returns a tuple of `range`s starting at 1.
    plotly_rows, plotly_cols = fig._get_subplot_rows_columns()  # noqa: SLF001
    n_rows = len(plotly_rows)
    n_cols = len(plotly_cols)
    if row >= n_rows or col >= n_cols:
        msg = (
            f"The `fig` only has {n_rows} rows and {n_cols} columns. You specified "
            f"(0-based) {row=} and {col=}."
        )
        raise ValueError(msg)
    i = n_cols * row + col
    # Plotly uses 1-based indices:
    row += 1
    col += 1
    # Transfer the x-axis titles of the subfig to fig.
    xaxis = "xaxis" if i == 0 else f"xaxis{i + 1}"
    fig["layout"][xaxis]["title"] = subfig["layout"]["xaxis"]["title"]
    # Change sides of y-axis.
    yaxis = "yaxis" if i == 0 else f"yaxis{2 * i + 1}"
    yaxis2 = f"yaxis{2 * (i + 1)}"
    fig.update_layout(
        **{
            yaxis: {"side": "right", "showgrid": False},
            yaxis2: {"side": "left", "title": "y"},
        }
    )
    # Only the last added subfig should show the legends, but all the ones before
    # should not.
    # So don't show legends for row-1 and col-1.
    if row > 1:
        fig.update_traces(patch={"showlegend": False}, row=row - 1, col=col)
    if col > 1:
        fig.update_traces(patch={"showlegend": False}, row=row, col=col - 1)
    for d in subfig.data:
        fig.add_trace(d, row=row, col=col, secondary_y=d["yaxis"] == "y2")

    fig.update_layout(title=subfig.layout.title.text)
