import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from sklearn.isotonic import IsotonicRegression

from .._utils.array import array_name
from .identification import compute_bias


def plot_reliability_diagram(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    *,
    ax=None,
):
    r"""Plot a reliability diagram.

    A reliability diagram or calibration curve assess auto-calibration. It plots the
    conditional expectation given the predictions (y-axis) vs the predictions (x-axis).
    The conditional expectation is estimated via isotonic regression (PAV algorithm)
    of `y_obs` on `y_pred`.
    See Notes for further details.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs)
        Predicted values of the conditional expectation of Y, \(E(Y|X)\).
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

    y_obs_min, y_obs_max = np.min(y_obs), np.max(y_obs)
    y_pred_min, y_pred_max = np.min(y_pred), np.max(y_pred)
    iso = IsotonicRegression(y_min=y_obs_min, y_max=y_obs_max).fit(y_pred, y_obs)
    # diagonal line
    ax.plot([y_pred_min, y_pred_max], [y_pred_min, y_pred_max], "k:")
    # reliability curve
    ax.plot(iso.X_thresholds_, iso.y_thresholds_)
    ax.set(xlabel="prediction for E(Y|X)", ylabel="estimated E(Y|prediction)")
    model_name = array_name(y_pred, default="")
    if model_name == "":
        ax.set_title("Reliability Diagram")
    else:
        ax.set_title("Reliability Diagram " + model_name)
    return ax


def plot_bias(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    feature: npt.ArrayLike,
    *,
    functional: str = "mean",
    level: float = 0.5,
    n_bins: int = 10,
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
    y_pred : array-like of shape (n_obs)
        Predicted values of the conditional expectation of Y, :math:`E(Y|X)`.
    feature : array-like of shape (n_obs)
        Some feature column.
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
        [arxiv:https://arxiv.org/abs/2202.12780](https://arxiv.org/abs/2202.12780).
    """
    if ax is None:
        ax = plt.gca()

    feature_name = array_name(feature, default="feature")
    df = compute_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        feature=feature,
        functional=functional,
        level=level,
        n_bins=n_bins,
    )

    is_categorical = False
    is_string = False
    if pa.types.is_dictionary(df.column(feature_name).type):
        is_categorical = True
    elif pa.types.is_string(df.column(feature_name).type):
        is_string = True

    # horizontal line at y=0
    if is_categorical or is_string:
        min_max = {"min": 0, "max": df.shape[0]}
    else:
        min_max = pc.min_max(df[feature_name]).as_py()
    ax.hlines(0, min_max["min"], min_max["max"], color="k", linestyles="dotted")
    # bias plot
    if df["bias_stderr"].null_count > 0:

        ax.plot(df[feature_name], df["bias_mean"], "o-")
    else:
        ax.errorbar(
            df[feature_name],
            df["bias_mean"],
            yerr=df["bias_stderr"],
            fmt="o-",
            capsize=4,
        )
    if is_categorical or is_string:
        ax.set(xlabel=feature_name, ylabel="bias")
    else:
        ax.set(xlabel="binned " + feature_name, ylabel="bias")
    model_name = array_name(y_pred, default="")
    if model_name == "":
        ax.set_title("Bias Plot")
    else:
        ax.set_title("Bias Plot " + model_name)

    return ax
