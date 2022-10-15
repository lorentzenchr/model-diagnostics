import typing

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression


def plot_reliability_diagram(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    y_min: typing.Optional[float] = 0,
    y_max: typing.Optional[float] = 1,
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
    ax.plot([y_pred_min, y_pred_max], [y_pred_min, y_pred_max], "k:")
    ax.plot(iso.X_thresholds_, iso.y_thresholds_)
    ax.set(xlabel="prediction for E(Y|X)", ylabel="estimated E(Y|prediction)")
    ax.set_title("Reliability Diagram")

    return ax
