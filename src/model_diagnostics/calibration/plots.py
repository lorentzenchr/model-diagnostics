import typing

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.isotonic import IsotonicRegression


def plot_reliability_diagram(
    y_obs : npt.ArrayLike,
    y_pred : npt.ArrayLike,
    y_min : typing.Optional[float] = 0,
    y_max : typing.Optional[float] = 1,
    *,
    ax = None,
):
    """Plot a reliability diagram.

    A reliability diagram calibration curve plots the E(y_obs|y_pred) vs y_pred.

    Parameters
    ----------
    y_obs : array-like, shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like, shape (n_obs)
        Predicted values of the conditional expectation of Y, :math:`E(Y|X)`.
    ax : matplotlib.axes.Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax
    """
    if ax is None:
        ax = plt.gca()
    
    y_obs_min, y_obs_max = np.min(y_obs), np.max(y_obs)
    y_pred_min, y_pred_max = np.min(y_pred), np.max(y_pred)
    iso = IsotonicRegression(y_min=y_obs_min, y_max=y_obs_max).fit(y_pred, y_obs)
    ax.plot([y_pred_min, y_pred_max], [y_pred_min, y_pred_max], "k:")
    ax.plot(iso.X_thresholds_, iso.y_thresholds_, "s-")
    ax.set(xlabel="prediction for E(Y|X)", ylabel="estimated E(Y|prediction)")

    return ax