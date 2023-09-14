import numbers
from typing import Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from model_diagnostics._utils._array import (
    get_array_min_max,
    get_second_dimension,
    get_sorted_array_names,
    length_of_second_dimension,
)

from .scoring import ElementaryScore


def plot_murphy_diagram(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    weights: Optional[npt.ArrayLike] = None,
    *,
    etas: Union[int, npt.ArrayLike] = 100,
    functional: str = "mean",
    level: float = 0.5,
    ax: Optional[mpl.axes.Axes] = None,
):
    r"""Plot a Murphy diagram.

    A Murphy diagram plots the scores of elementary scoring functions `ElementaryScore`
    over a range of their free parameter `eta`. This shows, if a model dominates all
    others over a wide class of scoring functions or if the ranking is very much
    dependent on the choice of scoring function.
    See [Notes](#notes) for further details.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs) or (n_obs, n_models)
        Predicted values of the conditional expectation of Y, `E(Y|X)`.
    weights : array-like of shape (n_obs) or None
        Case weights.
    etas : int or array-like
        If an integer is given, equidistant points between min and max y values are
        generater. If an array-like is given, those points are used.
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
    ax : matplotlib.axes.Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax

    Notes
    -----
    For details, refer to `[Ehm2015]`.

    References
    ----------
    `[Ehm2015]`

    :   W. Ehm, T. Gneiting, A. Jordan, F. Krüger.
        "Of Quantiles and Expectiles: Consistent Scoring Functions, Choquet
        Representations, and Forecast Rankings".
        [arxiv:1503.08195](https://arxiv.org/abs/1503.08195).
    """
    if ax is None:
        ax = plt.gca()

    if (n_cols := length_of_second_dimension(y_obs)) > 0:
        if n_cols == 1:
            y_obs = get_second_dimension(y_obs, 0)
        else:
            msg = (
                f"Array-like y_obs has more than 2 dimensions, y_obs.shape[1]={n_cols}"
            )
            raise ValueError(msg)

    y_pred_min, y_pred_max = get_array_min_max(y_pred)
    y_obs_min, y_obs_max = get_array_min_max(y_obs)
    y_min, y_max = min(y_pred_min, y_obs_min), max(y_pred_max, y_obs_max)

    if y_min == y_max:
        msg = "All values y_obs and y_pred are one single and same value."
        raise ValueError(msg)
    elif isinstance(etas, numbers.Integral):
        etas = np.linspace(y_min, y_max, num=etas, endpoint=True)
    else:
        etas = np.asarray(etas).astype(float)
        if etas.ndim > 1:
            etas = etas.reshape(max(etas.shape))

    def elementary_score(y_obs, y_pred, weights, eta):
        sf = ElementaryScore(eta, functional=functional, level=level)
        return sf(y_obs=y_obs, y_pred=y_pred, weights=weights)

    n_pred = length_of_second_dimension(y_pred)
    pred_names, _ = get_sorted_array_names(y_pred)

    for i in range(len(pred_names)):
        y_pred_i = y_pred if n_pred == 0 else get_second_dimension(y_pred, i)

        y_plot = [
            elementary_score(y_obs=y_obs, y_pred=y_pred_i, weights=weights, eta=eta)
            for eta in etas
        ]
        label = pred_names[i] if n_pred >= 2 else None
        ax.plot(etas, y_plot, label=label)

    title = "Murphy Diagram"
    ax.set(xlabel="eta", ylabel="score")

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
