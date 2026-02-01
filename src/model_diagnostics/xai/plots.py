from typing import Callable, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from scipy import special

from model_diagnostics import get_config
from model_diagnostics._utils.plot_helper import is_plotly_figure
from model_diagnostics.scoring import SquaredError
from model_diagnostics.xai import compute_permutation_importance


def plot_permutation_importance(
    pred_fun: Callable,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    features: Optional[Union[list, tuple, set, dict]] = None,
    scoring_function: Callable = SquaredError(),
    weights: Optional[npt.ArrayLike] = None,
    n_repeats: int = 5,
    n_max: int = 10_000,
    scoring_orientation: str = "smaller",
    rng: Optional[Union[np.random.Generator, int]] = None,
    max_display: int = 15,
    which: str = "difference",
    confidence_level: float = 0.95,
    ax: Optional[mpl.axes.Axes] = None,
):
    """
    Plot permutation importance as barplot with confidence intervals.

    Parameters
    ----------
    pred_fun : callable
        A callable to get predictions, i.e. `pred_fun(X)`.
    X : array-like of shape (n_obs, n_features)
        The dataframe or array of features to be passed to the model predict function.
    y : npt.ArrayLike
        1D array of shape (n_observations,) containing the target values.
    features: list, tuple, dict, default=None
        Iterable of feature names/indices of features in `X`. The default None
        will use all features in `X`. Can also be a dictionary with lists of feature
        names/indices as values. The keys of the dictionary are used as feature group
        names. Example: `{"x1": ["x1"], "x2": ["x2"], "size": ["x1", "x2"]}`.
        Passing a dictionary is also useful if you want to represent feature indices
        of a numpy array as strings. Example: `{"area": 0, "age": 1}`.
    scoring_function : callable, default=SquaredError()
        A scoring function with signature roughly
        `fun(y_obs, y_pred, weights) -> float`.
    weights : array-like of shape (n_obs) or None, default=None
        Case weights passed to the scoring_function.
    n_repeats : int, default=5
        Number of times to repeat the permutation for each feature group.
    n_max : int or None, default=10_000
        Maximum number of observations used. If the number of observations is greater
        than `n_max`, a random subset of size `n_max` will be drawn from `X`, `y`, (and
        `weights`). Pass None for no subsampling.
    scoring_orientation : str, default="smaller"
        Direction of scoring function. Use "smaller" if smaller values are better
        (e.g., average losses), or "greater" if greater values are better
        (e.g., R-squared).
    rng : np.random.Generator, int or None, default=None
        The random number generator used for shuffling values and for subsampling
        `n_max` rows. The input is internally wrapped by `np.random.default_rng(rng)`.
    max_display : int or None, optional
        Maximum number of features to display, by default 15.
        If None, all features are displayed.
    which : str, default="difference"
        Should difference or ratio scores be shown? Either "difference" or "ratio".
    confidence_level : float, default=0.95
        Confidence level for error bars. If 0, no error bars are plotted. Value must
        fulfil `0 <= confidence_level < 1`. Set to 0.683 to show standard errors.
    ax : matplotlib.axes.Axes or plotly Figure, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax :
        Either the matplotlib axes or the plotly figure.
    """
    if max_display is not None and max_display < 1:
        msg = f"Argument max_display must be None or >=1, got {max_display}."
        raise ValueError(msg)

    if which not in ("difference", "ratio"):
        msg = f"Unknown normalization method: {which}"
        raise ValueError(msg)

    if not (0 <= confidence_level < 1):
        msg = (
            f"Argument confidence_level must fulfil 0 <= confidence_level < 1, got "
            f"{confidence_level}."
        )
        raise ValueError(msg)

    df = compute_permutation_importance(
        pred_fun=pred_fun,
        X=X,
        y=y,
        features=features,
        scoring_function=scoring_function,
        weights=weights,
        n_repeats=n_repeats,
        n_max=n_max,
        scoring_orientation=scoring_orientation,
        rng=rng,
    )
    # Plot axes are reversed
    df = df.sort(which + "_mean", descending=False)

    if max_display is not None:
        df = df.tail(max_display)

    feature_groups = df["feature"]
    importances = df[which + "_mean"]
    if confidence_level > 0 and n_repeats >= 2:
        xerr = df[which + "_stderr"] * special.stdtrit(
            n_repeats - 1, (1 + confidence_level) / 2
        )
    else:
        xerr = None

    # set-up backend
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

    # bars
    title = "Permutation Feature Importance"
    xlab = "Importance"
    if plot_backend == "matplotlib":
        y_pos = np.arange(len(feature_groups))
        _ = ax.barh(y_pos, importances, xerr=xerr)
        ax.set_yticks(y_pos, labels=feature_groups)
        ax.set_xlabel(xlab)
        ax.set_title(title)
    else:
        fig.add_bar(
            y=feature_groups,
            x=importances,
            orientation="h",
            error_x={"array": xerr, "width": 0},
        )
        fig.update_layout(
            xaxis_title=xlab, yaxis_title=None, title=title, yaxis_type="category"
        )

    return ax
