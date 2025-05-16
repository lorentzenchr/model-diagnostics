from typing import Callable, Optional, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import scipy

from model_diagnostics import get_config
from model_diagnostics._utils.plot_helper import is_plotly_figure
from model_diagnostics.scoring import SquaredError
from model_diagnostics.xai import compute_permutation_importance


def plot_permutation_importance(
    predict_function: Callable,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    features: Optional[Union[list, tuple, set, dict]] = None,
    scoring_function: Callable = SquaredError(),
    weights: Optional[npt.ArrayLike] = None,
    n_repeats: Optional[int] = 5,
    n_max: Optional[int] = 10_000,
    method: Optional[str] = "difference",
    smaller_is_better: Optional[bool] = True,
    rng: Optional[Union[np.random.Generator, int]] = None,
    max_display: Optional[int] = 15,
    error_bars: Optional[str] = "se",
    confidence_level: Optional[float] = 0.95,
    ax: Optional[mpl.axes.Axes] = None,
):
    """
    Plot permutation feature importance as a horizontal barplot with error bars.

    Note that error bars are representing standard errors of the mean importance
    (over the n_repeats). To get standard deviations, use `error_bars="std"`.
    For Student confidence intervals, use `error_bars="ci"` along with the argument
    `confidence_level=0.95`.

    Parameters
    ----------
    Same as compute_permutation_importance, plus:
    max_display : int, optional
        Maximum number of features to display.
    error_bars : str, optional
        Error bars to display. Can be "se" (standard error), "std" (standard deviation),
        "ci" (t confidence interval) or None. Default is "se". Only if `n_repeats > 1`.
    confidence_level: Confidence level of the approximate t confidence interval.
        Default is 0.95. Only used if `error_bars="ci"`.
    ax : matplotlib.axes.Axes or plotly Figure, optional
        Axes object to draw the plot onto, otherwise uses the current Axes.

    Returns
    -------
    ax :
        Either the matplotlib axes or the plotly figure.
    """
    df = compute_permutation_importance(
        predict_function=predict_function,
        X=X,
        y=y,
        features=features,
        scoring_function=scoring_function,
        weights=weights,
        n_repeats=n_repeats,
        n_max=n_max,
        method=method,
        smaller_is_better=smaller_is_better,
        rng=rng,
    )

    df = df.head(max_display).reverse()

    features = df["feature"]
    importances = df["importance"]

    if n_repeats > 1 and error_bars is not None:
        if error_bars not in ("se", "std", "ci"):
            msg = (
                f"error_bars must be 'se', 'std', 'ci' or None, got {error_bars}."
            )
            raise ValueError(msg)
        if error_bars in ("se", "ci"):
            xerr = df["standard_deviation"] / np.sqrt(n_repeats)
            if error_bars == "ci":
                xerr *= scipy.stats.t.ppf((1 + confidence_level) / 2, n_repeats - 1)
        if error_bars == "std":
            xerr = df["standard_deviation"]
    else:
        xerr = None

    # Setup backend
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

    # Plot
    title = "Permutation Feature Importance"
    xlab = "Importance"
    if plot_backend == "matplotlib":
        y_pos = np.arange(len(features))
        _ = ax.barh(y_pos, importances, xerr=xerr)
        ax.set_yticks(y_pos, labels=features)
        ax.set_xlabel(xlab)
        ax.set_title(title)
    else:
        fig.add_bar(
            y=features,
            x=importances,
            orientation="h",
            error_x={"array": xerr, "width": 0}
        )
        fig.update_layout(
            xaxis_title=xlab,
            yaxis_title=None,
            title=title,
            yaxis_type="category"
        )

    return ax
