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
    predict_function: Callable,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    features: Optional[Union[list, tuple, set, dict]] = None,
    scoring_function: Callable = SquaredError(),
    weights: Optional[npt.ArrayLike] = None,
    n_repeats: int = 5,
    n_max: Optional[int] = 10_000,
    method: str = "difference",
    scoring_direction: str = "smaller",
    rng: Optional[Union[np.random.Generator, int]] = None,
    max_display: Optional[int] = 15,
    error_bars: Optional[str] = "se",
    confidence_level: float = 0.95,
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
    predict_function : callable
        A callable to get predictions, i.e. `predict_function(X)`.
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
    method : str, default="difference"
        Normalization method for the importance scores. The options are: "difference",
        "ratio", and "raw" (no normalization).
    scoring_direction : str, default="smaller"
        Direction of scoring function. Use "smaller" if smaller values are better
        (e.g., average losses), or "greater" if greater values are better
        (e.g., R-squared).
    rng : np.random.Generator, int or None, default=None
        The random number generator used for shuffling values and for subsampling
        `n_max` rows. The input is internally wrapped by `np.random.default_rng(rng)`.
    max_display : int or None, optional
        Maximum number of features to display, by default 15.
        If None, all features are displayed.
    error_bars : str or None, optional
        Error bars to display. Can be "se" (standard error), "std" (standard deviation),
        "ci" (t confidence interval), or None. Default is "se". Only if `n_repeats > 1`.
    confidence_level: float
        Confidence level of the approximate t confidence interval.
        Default is 0.95. Only used if `error_bars="ci"`.
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

    if error_bars is not None and n_repeats >= 2:
        if error_bars not in ("se", "std", "ci"):
            msg = (
                f"Argument error_bars must be one of 'se', 'std', 'ci', or None, got "
                f"{error_bars}."
            )
            raise ValueError(msg)
        if error_bars in ("se", "ci") and not (0 < confidence_level < 1):
            msg = (
                f"Argument confidence_level must fulfil 0 < confidence_level < 1, got "
                f"{confidence_level}."
            )
            raise ValueError(msg)
        with_error_bars = True
    else:
        with_error_bars = False

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
        scoring_direction=scoring_direction,
        rng=rng,
    ).reverse()  # because the plot axes are reversed as well

    if max_display is not None:
        df = df.tail(max_display)

    feature_groups = df["feature"]
    importances = df["importance"]

    # length of error bars
    if with_error_bars:
        xerr = df["standard_deviation"]
        if error_bars in ("se", "ci"):
            xerr /= np.sqrt(n_repeats)
            if error_bars == "ci":
                xerr *= special.stdtrit(n_repeats - 1, (1 + confidence_level) / 2)
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
