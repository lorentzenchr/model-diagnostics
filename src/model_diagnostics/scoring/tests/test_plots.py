import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model_diagnostics import config_context
from model_diagnostics._utils.plot_helper import (
    get_legend_list,
    get_title,
    get_xlabel,
    get_ylabel,
)
from model_diagnostics.scoring import plot_murphy_diagram


@pytest.mark.parametrize(
    ("param", "value", "msg"),
    [
        ("etas", [[1, 2], [3, 4]], "cannot reshape array of size 4 into shape"),
        (
            "y",
            [[1, 1], [1, 1]],
            "All values y_obs and y_pred are one single and same value",
        ),
        (
            "ax",
            "XXX",
            "The ax argument must be None, a matplotlib Axes or a plotly Figure",
        ),
    ],
)
def test_plot_murphy_diagram_raises(param, value, msg):
    """Test that plot_murphy_diagram raises errors."""
    if param == "y":
        y_obs, y_pred = value[0], value[1]
        kwargs = {}
    else:
        y_obs = [0, 1]
        y_pred = [-1, 1]
        kwargs = {param: value}
    with pytest.raises(ValueError, match=msg):
        plot_murphy_diagram(y_obs=y_obs, y_pred=y_pred, **kwargs)


def test_plot_murphy_diagram_raises_y_obs_multdim():
    """Test that plot_murphy_diagram raises errors for y_obs.ndim > 1."""
    y_obs = [[0], [1]]
    y_pred = [-1, 1]
    plot_murphy_diagram(y_obs=y_obs, y_pred=y_pred)
    y_obs = [[0, 1], [1, 2]]
    with pytest.raises(ValueError, match="Array-like y_obs has more than 2 dimensions"):
        plot_murphy_diagram(y_obs=y_obs, y_pred=y_pred)


@pytest.mark.parametrize(
    ("functional", "level"), [("expectile", 0.5), ("quantile", 0.8)]
)
@pytest.mark.parametrize("etas", [10, np.arange(10)])
@pytest.mark.parametrize("weights", [None, True])
@pytest.mark.parametrize("ax", [None, plt.subplots()[1], "plotly"])
@pytest.mark.parametrize("plot_backend", ["matplotlib", "plotly"])
def test_plot_murphy_diagram(functional, level, etas, weights, ax, plot_backend):
    """Test that plot_murphy_diagram works."""
    if plot_backend == "plotly" or ax == "plotly":
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

    if ax == "plotly":
        ax = go.Figure()

    X, y = make_classification(random_state=42, n_classes=2)
    if weights is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        w_train, w_test = None, None
    else:
        weights = np.random.default_rng(42).integers(low=0, high=10, size=y.shape)
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, random_state=0
        )
    clf = LogisticRegression(solver="newton-cholesky")
    clf.fit(X_train, y_train, w_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    with config_context(plot_backend=plot_backend):
        plt_ax = plot_murphy_diagram(
            y_obs=y_test,
            y_pred=y_pred,
            weights=w_test,
            etas=etas,
            functional=functional,
            level=level,
            ax=ax,
        )

    if ax is not None:
        assert ax is plt_ax

    assert get_xlabel(plt_ax) == "eta"
    assert get_ylabel(plt_ax) == "score"
    assert get_title(plt_ax) == "Murphy Diagram"

    with config_context(plot_backend=plot_backend):
        plt_ax = plot_murphy_diagram(
            y_obs=y_test,
            y_pred=pl.Series(values=y_pred, name="simple"),
            weights=w_test,
            etas=etas,
            functional=functional,
            level=level,
            ax=ax,
        )
    assert get_title(plt_ax) == "Murphy Diagram simple"


@pytest.mark.parametrize("plot_backend", ["matplotlib", "plotly"])
def test_plot_murphy_diagram_multiple_predictions(plot_backend):
    """Test that plot_murphy_diagram works for multiple predictions."""
    if plot_backend == "plotly":
        pytest.importorskip("plotly")

    n_obs = 10
    y_obs = np.arange(n_obs)
    y_obs[::2] = 0
    y_pred = pl.DataFrame({"model_2": np.ones(n_obs), "model_1": 3 * np.ones(n_obs)})
    fig, ax = plt.subplots()
    with config_context(plot_backend=plot_backend):
        plt_ax = plot_murphy_diagram(
            y_obs=y_obs,
            y_pred=y_pred,
            ax=ax,
        )
    assert get_title(plt_ax) == "Murphy Diagram"
    legend_text = get_legend_list(plt_ax)
    assert len(legend_text) == 2
    assert legend_text[0] == "model_2"
    assert legend_text[1] == "model_1"
