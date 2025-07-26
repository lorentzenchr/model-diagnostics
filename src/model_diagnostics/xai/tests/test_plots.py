import matplotlib.pyplot as plt
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

from model_diagnostics import config_context
from model_diagnostics._utils.plot_helper import get_title, get_xlabel
from model_diagnostics.xai import plot_permutation_importance


@pytest.mark.parametrize(
    ("param", "value", "msg"),
    [
        ("max_display", 0, "Argument max_display must be None or >=1, got 0."),
        ("which", "invalid_method", "Unknown normalization method: invalid_method"),
        (
            "confidence_level",
            1,
            "Argument confidence_level must fulfil 0 <= confidence_level < 1, got 1.",
        ),
        (
            "ax",
            "XXX",
            "The ax argument must be None, a matplotlib Axes or a plotly Figure",
        ),
    ],
)
def test_plot_permutation_importance_raises(param, value, msg):
    """Test that plot_permutation_importance raises errors."""
    X, y = make_regression(n_samples=100, random_state=1, n_features=3)
    lr = LinearRegression()
    lr.fit(X, y)

    kwargs = {param: value}

    with pytest.raises(ValueError, match=msg):
        plot_permutation_importance(lr.predict, X=X, y=y, **kwargs)


@pytest.mark.parametrize("max_display", [None, 2])
@pytest.mark.parametrize("confidence_level", [0, 0.9, 0.95])
@pytest.mark.parametrize("ax", [None, plt.subplots()[1], "plotly"])
@pytest.mark.parametrize("plot_backend", ["matplotlib", "plotly"])
def test_plot_permutation_importance(max_display, confidence_level, ax, plot_backend):
    """Test that plot_permutation_importance works."""
    if plot_backend == "plotly" or ax == "plotly":
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

    if ax == "plotly":
        ax = go.Figure()

    X, y = make_regression(n_samples=100, random_state=1, n_features=3)
    lr = LinearRegression()
    lr.fit(X, y)

    with config_context(plot_backend=plot_backend):
        plt_ax = plot_permutation_importance(
            predict_function=lr.predict,
            X=X,
            y=y,
            max_display=max_display,
            confidence_level=confidence_level,
            ax=ax,
        )

    if ax is not None:
        assert ax is plt_ax

    assert get_xlabel(plt_ax) == "Importance"
    assert get_title(plt_ax) == "Permutation Feature Importance"
