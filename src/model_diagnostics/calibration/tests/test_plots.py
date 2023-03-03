import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model_diagnostics.calibration import plot_bias, plot_reliability_diagram


@pytest.mark.parametrize(
    ("param", "value", "msg"),
    [("diagram_type", "XXX", "Parameter diagram_type must be either.*XXX")],
)
def test_plot_reliability_diagram_raises(param, value, msg):
    """Test that plot_reliability_diagram raises errors."""
    y_obs = [0, 1]
    y_pred = [-1, 1]
    with pytest.raises(ValueError, match=msg):
        plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred, **{param: value})


@pytest.mark.parametrize("diagram_type", ["reliability", "bias"])
@pytest.mark.parametrize("n_bootstrap", [None, 10])
@pytest.mark.parametrize("weights", [None, True])
@pytest.mark.parametrize("ax", [None, plt.subplots()[1]])
def test_plot_reliability_diagram(diagram_type, n_bootstrap, weights, ax):
    """Test that plot_reliability_diagram works."""
    X, y = make_classification(random_state=42)
    if weights is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        w_train, w_test = None, None
    else:
        weights = np.random.default_rng(42).integers(low=0, high=10, size=y.shape)
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, random_state=0
        )
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X_train, y_train, w_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    plt_ax = plot_reliability_diagram(
        y_obs=y_test,
        y_pred=y_pred,
        weights=w_test,
        ax=ax,
        n_bootstrap=n_bootstrap,
        diagram_type=diagram_type,
    )

    if ax is not None:
        assert ax is plt_ax
    assert plt_ax.get_xlabel() == "prediction for E(Y|X)"
    if diagram_type == "reliability":
        assert plt_ax.get_ylabel() == "estimated E(Y|prediction)"
        assert plt_ax.get_title() == "Reliability Diagram"
    else:
        assert plt_ax.get_ylabel() == "bias = prediction - estimated E(Y|prediction)"
        assert plt_ax.get_title() == "Bias Reliability Diagram"

    plt_ax = plot_reliability_diagram(
        y_obs=y_test,
        y_pred=pd.Series(y_pred, name="simple"),
        weights=w_test,
        ax=ax,
        n_bootstrap=n_bootstrap,
        diagram_type=diagram_type,
    )
    if diagram_type == "reliability":
        assert plt_ax.get_title() == "Reliability Diagram simple"
    else:
        assert plt_ax.get_title() == "Bias Reliability Diagram simple"


def test_plot_reliability_diagram_multiple_predictions():
    """Test that plot_reliability_diagram works for multiple predictions."""
    n_obs = 10
    y_obs = np.arange(n_obs)
    y_obs[::2] = 0
    y_pred = pd.DataFrame({"model_1": np.ones(n_obs), "model_2": 3 * np.ones(n_obs)})
    plt_ax = plot_reliability_diagram(
        y_obs=y_obs,
        y_pred=y_pred,
    )
    assert plt_ax.get_title() == "Reliability Diagram"
    legend_text = plt_ax.get_legend().get_texts()
    assert len(legend_text) == 2
    assert legend_text[0].get_text() == "model_1"
    assert legend_text[1].get_text() == "model_2"


@pytest.mark.parametrize(
    ("list2array", "multidim"),
    [(lambda x: x, True), (np.asarray, True), (pd.Series, False), (pl.Series, False)],
)
def test_plot_reliability_diagram_array_like(list2array, multidim):
    """Test that plot_reliability_diagram raises errors."""
    y_obs = list2array([0, 1, 2])
    y_pred = list2array([-1, 1, 0])
    plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred)

    if multidim:
        y_pred = list2array([[-1, 1, 0], [1, 2, 3], [-3, -2, -1]])
        plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred)


@pytest.mark.parametrize("ax", [None, plt.subplots()[1]])
@pytest.mark.parametrize("feature_type", ["cat", "num", "string"])
def test_plot_bias(ax, feature_type):
    """Test that plot_bias works."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X_train, y_train)
    feature = X_test[:, 0]
    if feature_type == "cat":
        feature = pd.Series(feature.astype("=U8"), dtype="category")
    elif feature_type == "string":
        feature = feature.astype("=U8")

    plt_ax = plot_bias(
        y_obs=y_test,
        y_pred=clf.predict_proba(X_test)[:, 1],
        feature=feature,
        ax=ax,
    )

    if ax is not None:
        assert ax is plt_ax
    if feature_type == "num":
        assert plt_ax.get_xlabel() == "binned feature"
    else:
        assert plt_ax.get_xlabel() == "feature"
    assert plt_ax.get_ylabel() == "bias"
    assert plt_ax.get_title() == "Bias Plot"

    plt_ax = plot_bias(
        y_obs=y_test,
        y_pred=pd.Series(clf.predict_proba(X_test)[:, 1], name="simple"),
        feature=feature,
        ax=ax,
    )
    assert plt_ax.get_title() == "Bias Plot simple"


def test_plot_bias_feature_none():
    """Test that plot_bias works."""
    y_obs = np.arange(10)
    y_pred = pd.DataFrame(
        {
            "model_1": np.arange(10) + 0.5,
            "model_2": (y_obs - 5) ** 2,
            "model_3": (y_obs - 3) ** 2,
        }
    )
    fig, ax = plt.subplots()
    ax = plot_bias(y_obs=y_obs, y_pred=y_pred, feature=None, ax=ax)
    assert ax.get_title() == "Bias Plot"
    assert ax.get_legend() is None
    assert ax.get_xlabel() == "model"
    assert [x.get_text() for x in ax.get_xmajorticklabels()] == [
        "model_1",
        "model_2",
        "model_3",
    ]


def test_plot_bias_multiple_predictions():
    """Test that plot_bias works for multiple predictions."""
    y_obs = np.arange(10)
    y_pred = pd.DataFrame(
        {
            "model_1": np.arange(10) + 0.5,
            "model_2": (y_obs - 5) ** 2,
            "model_3": (y_obs - 3) ** 2,
        }
    )
    feature = np.ones(10)
    feature[::2] = 0
    fig, ax = plt.subplots()
    ax = plot_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        feature=feature,
    )
    assert ax.get_title() == "Bias Plot"
    legend_text = ax.get_legend().get_texts()
    assert len(legend_text) == 3
    assert legend_text[0].get_text() == "model_1"
    assert legend_text[1].get_text() == "model_2"
    assert legend_text[2].get_text() == "model_3"
