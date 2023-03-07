import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
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


def test_plot_reliability_diagram_raises_y_obs_multdim():
    """Test that plot_reliability_diagram raises errors for y_obs.ndim > 1."""
    y_obs = [[0], [1]]
    y_pred = [-1, 1]
    plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred)
    y_obs = [[0, 1], [1, 2]]
    with pytest.raises(ValueError, match="Array-like y_obs has more than 2 dimensions"):
        plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred)


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
        assert plt_ax.get_ylabel() == "prediction - estimated E(Y|prediction)"
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
    y_pred = pd.DataFrame({"model_2": np.ones(n_obs), "model_1": 3 * np.ones(n_obs)})
    plt_ax = plot_reliability_diagram(
        y_obs=y_obs,
        y_pred=y_pred,
    )
    assert plt_ax.get_title() == "Reliability Diagram"
    legend_text = plt_ax.get_legend().get_texts()
    assert len(legend_text) == 2
    assert legend_text[0].get_text() == "model_2"
    assert legend_text[1].get_text() == "model_1"


@pytest.mark.parametrize(
    "list2array",
    [lambda x: x, np.asarray, pa.array, pd.Series, pl.Series],
)
def test_plot_reliability_diagram_1d_array_like(list2array):
    """Test that plot_reliability_diagram workds for 1d array-likes."""
    y_obs = list2array([0, 1, 2])
    y_pred = list2array([-1, 1, 0])
    plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred)


@pytest.mark.parametrize(
    "list2array",
    [
        lambda x: x,
        np.asarray,
        lambda x: pa.table(x, names=["0", "1", "2"]),
        lambda x: pl.DataFrame(x, schema=["0", "1", "2"], orient="row"),
    ],
)
def test_plot_reliability_diagram_2d_array_like(list2array):
    """Test that plot_reliability_diagram workds for 2d array-likes."""
    y_obs = [0, 1, 2]
    y_pred = list2array([[-1, 1, 0], [1, 2, 3], [-3, -2, -1]])
    plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred)


def test_plot_reliability_diagram_constant_prediction_transform_output():
    """Test that plot_reliability_diagram works for a constant prediction.

    This is tested with and without a scikit-learn context manager which sets
    transform_output="pandas".
    """
    n_obs = 10
    np.random.default_rng(42)
    y_obs = np.arange(n_obs)
    y_pred = np.full_like(y_obs, 14.1516)  # a constant prediction
    y_obs = pd.Series(y_obs, name="y")
    y_pred = pd.Series(y_pred, name="z")

    plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred, n_bootstrap=10)

    import sklearn

    with sklearn.config_context(transform_output="pandas"):
        # Without our internal code setting set_output(transform="default"),
        # this test will error.
        plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred, n_bootstrap=10)


@pytest.mark.parametrize("feature_type", ["cat", "num", "string"])
@pytest.mark.parametrize("with_errorbars", [False, True])
@pytest.mark.parametrize("ax", [None, plt.subplots()[1]])
def test_plot_bias(feature_type, with_errorbars, ax):
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
        # We first convert to string as polars does not like pandas.categorical with
        # non-string values.
        bins = np.quantile(feature, [0.2, 0.5, 0.8])
        feature = pd.Series(
            np.digitize(feature, bins=bins).astype("=U8"), dtype="category"
        )
    elif feature_type == "string":
        bins = np.quantile(feature, [0.2, 0.5, 0.8])
        feature = np.digitize(feature, bins=bins).astype("=U8")

    plt_ax = plot_bias(
        y_obs=y_test,
        y_pred=clf.predict_proba(X_test)[:, 1],
        feature=feature,
        with_errorbars=with_errorbars,
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
        with_errorbars=with_errorbars,
        ax=ax,
    )
    assert plt_ax.get_title() == "Bias Plot simple"


def test_plot_bias_feature_none():
    """Test that plot_bias works."""
    y_obs = np.arange(10)
    y_pred = pd.DataFrame(
        {
            "model_1": np.arange(10) + 0.5,
            "model_3": (y_obs - 5) ** 2,
            "model_2": (y_obs - 3) ** 2,
        }
    )
    fig, ax = plt.subplots()
    ax = plot_bias(y_obs=y_obs, y_pred=y_pred, feature=None, ax=ax)
    assert ax.get_title() == "Bias Plot"
    assert ax.get_legend() is None
    assert ax.get_xlabel() == "model"
    assert [x.get_text() for x in ax.get_xmajorticklabels()] == [
        "model_1",
        "model_3",
        "model_2",
    ]


@pytest.mark.parametrize("feature_type", ["num", "string"])
def test_plot_bias_multiple_predictions(feature_type):
    """Test that plot_bias works for multiple predictions.

    This also tests feature to be a string with many different values
    """
    n_obs = 100
    y_obs = np.arange(n_obs)
    y_pred = pl.DataFrame(
        {
            "model_1": np.arange(n_obs) + 0.5,
            "model_3": (y_obs - 5) ** 2,
            "model_2": (y_obs - 3) ** 2,
        }
    )
    # string
    rng = np.random.default_rng(42)
    feature = rng.integers(low=0, high=n_obs // 2, size=n_obs)
    if feature_type == "string":
        feature = feature.astype(str)

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
    assert legend_text[1].get_text() == "model_3"
    assert legend_text[2].get_text() == "model_2"
