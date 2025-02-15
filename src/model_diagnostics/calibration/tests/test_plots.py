from math import ceil

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from model_diagnostics import config_context
from model_diagnostics._utils.plot_helper import (
    get_legend_list,
    get_title,
    get_xlabel,
    get_ylabel,
)
from model_diagnostics._utils.test_helper import (
    SkipContainer,
    pa_array,
    pa_table,
    pd_available,
    pd_Series,
)
from model_diagnostics.calibration import (
    add_marginal_subplot,
    plot_bias,
    plot_marginal,
    plot_reliability_diagram,
)


@pytest.mark.parametrize(
    ("param", "value", "msg"),
    [
        ("diagram_type", "XXX", "Parameter diagram_type must be either.*XXX"),
        ("functional", "XXX", "Argument functional must be one of.*XXX"),
        ("level", 2, "Argument level must fulfil 0 < level < 1, got 2"),
        (
            "ax",
            "XXX",
            "The ax argument must be None, a matplotlib Axes or a plotly Figure",
        ),
    ],
)
def test_plot_reliability_diagram_raises(param, value, msg):
    """Test that plot_reliability_diagram raises errors."""
    y_obs = [0, 1]
    y_pred = [-1, 1]
    d = {param: value}
    if "functional" not in d:
        d["functional"] = "quantile"  # as a default
    with pytest.raises(ValueError, match=msg):
        plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred, **d)


def test_plot_reliability_diagram_raises_y_obs_multdim():
    """Test that plot_reliability_diagram raises errors for y_obs.ndim > 1."""
    y_obs = [[0], [1]]
    y_pred = [-1, 1]
    plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred)
    y_obs = [[0, 1], [1, 2]]
    with pytest.raises(ValueError, match="Array-like y_obs has more than 2 dimensions"):
        plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred)


@pytest.mark.parametrize("diagram_type", ["reliability", "bias"])
@pytest.mark.parametrize("functional", ["mean", "expectile", "quantile"])
@pytest.mark.parametrize("n_bootstrap", [None, 10])
@pytest.mark.parametrize("weights", [None, True])
@pytest.mark.parametrize("ax", [None, plt.subplots()[1], "plotly"])
@pytest.mark.parametrize("plot_backend", ["matplotlib", "plotly"])
def test_plot_reliability_diagram(
    diagram_type, functional, n_bootstrap, weights, ax, plot_backend
):
    """Test that plot_reliability_diagram works."""
    if plot_backend == "plotly" or ax == "plotly":
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

    if ax == "plotly":
        ax = go.Figure()

    X, y = make_classification(random_state=42, n_classes=2)
    if weights is None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
        w_train, w_test = None, None
    elif functional == "quantile":
        pytest.skip("Weighted quantiles are not implemented.")
    else:
        weights = np.random.default_rng(42).integers(low=1, high=10, size=y.shape)
        X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
            X, y, weights, random_state=0
        )
    clf = LogisticRegression(solver="newton-cholesky")
    clf.fit(X_train, y_train, w_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    with config_context(plot_backend=plot_backend):
        plt_ax = plot_reliability_diagram(
            y_obs=y_test,
            y_pred=y_pred,
            weights=w_test,
            functional=functional,
            level=0.8,
            n_bootstrap=n_bootstrap,
            diagram_type=diagram_type,
            ax=ax,
        )

    xlabel_mapping = {
        "mean": "E(Y|X)",
        "median": "median(Y|X)",
        "expectile": "0.8-expectile(Y|X)",
        "quantile": "0.8-quantile(Y|X)",
    }
    ylabel_mapping = {
        "mean": "E(Y|prediction)",
        "median": "median(Y|prediction)",
        "expectile": "0.8-expectile(Y|prediction)",
        "quantile": "0.8-quantile(Y|prediction)",
    }

    if ax is not None:
        assert ax is plt_ax

    assert get_xlabel(plt_ax) == "prediction for " + xlabel_mapping[functional]

    if diagram_type == "reliability":
        assert get_ylabel(plt_ax) == "estimated " + ylabel_mapping[functional]
        assert get_title(plt_ax) == "Reliability Diagram"
    else:
        assert (
            get_ylabel(plt_ax) == "prediction - estimated " + ylabel_mapping[functional]
        )
        assert get_title(plt_ax) == "Bias Reliability Diagram"

    with config_context(plot_backend=plot_backend):
        plt_ax = plot_reliability_diagram(
            y_obs=y_test,
            y_pred=pl.Series(values=y_pred, name="simple"),
            weights=w_test,
            ax=ax,
            n_bootstrap=n_bootstrap,
            diagram_type=diagram_type,
        )
    if diagram_type == "reliability":
        assert get_title(plt_ax) == "Reliability Diagram simple"
    else:
        assert get_title(plt_ax) == "Bias Reliability Diagram simple"


@pytest.mark.parametrize("plot_backend", ["matplotlib", "plotly"])
def test_plot_reliability_diagram_multiple_predictions(plot_backend):
    """Test that plot_reliability_diagram works for multiple predictions."""
    if plot_backend == "plotly":
        pytest.importorskip("plotly")

    n_obs = 10
    y_obs = np.arange(n_obs)
    y_obs[::2] = 0
    y_pred = pl.DataFrame({"model_2": np.ones(n_obs), "model_1": 3 * np.ones(n_obs)})
    with config_context(plot_backend=plot_backend):
        plt_ax = plot_reliability_diagram(
            y_obs=y_obs,
            y_pred=y_pred,
        )
    assert get_title(plt_ax) == "Reliability Diagram"
    legend_text = get_legend_list(plt_ax)
    assert len(legend_text) == 2
    assert legend_text[0] == "model_2"
    assert legend_text[1] == "model_1"


@pytest.mark.parametrize(
    "list2array",
    [lambda x: x, np.asarray, pa_array, pd_Series, pl.Series],
)
def test_plot_reliability_diagram_1d_array_like(list2array):
    """Test that plot_reliability_diagram workds for 1d array-likes."""
    y_obs = list2array([0, 1, 2])
    y_pred = list2array([-1, 1, 0])
    if isinstance(y_pred, SkipContainer):
        pytest.skip("Module for data container not imported.")
    plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred)


@pytest.mark.parametrize(
    "list2array",
    [
        lambda x: x,
        np.asarray,
        lambda x: pa_table(x, names=["0", "1", "2"]),
        lambda x: pl.DataFrame(x, schema=["0", "1", "2"], orient="row"),
    ],
)
def test_plot_reliability_diagram_2d_array_like(list2array):
    """Test that plot_reliability_diagram workds for 2d array-likes."""
    y_obs = [0, 1, 2]
    y_pred = list2array([[-1, 1, 0], [1, 2, 3], [-3, -2, -1]])
    if isinstance(y_pred, SkipContainer):
        pytest.skip("Module for data container not imported.")
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
    y_obs = pl.Series(values=y_obs, name="y")
    y_pred = pl.Series(values=y_pred, name="z")

    plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred, n_bootstrap=10)

    if pd_available:
        import sklearn

        with sklearn.config_context(transform_output="pandas"):
            # Without our internal code setting set_output(transform="default"),
            # this test will error.
            plot_reliability_diagram(y_obs=y_obs, y_pred=y_pred, n_bootstrap=10)


@pytest.mark.parametrize(
    ("param", "value", "msg"),
    [
        (
            "confidence_level",
            1,
            "Argument confidence_level must fulfil 0 <= level < 1, got 1",
        ),
        (
            "ax",
            "XXX",
            "The ax argument must be None, a matplotlib Axes or a plotly Figure",
        ),
        (
            "bin_method",
            "XXX",
            "Parameter bin_method must be one of .*quantile",
        ),
    ],
)
def test_plot_bias_raises(param, value, msg):
    """Test that plot_bias raises errors."""
    y_obs = [0, 1, 2]
    y_pred = [-1, 1, 1]
    feature = ["a", "a", "b"]
    d = {param: value}
    with pytest.raises(ValueError, match=msg):
        plot_bias(y_obs=y_obs, y_pred=y_pred, feature=feature, **d)


def test_plot_bias_warning_for_null_stderr():
    """Test that plot_bias gives warning when some stderr are Null."""
    y_obs = np.arange(3).astype(float)
    y_obs[0] = np.nan
    y_pred = y_obs + 1
    feature = ["a", "a", "b"]

    with pytest.warns(
        UserWarning,
        match="Some values of 'bias_stderr' are null. Therefore no error bars are",
    ):
        plot_bias(
            y_obs=y_obs,
            y_pred=y_pred,
            feature=feature,
            confidence_level=0.95,
        )


@pytest.mark.parametrize("with_null_values", [False, True])
@pytest.mark.parametrize(
    "feature_type", ["cat", "cat_pandas", "cat_physical", "enum", "num", "string"]
)
@pytest.mark.parametrize("bin_method", ["quantile", "uniform"])
@pytest.mark.parametrize("confidence_level", [0, 0.95])
@pytest.mark.parametrize("ax", [None, plt.subplots()[1], "plotly"])
@pytest.mark.parametrize("plot_backend", ["matplotlib", "plotly"])
def test_plot_bias(
    with_null_values, feature_type, bin_method, confidence_level, ax, plot_backend
):
    """Test that plot_bias works."""
    if plot_backend == "plotly" or ax == "plotly":
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

    if ax == "plotly":
        ax = go.Figure()

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X_train, y_train)
    feature = X_test[:, 0]
    if feature_type != "num":
        bins = np.quantile(feature, [0.2, 0.5, 0.8])
        feature = np.digitize(feature, bins=bins).astype("=U8")
        if feature_type == "cat_pandas":
            # Note that feature is already converted to string as polars does not like
            # pandas.categorical with non-string values.
            feature = pd_Series(feature, dtype="category")
            dtype = pl.Categorical
        elif feature_type == "cat":
            dtype = pl.Categorical
        elif feature_type == "cat_physical":
            dtype = pl.Categorical(ordering="physical")
        elif feature_type == "enum":
            dtype = pl.Enum(categories=np.unique(feature))
        else:
            dtype = pl.Utf8

        if feature_type in ["cat", "cat_physical", "enum"]:
            feature = pl.Series(feature, dtype=dtype)

    if isinstance(feature, SkipContainer):
        pytest.skip("Module for data container not imported.")

    with pl.StringCache():
        if with_null_values:
            if feature_type == "cat_pandas":
                feature[0] = None
            elif feature_type in ["cat", "cat_physical", "enum"]:
                feature = pl.Series(feature).cast(str).scatter(0, None).cast(dtype)
            else:
                feature = pl.Series(feature).scatter(0, None)

        with config_context(plot_backend=plot_backend):
            plt_ax = plot_bias(
                y_obs=y_test,
                y_pred=clf.predict_proba(X_test)[:, 1],
                feature=feature,
                bin_method=bin_method,
                confidence_level=confidence_level,
                ax=ax,
            )

        if ax is not None:
            assert ax is plt_ax

        if feature_type == "num":
            assert get_xlabel(plt_ax) == "binned feature"
        else:
            assert get_xlabel(plt_ax) == "feature"

        assert get_ylabel(plt_ax) == "bias"
        assert get_title(plt_ax) == "Bias Plot"

        if (
            isinstance(ax, mpl.axes.Axes)
            and with_null_values
            and feature_type
            in [
                "cat",
                "cat_pandas",
                "cat_physical",
                "enum",
                "string",
            ]
        ):
            xtick_labels = plt_ax.xaxis.get_ticklabels()
            assert xtick_labels[-1].get_text() == "Null"

        with config_context(plot_backend=plot_backend):
            plt_ax = plot_bias(
                y_obs=y_test,
                y_pred=pl.Series(values=clf.predict_proba(X_test)[:, 1], name="simple"),
                feature=feature,
                confidence_level=confidence_level,
                ax=ax,
            )
        assert get_title(plt_ax) == "Bias Plot simple"


def test_plot_bias_feature_none():
    """Test that plot_bias works for feature=None."""
    y_obs = np.arange(10)
    y_pred = pl.DataFrame(
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


@pytest.mark.parametrize("with_null", [False, True])
@pytest.mark.parametrize("feature_type", ["num", "string"])
@pytest.mark.parametrize("confidence_level", [0, 0.95])
def test_plot_bias_multiple_predictions(with_null, feature_type, confidence_level):
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
    rng = np.random.default_rng(42)
    feature = rng.integers(low=0, high=n_obs // 2, size=n_obs)
    if feature_type == "string":
        feature = feature.astype(str)

    if with_null:
        feature = pl.Series(feature).scatter([0, 5], None)

    fig, ax = plt.subplots()
    ax = plot_bias(
        y_obs=y_obs,
        y_pred=y_pred,
        feature=feature,
        confidence_level=confidence_level,
    )
    assert ax.get_title() == "Bias Plot"
    legend_text = ax.get_legend().get_texts()
    assert len(legend_text) == 3 + with_null
    assert legend_text[0].get_text() == "model_1"
    assert legend_text[1].get_text() == "model_3"
    assert legend_text[2].get_text() == "model_2"
    if with_null:
        assert legend_text[3].get_text() == "Null values"


@pytest.mark.parametrize(
    ("param", "value", "msg"),
    [
        (
            "ax",
            "XXX",
            "The ax argument must be None, a matplotlib Axes or a plotly Figure",
        ),
        (
            "bin_method",
            "XXX",
            "Parameter bin_method must be one of .*quantile",
        ),
        ("show_lines", 2, "The argument show_lines mut be 'always' or 'numerical'"),
    ],
)
def test_plot_marginal_raises(param, value, msg):
    """Test that plot_marginal raises errors."""
    y_obs = [0, 1, 2]
    d = {param: value}
    with pytest.raises(ValueError, match=msg):
        plot_marginal(
            y_obs=y_obs,
            y_pred=y_obs,
            X=np.ones_like(y_obs)[:, None],
            feature_name=0,
            **d,
        )


def test_plot_marginal_raises_more_than_one_model():
    msg = (
        r"Parameter y_pred has shape \(n_obs, 3\), but only \(n_obs\) and \(n_obs, 1\)"
    )
    with pytest.raises(ValueError, match=msg):
        plot_marginal(
            y_obs=[1, 2],
            y_pred=[[1, 2, 3], [10, 20, 30]],
            X=np.ones(2)[:, None],
            feature_name=0,
        )


@pytest.mark.parametrize("with_null_values", [False, True])
@pytest.mark.parametrize(
    "feature_type", ["cat", "cat_pandas", "cat_physical", "enum", "num", "string"]
)
@pytest.mark.parametrize("bin_method", ["quantile", "uniform"])
@pytest.mark.parametrize("ax", [None, plt.subplots()[1], "plotly"])
@pytest.mark.parametrize("plot_backend", ["matplotlib", "plotly"])
def test_plot_marginal(with_null_values, feature_type, bin_method, ax, plot_backend):
    """Test that plot_marginal works."""
    if plot_backend == "plotly" or ax == "plotly":
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

    if ax == "plotly":
        ax = go.Figure()

    X, y = make_classification(
        n_samples=100,
        n_features=10,
        random_state=42,
    )
    feature = X[:, 0]
    if feature_type != "num":
        bins = np.quantile(feature, [0.2, 0.5, 0.8])
        feature = np.digitize(feature, bins=bins).astype("=U8")
        if feature_type == "cat_pandas":
            # Note that feature is already converted to string as polars does not like
            # pandas.categorical with non-string values.
            feature = pd_Series(feature, dtype="category")
            dtype = pl.Categorical
        elif feature_type == "cat":
            dtype = pl.Categorical
        elif feature_type == "cat_physical":
            dtype = pl.Categorical(ordering="physical")
        elif feature_type == "enum":
            dtype = pl.Enum(categories=np.unique(feature))
        else:
            dtype = pl.Utf8

        if feature_type in ["cat", "cat_physical", "enum"]:
            feature = pl.Series(feature, dtype=dtype)

    if isinstance(feature, SkipContainer):
        pytest.skip("Module for data container not imported.")

    if with_null_values:
        with pl.StringCache():
            if feature_type == "cat_pandas":
                feature[0] = None
            elif feature_type in ["cat", "cat_physical", "enum"]:
                feature = pl.Series(feature).cast(str).scatter(0, None).cast(dtype)
            else:
                feature = pl.Series(feature).scatter(0, None)

    if feature_type in ["cat_pandas"]:
        import pandas as pd

        X = pd.DataFrame(X[:, 1:], columns=[str(i) for i in range(1, X.shape[1])])
        X.insert(0, "feature_0", feature)
    else:
        X = pl.DataFrame(X[:, 1:])
        X = X.insert_column(0, pl.Series(name="feature_0", values=feature))

    class TransformToString(TransformerMixin, BaseEstimator):
        def fit(self, X, y=None):
            self.__sklearn_is_fitted__ = True
            return self

        def transform(self, X, y=None):
            if isinstance(X, pl.DataFrame):
                return X.with_columns(pl.all().cast(pl.Utf8))
            elif hasattr(X, "iloc"):
                return X.astype(str)
            else:
                return X

    safe_ohe = make_pipeline(
        TransformToString(), OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    )

    clf = make_pipeline(
        ColumnTransformer([("ohe", safe_ohe, [0])], remainder="passthrough"),
        LogisticRegression(solver="lbfgs"),
    )
    clf.fit(X, y)

    with config_context(plot_backend=plot_backend):
        plt_ax = plot_marginal(
            y_obs=y,
            y_pred=pl.Series(name="modelX", values=clf.predict_proba(X)[:, 1]),
            X=X,
            feature_name="feature_0",
            predict_function=lambda x: clf.predict_proba(x)[:, 1],
            bin_method=bin_method,
            ax=ax,
        )

    if ax is not None:
        assert ax is plt_ax

    if feature_type == "num":
        assert get_xlabel(plt_ax) == "binned feature_0"
    else:
        assert get_xlabel(plt_ax) == "feature_0"

    assert get_ylabel(plt_ax, yaxis=2) == "y"
    assert get_title(plt_ax) == "Marginal Plot modelX"

    if (
        isinstance(ax, mpl.axes.Axes)
        and with_null_values
        and feature_type
        in [
            "cat",
            "cat_pandas",
            "cat_physical",
            "enum",
            "string",
        ]
    ):
        xtick_labels = plt_ax.xaxis.get_ticklabels()
        assert xtick_labels[-1].get_text() == "Null"

    legend_text = get_legend_list(plt_ax)
    # TODO: It is not 100% clear why legend_text has most often more entries than 3 or
    # 4. We therefor test >= instead of ==.
    # It is also unclear why for matplotlib the order varies.
    if with_null_values:
        assert len(legend_text) >= 4
    else:
        assert len(legend_text) >= 3
    if plot_backend == "matplotlib":
        assert "mean y_obs" in legend_text
        assert "mean y_pred" in legend_text
        assert "partial dependence" in legend_text
        if with_null_values:
            assert "Null values" in legend_text
    else:
        assert legend_text[0] == "mean y_obs"
        assert legend_text[1] == "mean y_pred"
        assert legend_text[2] == "partial dependence"
        if with_null_values:
            assert legend_text[-1] == "Null values"


@pytest.mark.parametrize("show_lines", ["always", "numerical"])
@pytest.mark.parametrize("feature_type", ["num", "string"])
@pytest.mark.parametrize("plot_backend", ["matplotlib", "plotly"])
def test_plot_marginal_show_lines(show_lines, feature_type, plot_backend):
    """Test that plot_marginal works with different show_lines settings."""
    if plot_backend == "plotly":
        pytest.importorskip("plotly")

    if feature_type == "num":
        x = np.arange(4)
    elif feature_type == "string":
        x = np.array(list("abcd"))
    with config_context(plot_backend=plot_backend):
        ax = plot_marginal(
            y_obs=np.arange(4),
            y_pred=np.arange(4),
            X=x[:, None],
            feature_name=0,
            predict_function=lambda x: np.arange(len(x)),
            show_lines=show_lines,
        )

    if plot_backend == "matplotlib":
        from matplotlib.lines import Line2D

        # Filter 3 Line2d children for the 3 lines for mean y_obs, mean y_pred, pd
        lines = [x for x in ax.get_children() if isinstance(x, Line2D)]
        assert len(lines) == 3
        if show_lines == "always" or feature_type == "num":
            assert lines[0].get_linestyle() == "-"
            assert lines[1].get_linestyle() == "-"
            assert lines[2].get_linestyle() == "--"
        else:
            for i in range(3):
                assert lines[i].get_linestyle() == "None"
    else:
        from plotly.graph_objs import Scatter

        mode = (
            "lines+markers"
            if show_lines == "always" or feature_type == "num"
            else "markers"
        )
        # Data elements 1, 2, 3 are the lines for mean y_obs, mean y_pred, pd
        for i in range(1, 1 + 3):
            assert isinstance(ax.data[i], Scatter)
            assert ax.data[i].mode == mode


def test_add_marginal_subplot_raises():
    """Test that add_marginal_subplot raises errors."""
    pytest.importorskip("plotly")
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    n_rows, n_cols = 4, 3
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
    )
    msg = f"The `fig` only has {n_rows} rows and {n_cols} columns"
    with pytest.raises(ValueError, match=msg):
        add_marginal_subplot(go.Figure(), fig, 5, 4)


def test_add_marginal_subplot():
    """Test that add_marginal_subplot works."""
    pytest.importorskip("plotly")
    from plotly.subplots import make_subplots

    n_features = 12
    n_obs = 10
    y_obs = np.arange(n_obs)
    X = np.ones((n_obs, n_features))
    X[:n_obs, 0] = np.sin(np.arange(n_obs))
    X[:, 1] = y_obs**2

    def model_predict(X):
        s = 0.5 * n_obs * np.sin(X)
        return s.sum(axis=1) + np.sqrt(X[:, 1])

    # Now the plotting.
    feature_list = list(range(n_features))
    n_cols = 3
    n_rows = ceil(len(feature_list) / n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=[[{"secondary_y": True}] * n_cols] * n_rows,
    )
    for row in range(n_rows):
        for col in range(n_cols):
            i = n_cols * row + col
            with config_context(plot_backend="plotly"):
                subfig = plot_marginal(
                    y_obs=y_obs,
                    y_pred=model_predict(X),
                    X=X,
                    feature_name=feature_list[i],
                    predict_function=model_predict,
                )
            add_marginal_subplot(subfig, fig, row, col)

    assert get_xlabel(fig, xaxis=1) == "binned feature 0"
    assert get_xlabel(fig, xaxis=3) == "binned feature 2"
    assert get_xlabel(fig, xaxis=4) == "binned feature 3"
    assert get_ylabel(fig, yaxis=2) == "y"
    assert get_title(fig) == "Marginal Plot"

    legend_text = get_legend_list(fig)
    # TODO: It is not 100% clear why legend_text has most often more entries than 3 or
    # 4. We therefor test >= instead of ==.
    # It is also unclear why for matplotlib the order varies.
    assert len(legend_text) >= 3
    assert legend_text[0] == "mean y_obs"
    assert legend_text[1] == "mean y_pred"
    assert legend_text[2] == "partial dependence"
