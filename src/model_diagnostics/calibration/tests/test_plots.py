import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest
from packaging.version import Version
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model_diagnostics import config_context, polars_version
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
from model_diagnostics.calibration import plot_bias, plot_reliability_diagram


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
    if "functional" not in d.keys():
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
@pytest.mark.parametrize("confidence_level", [0, 0.95])
@pytest.mark.parametrize("ax", [None, plt.subplots()[1], "plotly"])
@pytest.mark.parametrize("plot_backend", ["matplotlib", "plotly"])
def test_plot_bias(with_null_values, feature_type, confidence_level, ax, plot_backend):
    """Test that plot_bias works."""
    if plot_backend == "plotly" or ax == "plotly":
        pytest.importorskip("plotly")
        import plotly.graph_objects as go

    if ax == "plotly":
        ax = go.Figure()

    if feature_type in ["cat_physical", "enum"] and polars_version < Version("0.20.0"):
        pytest.skip("Test needs polars >= 0.20.0")
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
                # FIXME: polars >= 0.19.14
                if polars_version >= Version("0.19.14"):
                    feature = pl.Series(feature).cast(str).scatter(0, None).cast(dtype)
                else:
                    feature = (
                        pl.Series(feature).cast(str).set_at_idx(0, None).cast(dtype)
                    )
            else:
                # FIXME: polars >= 0.19.14
                if polars_version >= Version("0.19.14"):  # noqa: PLR5501
                    feature = pl.Series(feature).scatter(0, None)
                else:
                    feature = pl.Series(feature).set_at_idx(0, None)

        with config_context(plot_backend=plot_backend):
            plt_ax = plot_bias(
                y_obs=y_test,
                y_pred=clf.predict_proba(X_test)[:, 1],
                feature=feature,
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
        # FIXME: polars >= 0.19.14
        if polars_version >= Version("0.19.14"):
            feature = pl.Series(feature).scatter([0, 5], None)
        else:
            feature = pl.Series(feature).set_at_idx([0, 5], None)

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
