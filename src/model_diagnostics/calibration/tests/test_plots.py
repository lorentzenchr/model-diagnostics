import matplotlib.pyplot as plt
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model_diagnostics.calibration import plot_bias, plot_reliability_diagram


@pytest.mark.parametrize("ax", [None, plt.subplots()[1]])
def test_plot_reliability_diagram(ax):
    """Test that plot_reliability_diagram works."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    plt_ax = plot_reliability_diagram(y_obs=y_test, y_pred=y_pred, ax=ax)

    if ax is not None:
        assert ax is plt_ax
    assert plt_ax.get_xlabel() == "prediction for E(Y|X)"
    assert plt_ax.get_ylabel() == "estimated E(Y|prediction)"
    assert plt_ax.get_title() == "Reliability Diagram"

    plt_ax = plot_reliability_diagram(
        y_obs=y_test,
        y_pred=pd.Series(y_pred, name="simple"),
        ax=ax,
    )
    assert plt_ax.get_title() == "Reliability Diagram simple"


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
