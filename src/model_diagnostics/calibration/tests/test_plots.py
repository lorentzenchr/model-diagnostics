from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from model_diagnostics.calibration import plot_reliability_diagram


def test_plot_reliability_diagram():
    """Test that plot_reliability_diagram works."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    ax = plot_reliability_diagram(y_test, y_pred)

    assert ax.get_xlabel() == "prediction for E(Y|X)"
    assert ax.get_ylabel() == "estimated E(Y|prediction)"
    assert ax.get_title() == "Reliability Diagram"
