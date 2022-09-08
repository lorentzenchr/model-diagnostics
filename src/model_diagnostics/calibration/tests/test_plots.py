from model_diagnostics.calibration import plot_reliability_diagram
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def test_plot_reliability_diagram():
    """Test that plot_reliability_diagram works."""
    X, y = make_classification(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LogisticRegression(solver="lbfgs")
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:, 1]
    plot_reliability_diagram(y_test, y_pred)
