import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression as skl_IsotonicRegression

from model_diagnostics._utils.isotonic import (
    IsotonicRegression,
    gpava,
    isotonic_regression,
    pava,
)
from model_diagnostics.scoring import PinballLoss, SquaredError


def mean_fun(x, w):
    return np.average(x, weights=w)


mean_fun.loss = SquaredError()  # type: ignore


def median_lower(x, w=None):
    return np.quantile(x, q=0.5, method="inverted_cdf")


median_lower.loss = PinballLoss(level=0.5)  # type: ignore


def median_upper(x, w=None):
    return -np.quantile(-np.asarray(x), q=0.5, method="inverted_cdf")


median_upper.loss = PinballLoss(level=0.5)  # type: ignore


def quantile80_lower(x, w=None):
    return np.quantile(x, q=0.8, method="inverted_cdf")


quantile80_lower.loss = PinballLoss(level=0.8)  # type: ignore


def quantile80_upper(x, w=None):
    return -np.quantile(-np.asarray(x), q=1 - 0.8, method="inverted_cdf")


quantile80_upper.loss = PinballLoss(level=0.8)  # type: ignore


def quantile10_lower(x, w=None):
    return np.quantile(x, q=0.1, method="inverted_cdf")


quantile10_lower.loss = PinballLoss(level=0.1)  # type: ignore


def quantile10_upper(x, w=None):
    return -np.quantile(-np.asarray(x), q=1 - 0.1, method="inverted_cdf")


quantile10_upper.loss = PinballLoss(level=0.1)  # type: ignore


def gpava_mean(x, w=None):
    return gpava(mean_fun, x, w)


@pytest.mark.parametrize("pava", [pava, gpava_mean])
def test_pava_simple(pava):
    # Test case of Busing 2020
    # https://doi.org/10.18637/jss.v102.c01
    y = [8, 4, 8, 2, 2, 0, 8]
    w = np.ones_like(y)
    x, r = pava(y, w)
    assert_array_equal(x, [4, 4, 4, 4, 4, 4, 8])
    assert_array_equal(r, [0, 6, 7])

    y = [9, 1, 8, 2, 7, 3, 6]
    w = None
    x, r = pava(y, w)
    assert_array_equal(x, [5, 5, 5, 5, 5, 5, 6])
    assert_array_equal(r, [0, 6, 7])

    y = [9, 1, 8, 4, -2, 8, 6]
    x, r = pava(y, w)
    assert_array_equal(x, [4, 4, 4, 4, 4, 7, 7])
    assert_array_equal(r, [0, 5, 7])


@pytest.mark.parametrize("pava", [pava, gpava_mean])
def test_pava_weighted(pava):
    y = [8, 4, 8, 2, 2, 0, 8]
    w = [1, 2, 3, 4, 5, 5, 4]
    x, r = pava(y, w)
    assert_array_equal(x, [2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 8])
    assert_array_equal(r, [0, 6, 7])


@pytest.mark.parametrize(
    ("y", "fun_lower", "x_lower", "fun_upper", "x_upper"),
    [
        (
            [8, 4, 8, 2, 2, 0, 8],
            median_lower,
            [2, 2, 2, 2, 2, 2, 8],
            median_upper,
            [4, 4, 4, 4, 4, 4, 8],
        ),
        (
            [9, 1, 8, 2, 7, 3, 6],
            median_lower,
            [1, 1, 2, 2, 3, 3, 6],
            median_upper,
            [6, 6, 6, 6, 6, 6, 6],
        ),
        (
            [9, 1, 8, 4, -2, 8, 6],
            median_lower,
            [1, 1, 4, 4, 4, 6, 6],
            median_upper,
            [4, 4, 4, 4, 4, 8, 8],
        ),
        (
            [8, 4, 8, 2, 2, 0, 8],
            quantile80_lower,
            [8, 8, 8, 8, 8, 8, 8],
            quantile80_upper,
            [8, 8, 8, 8, 8, 8, 8],
        ),
        (
            [9, 1, 8, 2, 7, 3, 6],
            quantile80_lower,
            [8, 8, 8, 8, 8, 8, 8],
            quantile80_upper,
            [8, 8, 8, 8, 8, 8, 8],
        ),
        (
            [9, 1, 8, 4, -2, 8, 6],
            quantile80_lower,
            [8, 8, 8, 8, 8, 8, 8],
            quantile80_upper,
            [8, 8, 8, 8, 8, 8, 8],
        ),
        (
            [8, 4, 4, 4, 0, 8, 2, 2, 2, 0, 8],
            median_lower,
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 8],
            median_upper,
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 8],
        ),
        (
            [
                -3.444949308,
                -2.854784791,
                -3.071818855,
                -1.164616209,
                -3.947223934,
                -2.644162705,
                -2.609553073,
                -0.270605801,
                -3.208354459,
                -2.012322283,
                -1.059124491,
                -2.199654113,
                -1.906487545,
                -2.681030502,
                -1.553997732,
                -3.489485619,
                -1.672190775,
                -2.863511214,
                -0.491263988,
                0.169577379,
            ],
            quantile10_lower,
            # from R package isotonic, function
            # gpava(1:20, y, solver = weighted.fractile, p=0.1, ties = "secondary")$x
            np.repeat(
                [-3.947223934, -3.208354459, -2.863511214, -0.491263988, 0.169577379],
                [5, 11, 2, 1, 1],
            ),
            quantile10_upper,
            np.repeat(
                [-3.947223934, -3.208354459, -2.863511214, -0.491263988, 0.169577379],
                [5, 11, 2, 1, 1],
            ),
        ),
    ],
)
def test_gpava_simple(y, fun_lower, x_lower, fun_upper, x_upper):
    w = np.ones_like(y)
    xl, rl = gpava(fun_lower, y, w)
    # The distinction of lower and upper only applies to set-valued functionals like
    # quantiles.
    # Applying gpava on fun=np.quantile(x, level, method="higher") does not work
    # for unknown reasons. What works is to calculate the upper quantile on the
    # block given by rl.
    upper_values = np.fromiter(
        (fun_upper(y[rl[i] : rl[i + 1]]) for i in range(len(rl) - 1)), dtype=xl.dtype
    )
    # Take mininum from the right.
    upper_values = np.minimum.accumulate(upper_values[::-1])[::-1]
    xu = np.repeat(upper_values, np.diff(rl))

    # Check that isotonic_regression gives midpoint.
    functional = fun_lower.loss.functional
    level = getattr(fun_lower.loss, "level", 0.5)
    x_iso, _ = isotonic_regression(y=y, functional=functional, level=level)
    assert_allclose(0.5 * (xl + xu), x_iso)

    for i in range(len(xl) - 1):
        # Eq. 10 of Jordan et al. https://doi.org/10.1007/s10463-021-00808-0
        assert xl[i] <= xu[i + 1]

    # Check manually precomputed values.
    assert_array_equal(xl, x_lower)
    assert_array_equal(xu, x_upper)

    # Check that it is a minimum.
    loss_lower = fun_lower.loss(y_obs=y, y_pred=xl)
    loss_upper = fun_lower.loss(y_obs=y, y_pred=xu)
    assert loss_lower == pytest.approx(loss_upper)

    def objective(y_pred, y_obs):
        return fun_lower.loss(y_obs=y_obs, y_pred=y_pred)

    def constraint(y_pred, y_obs):
        # This is for a monotonically increasing regression.
        return np.diff(y_pred)

    result = minimize(
        fun=objective,
        x0=y,
        args=(y,),
        constraints=[{"type": "ineq", "fun": lambda x: constraint(x, y)}],
        tol=1e-12,
    )

    assert result.fun == pytest.approx(loss_lower, rel=1e-8)

    # Check that it is the boundary of solutions.
    eps = 1e-8 * np.mean(np.abs(xl))
    for i in range(len(xl)):
        x = np.array(xl, copy=True, dtype=float)
        if i == 0 or x[i] > x[i - 1]:
            x[i] -= eps
            assert fun_lower.loss(y_obs=y, y_pred=x) > loss_lower

        x = np.array(xu, copy=True, dtype=float)
        if i == len(xl) - 1 or x[i] < x[i + 1]:
            x[i] += eps
            assert fun_upper.loss(y_obs=y, y_pred=x) > loss_upper


@pytest.mark.parametrize(
    ("functional", "level", "msg"),
    [
        ("no good functional", 0.5, "Argument functional must be one of"),
        ("quantile", 0, "Argument level must fulfil 0 < level < 1"),
        ("quantile", 1, "Argument level must fulfil 0 < level < 1"),
        ("quantile", 1.1, "Argument level must fulfil 0 < level < 1"),
    ],
)
def test_isotonic_regression_raises(functional, level, msg):
    y, w = np.arange(5), np.arange(5)
    with pytest.raises(ValueError, match=msg):
        isotonic_regression(y=y, weights=w, functional=functional, level=level)


@pytest.mark.parametrize("functional", ["quantile", "median"])
def test_isotonic_regression_raises_weighted_quantile(functional):
    y, w = np.arange(5), np.arange(5)
    msg = "Weighted quantile"
    with pytest.raises(NotImplementedError, match=msg):
        isotonic_regression(y=y, weights=w, functional=functional, level=0.5)


@pytest.mark.parametrize(
    ("weights", "msg"),
    [
        ([0, 1, 2], "Input arrays y and w must have one dimension of equal length."),
        ([0, -1], "Weights w must be strictly positive."),
    ],
)
def test_isotonic_regression_raises_weight_array_validation(weights, msg):
    """Test input validation of weight array."""
    y = [0, 1]
    with pytest.raises(ValueError, match=msg):
        isotonic_regression(y=y, weights=weights)


@pytest.mark.parametrize("w", [None, np.ones(7)])
def test_simple_isotonic_regression(w):
    # Test case of Busing 2020
    # https://doi.org/10.18637/jss.v102.c01
    y = np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64)
    x, r = isotonic_regression(y, w)
    assert_allclose(x, [4, 4, 4, 4, 4, 4, 8])
    assert_allclose(np.add.reduceat(np.ones_like(y), r[:-1]), [6, 1])
    assert_allclose(r, [0, 6, 7])
    # Assert that y was not overwritten
    assert_array_equal(y, np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64))


@pytest.mark.parametrize(
    ("functional", "level", "x_res", "r_res"),
    [
        ("mean", None, [4, 4, 4, 4, 4, 4, 8], [0, 6, 7]),
        ("median", None, [3, 3, 3, 3, 3, 3, 8], [0, 6, 7]),
        ("quantile", 0.5, [3, 3, 3, 3, 3, 3, 8], [0, 6, 7]),
        ("quantile", 0.8, [8, 8, 8, 8, 8, 8, 8], [0, 7]),
        ("expectile", 0.5, [4, 4, 4, 4, 4, 4, 8], [0, 6, 7]),
        ("expectile", 0.8, [6, 6, 6, 6, 6, 6, 8], [0, 6, 7]),
    ],
)
def test_simple_isotonic_regression_functionals(functional, level, x_res, r_res):
    # Test case of Busing 2020
    # https://doi.org/10.18637/jss.v102.c01
    y = np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64)
    x, r = isotonic_regression(y, functional=functional, level=level)
    assert_allclose(x, x_res)
    assert_allclose(r, r_res)
    # Assert that y was not overwritten
    assert_array_equal(y, np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64))


@pytest.mark.parametrize("increasing", [True, False])
def test_linspace(increasing):
    n = 10
    y = np.linspace(0, 1, n) if increasing else np.linspace(1, 0, n)
    x, r = isotonic_regression(y, increasing=increasing)
    assert_allclose(x, y)
    assert_array_equal(r, np.arange(n + 1))


def test_weights():
    w = np.array([1, 2, 5, 0.5, 0.5, 0.5, 1, 3])
    y = np.array([3, 2, 1, 10, 9, 8, 20, 10])
    x, r = isotonic_regression(y, w)
    assert_allclose(x, [12 / 8, 12 / 8, 12 / 8, 9, 9, 9, 50 / 4, 50 / 4])
    assert_allclose(np.add.reduceat(w, r[:-1]), [8, 1.5, 4])
    assert_array_equal(r, [0, 3, 6, 8])

    # weights are like repeated observations, we repeat the 3rd element 5
    # times.
    w2 = np.array([1, 2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 1, 3])
    y2 = np.array([3, 2, 1, 1, 1, 1, 1, 10, 9, 8, 20, 10])
    x2, r2 = isotonic_regression(y2, w2)
    assert_allclose(np.diff(x2[0:7]), 0)
    assert_allclose(x2[4:], x)
    assert_allclose(np.add.reduceat(w2, r2[:-1]), np.add.reduceat(w, r[:-1]))
    assert_array_equal(r2 - [0, 4, 4, 4], r)


def test_against_R_monotone():
    y = [0, 6, 8, 3, 5, 2, 1, 7, 9, 4]
    x, r = isotonic_regression(y)
    # R code
    # library(monotone)
    # options(digits=8)
    # monotone(c(0, 6, 8, 3, 5, 2, 1, 7, 9, 4))
    res = [
        0,
        4.1666667,
        4.1666667,
        4.1666667,
        4.1666667,
        4.1666667,
        4.1666667,
        6.6666667,
        6.6666667,
        6.6666667,
    ]
    assert_allclose(x, res)
    assert_array_equal(r, [0, 1, 7, 10])

    n = 100
    y = np.linspace(0, 1, num=n, endpoint=False)
    y = 5 * y + np.sin(10 * y)
    x, r = isotonic_regression(y)
    # R code
    # library(monotone)
    # y <- 5 * ((1:n)-1)/n + sin(10 * ((1:n)-1)/n)
    # monotone(y)
    res = [
        0.0000000,
        0.1498334,
        0.2986693,
        0.4455202,
        0.5894183,
        0.7294255,
        0.8646425,
        0.9942177,
        1.1173561,
        1.2333269,
        1.3414710,
        1.4412074,
        1.5320391,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.5708110,
        1.6241853,
        1.7165453,
        1.8177326,
        1.9272355,
        2.0444597,
        2.1687334,
        2.2993145,
        2.4353978,
        2.5761233,
        2.7205845,
        2.8678375,
        3.0169106,
        3.1668139,
        3.3165492,
        3.4651200,
        3.6115414,
        3.7548499,
        3.8941134,
        4.0284398,
        4.1569866,
        4.2789690,
        4.3936679,
        4.5004366,
        4.5987081,
        4.6880000,
        4.7679197,
        4.8381682,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
        4.8656413,
    ]
    assert_allclose(x, res, rtol=2e-7)

    # Test increasing
    assert np.all(np.diff(x) >= 0)

    # Test balance property: sum(y) == sum(x)
    assert_allclose(np.sum(x), np.sum(y))

    # Reverse order
    x, rinv = isotonic_regression(-y, increasing=False)
    assert_allclose(-x, res, rtol=2e-7)
    assert_array_equal(rinv, r)


@pytest.mark.parametrize("increasing", [True, False])
def test_isotonic_regression_class(increasing):
    """Test that IsotonicRegression gives the same as the scikit-learn version."""
    X = [0, 2, 4, -1, -2, 3, 2, 2, 1, 4]
    y = np.arange(10)
    m_skl = skl_IsotonicRegression(increasing=increasing, out_of_bounds="clip").fit(
        X, y
    )
    m = IsotonicRegression(increasing=increasing).fit(X, y)

    assert_allclose(m.X_thresholds_, m_skl.X_thresholds_)
    assert_allclose(m.y_thresholds_, m_skl.y_thresholds_)

    assert m.X_thresholds_[0] == m_skl.X_min_
    assert m.X_thresholds_[-1] == m_skl.X_max_

    X_pred = [-10, -2, 1, 2.5, 10]
    assert_allclose(m.predict(X_pred), m_skl.predict(X_pred))

    m_exp = IsotonicRegression(increasing=increasing, functional="expectile").fit(X, y)
    assert_allclose(m.X_thresholds_, m_exp.X_thresholds_)
    assert_allclose(m.y_thresholds_, m_exp.y_thresholds_)
    assert_allclose(m.predict(X_pred), m_exp.predict(X_pred))


def test_isotonic_regression_class_median():
    """Test IsotonicRegression for median regression."""
    rng = np.random.default_rng(42)
    # Test case of Busing 2020
    # https://doi.org/10.18637/jss.v102.c01
    y = np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64)
    y_iso = np.array([3, 3, 3, 3, 3, 3, 8])
    X = np.arange(len(y))
    idx = rng.permutation(X)

    m = IsotonicRegression(functional="quantile", level=0.5).fit(X[idx], y[idx])
    assert_allclose(m.X_thresholds_, [0, 5, 6])
    assert_allclose(m.y_thresholds_, [3, 3, 8])
    assert_allclose(m.predict(X), y_iso)
