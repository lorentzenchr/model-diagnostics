import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from model_diagnostics._utils.isotonic import _pava, isotonic_regression


def test_pava_simple():
    # Test case of Busing 2020
    # https://doi.org/10.18637/jss.v102.c01
    y = [8, 4, 8, 2, 2, 0, 8]
    w = np.ones_like(y)
    x, r = _pava(y, w)
    assert_array_equal(x, [4, 4, 4, 4, 4, 4, 8])
    assert_array_equal(r, [0, 6, 7])

    y = [9, 1, 8, 2, 7, 3, 6]
    w = None
    x, r = _pava(y, w)
    assert_array_equal(x, [5, 5, 5, 5, 5, 5, 6])
    assert_array_equal(r, [0, 6, 7])

    y = [9, 1, 8, 4, -2, 8, 6]
    x, r = _pava(y, w)
    assert_array_equal(x, [4, 4, 4, 4, 4, 7, 7])
    assert_array_equal(r, [0, 5, 7])


def test_pava_weighted():
    y = [8, 4, 8, 2, 2, 0, 8]
    w = [1, 2, 3, 4, 5, 5, 4]
    x, r = _pava(y, w)
    assert_array_equal(x, [2.9, 2.9, 2.9, 2.9, 2.9, 2.9, 8])
    assert_array_equal(r, [0, 6, 7])


@pytest.mark.parametrize("w", [None, np.ones(7)])
def test_simple_isotonic_regression(w):
    # Test case of Busing 2020
    # https://doi.org/10.18637/jss.v102.c01
    y = np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64)
    x, r = isotonic_regression(y, w)
    assert_almost_equal(x, [4, 4, 4, 4, 4, 4, 8])
    assert_almost_equal(np.add.reduceat(np.ones_like(y), r[:-1]), [6, 1])
    assert_almost_equal(r, [0, 6, 7])
    # Assert that y was not overwritten
    assert_array_equal(y, np.array([8, 4, 8, 2, 2, 0, 8], dtype=np.float64))


@pytest.mark.parametrize("increasing", [True, False])
def test_linspace(increasing):
    n = 10
    y = np.linspace(0, 1, n) if increasing else np.linspace(1, 0, n)
    x, r = isotonic_regression(y, increasing=increasing)
    assert_almost_equal(x, y)
    assert_array_equal(r, np.arange(n + 1))


def test_weights():
    w = np.array([1, 2, 5, 0.5, 0.5, 0.5, 1, 3])
    y = np.array([3, 2, 1, 10, 9, 8, 20, 10])
    x, r = isotonic_regression(y, w)
    assert_almost_equal(x, [12 / 8, 12 / 8, 12 / 8, 9, 9, 9, 50 / 4, 50 / 4])
    assert_almost_equal(np.add.reduceat(w, r[:-1]), [8, 1.5, 4])
    assert_array_equal(r, [0, 3, 6, 8])

    # weights are like repeated observations, we repeat the 3rd element 5
    # times.
    w2 = np.array([1, 2, 1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 1, 3])
    y2 = np.array([3, 2, 1, 1, 1, 1, 1, 10, 9, 8, 20, 10])
    x2, r2 = isotonic_regression(y2, w2)
    assert_almost_equal(np.diff(x2[0:7]), 0)
    assert_almost_equal(x2[4:], x)
    assert_almost_equal(np.add.reduceat(w2, r2[:-1]), np.add.reduceat(w, r[:-1]))
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
    assert_almost_equal(x, res)
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
    assert_almost_equal(x, res)

    # Test increasing
    assert np.all(np.diff(x) >= 0)

    # Test balance property: sum(y) == sum(x)
    assert_almost_equal(np.sum(x), np.sum(y))

    # Reverse order
    x, rinv = isotonic_regression(-y, increasing=False)
    assert_almost_equal(-x, res)
    assert_array_equal(rinv, r)
