from numpy.testing import assert_array_equal

from model_diagnostics._utils.isotonic import _pava


def test_pava_simple():
    y = [8, 4, 8, 2, 2, 0, 8]
    w = [1, 1, 1, 1, 1, 1, 1]
    x, r = _pava(y, w)
    assert_array_equal(x, [4, 4, 4, 4, 4, 4, 8])
    assert_array_equal(r, [0, 6, 7, -1, -1, -1, -1, -1])

    y = [9, 1, 8, 2, 7, 3, 6]
    x, r = _pava(y, w)
    assert_array_equal(x, [5, 5, 5, 5, 5, 5, 6])
    assert_array_equal(r, [0, 6, 7, -1, -1, -1, -1, -1])

    y = [9, 1, 8, 4, -2, 8, 6]
    x, r = _pava(y, w)
    assert_array_equal(x, [4, 4, 4, 4, 4, 7, 7])
    assert_array_equal(r, [0, 5, 7, -1, -1, -1, -1, -1])

def test_pava_weighted():
    y = [8, 4, 8, 2, 2, 0, 8]
    w = [1, 2, 3, 4, 5, 6, 7]
    x, r = _pava(y, w)
    #assert_array_equal(x, [4, 4, 4, 4, 4, 4, 8])
