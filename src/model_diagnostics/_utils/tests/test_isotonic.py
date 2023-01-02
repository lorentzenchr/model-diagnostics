from numpy.testing import assert_array_equal

from model_diagnostics._utils.isotonic import _pava


def test_pava_simple():
    y = [8, 4, 8, 2, 2, 0, 8]
    w = [1, 1, 1, 1, 1, 1, 1]
    x, r = _pava(y, w)
    print(x)
    print(r)
    assert_array_equal(x, [4, 4, 4, 4, 4, 4, 8])
    assert_array_equal(r, [0, 6, 7, -1, -1, -1, -1, -1])
