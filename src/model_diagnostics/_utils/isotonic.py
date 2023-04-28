from typing import Optional

import numpy as np
import numpy.typing as npt

from ._array import validate_2_arrays


def pava(
    y: npt.NDArray,
    w: npt.NDArray,
):
    r"""Pool adjacent violators algorithm (PAVA).

    Uses the PAVA in an efficient O(n) verision according to `[Busing]`. Only the mean
    functional is implemented.

    Parameters
    ----------
    y : ndarray of shape (n_obs)
        Observed values of the response variable, already ordered according to some
        feature.
    w : ndarray of shape (n_obs)
        Case weights.

    Returns
    -------
    x : ndarray of shape (n_obs)
        Solution to the isotonic regression problem.
    r : (n_blocks + 1) ndarray
        Array of indices with the start position of each block / pool `n_blocks`.
        For the j-th block, all values of `y[r[j]:r[j+1]]` are the same. It always
        holds that `r[0] = 0`.

    Notes
    -----
    Solves:

    \[
    \sum_i^n w_i L(x_i, y_i) \quad \text{subject to } x_i \leq x_j \forall i<j\,.
    \]

    References
    ----------
    `[Busing]`

    :   Busing, F. M. T. A. (2022).
        "Monotone Regression: A Simple and Fast O(n) PAVA Implementation".
        Journal of Statistical Software, Code Snippets, 102(1), 1-25.
        https://doi.org/10.18637/jss.v102.c01
    """
    if w is None:
        y = np.asarray(y)
        w = np.ones_like(y, dtype=float)
    else:
        y, w = validate_2_arrays(y, w)
        w = w.astype(float)  # copies w
    n: int = y.shape[0]
    x = y.astype(float)  # copies y
    r = np.full(shape=n + 1, fill_value=-1, dtype=np.intp)

    # Algorithm 1 of Busing2022
    # Notes:
    #  - We translated it to 0-based indices.
    #  - xb, wb, sb instead of x, w and S to avoid name collisions
    #  - xb_prev and wb_prev instead of x_hat and w_hat
    #  - for loop of line 7 replaced by a while loop due to interactions with loop
    #    counter i
    #  - ERROR CORRECTED: Lines 9 and 10 have index i instead of b.
    #  - MODIFICATIONS: Lines 11 and 22 both have >= instead of >
    #    to get correct block indices in r. Otherwise, same values can get in
    #    different blocks, e.g. x = [2, 2] would produce
    #    r = [0, 1, 2] instead of r = [0, 2].
    #
    # procedure monotone(n, x, w)      # 1: x in expected order and w nonnegative
    r[0] = 0  # 2: initialize index 0
    r[1] = 1  # 3: initialize index 1
    b: int = 0  # 4: initialize block counter
    xb_prev = y[0]  # 5: set previous block value
    wb_prev = w[0]  # 6: set previous block weight
    # for(i=1; i<n; ++i)               # 7: loop over elements
    i = 1
    while i < n:
        b += 1  # 8: increase number of blocks
        xb = x[i]  # 9: set current block value xb (i, not b)
        wb = w[i]  # 10: set current block weight wb (i, not b)
        sb = 0.0
        if xb_prev >= xb:  # 11: check for down violation of x (>= instead of >)
            b -= 1  # 12: decrease number of blocks
            sb = wb_prev * xb_prev + wb * xb  # 13: set current weighted block sum
            wb += wb_prev  # 14: set new current block weight
            xb = sb / wb  # 15: set new current block value
            while i < n - 1 and xb >= x[i + 1]:  # 16: repair up violations
                i += 1
                sb += w[i] * x[i]  # 18: set new current weighted block sum
                wb += w[i]
                xb = sb / wb
            while b >= 1 and x[b - 1] >= xb:  # 22: repair down violations (>= instead)
                b -= 1
                sb += w[b] * x[b]
                wb += w[b]
                xb = sb / wb  # 26: set new current block value

        x[b] = xb_prev = xb  # 29: save block value
        w[b] = wb_prev = wb  # 30: save block weight
        r[b + 1] = i + 1  # 31: save block index
        i += 1

    f = n - 1  # 33: initialize "from" index
    for k in range(b, -1, -1):  # 34: loop over blocks
        t = r[k]  # 35: set "to" index
        xk = x[k]
        for i in range(f, t - 1, -1):  # 37: loop "from" downto "to"
            x[i] = xk  # 38: set all elements equal to block value
        f = t - 1  # 40: set new "from" equal to old "to" minus one

    return x, r[: b + 2]


def isotonic_regression(
    y: npt.ArrayLike,
    weights: Optional[npt.ArrayLike] = None,
    *,
    increasing: bool = True,
):
    r"""Nonparametric isotonic regression.
    A monotonically increasing array `x` with the same length as `y` is
    calculated by the pool adjacent violators algorithm (PAVA), see [1]_.
    See the Notes section for more details.

    Parameters
    ----------
    y : (n_obs,) array_like
        Response variable.
    weights : (n_obs,) array_like or None
        Case weights.
    increasing : bool
        If True, fit monotonic increasing, i.e. isotonic, regression.
        If False, fit a monotonic decreasing, i.e. antitonic, regression.

    Returns
    -------
    x : (n_obs,) ndarray
        Isotonic regression solution, i.e. an increasing (or decresing) array
        of the same length than y.
    r : (n_blokcs+1,) ndarray
        Array of indices with the start position of each block / pool B.
        For the j-th block, all values of `x[r[j]:r[j+1]]` are the same.

    Notes
    -----
    Given data :math:`y` and case weights :math:`w`, the isotonic regression
    solves the following optimization problem:

    .. math::
        \operatorname{argmin}_{x_i} \sum_i w_i (y_i - x_i)^2 \quad
        \text{subject to } x_i \leq x_j \text{ whenever } i \leq j \,.

    For every input value :math:`y_i`, it generates an interpolated value
    :math:`x_i` which are increasing. This is accomplished by the PAVA.
    The solution consists of pools or blocks, i.e. neighboring elements of
    :math:`x`, e.g. :math:`x_i` and :math:`x_{i+1}`, that all have the same
    value.
    Most interestingly, the solution stays the same if the squared loss is
    replaced by the wide class of Bregman functions which are the unique
    class of strictly consistent scoring functions for the mean, see [2]_
    and references therein.

    References
    ----------
    .. [1] Busing, F. M. T. A. (2022).
           Monotone Regression: A Simple and Fast O(n) PAVA Implementation.
           Journal of Statistical Software, Code Snippets, 102(1), 1-25.
           :doi:`10.18637/jss.v102.c01`
    .. [2] Jordan, A.I., Mühlemann, A. & Ziegel, J.F.
           Characterizing the optimal solutions to the isotonic regression
           problem for identifiable functionals.
           Ann Inst Stat Math 74, 489-514 (2022).
           :doi:`10.1007/s10463-021-00808-0`
    """
    y = np.asarray(y)
    if weights is None:
        weights = np.ones_like(y)
    else:
        weights = np.asarray(weights)

        if not (y.ndim == weights.ndim and y.shape[0] == weights.shape[0]):
            msg = "Input arrays y and w must have one dimension of equal length."
            raise ValueError(msg)
        if np.any(weights <= 0):
            msg = "Weights w must be strictly positive."
            raise ValueError(msg)

    order = np.s_[:] if increasing else np.s_[::-1]
    x = np.array(y[order], order="C", dtype=np.float64, copy=True)
    wx = np.array(weights[order], order="C", dtype=np.float64, copy=True)
    x.shape[0]
    x, r = pava(x, wx)
    if not increasing:
        x = x[::-1]
        r = r[-1] - r[::-1]
    return x, r
