from typing import Optional

import numpy as np
import numpy.typing as npt

from .array import validate_2_arrays


def _pava(
    y: npt.ArrayLike,
    w: Optional[npt.ArrayLike] = None,
):
    r"""Pool adjacent violators algorithm (PAVA).

    Uses the PAVA in an efficient O(n) verision according to `[Busing]`. Only the mean
    functional is implemented.

    Parameters
    ----------
    y : array-like of shape (n_obs)
        Observed values of the response variable, already ordered accoring to some
        feature.
    w : array-like of shape (n_obs)
        Case (aka sample) weights.

    Returns
    -------
    partitions :
    x : solution (mean functional for the moment)

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
        w = np.ones_like(y)
    else:
        y, w = validate_2_arrays(y, w)
    n: int = y.shape[0]
    x = np.copy(y)
    r = np.full(shape=n + 1, fill_value=-1, dtype=int)

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
    xb_prev = x[b]  # 5: set previous block value
    wb_prev = w[b]  # 6: set previous block weight
    # for(i=1; i<n; ++i)               # 7: loop over elements
    i = 1
    while i < n:
        b += 1  # 8: increase number of blocks
        xb = x[i]  # 9: set current block value xb (i, not b)
        wb = w[i]  # 10: set current block weight wb (i, not b)
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

    return x, r
