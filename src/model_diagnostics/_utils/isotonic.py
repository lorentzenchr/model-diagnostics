from decimal import Decimal
from functools import partial
from typing import Callable, Optional

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy.interpolate import interp1d
from scipy.stats import expectile

from .array import length_of_second_dimension, validate_2_arrays


def quantile_lower(x, wx=None, level=0.5):
    return np.quantile(x, level, method="inverted_cdf")


def quantile_upper(x, wx=None, level=0.5):
    # np.quantile(x, level, method="higher") is not the same as
    # -np.quantile(-x, 1 - level, method="inverted_cdf")
    # Also note that 1 - level can have a loss of precision, e.g. 1 - 0.1
    # = 0.09999999999999998 but 0.1 would be right.
    return -np.quantile(-x, float(1 - Decimal(str(level))), method="inverted_cdf")


def pava(
    y: npt.NDArray,
    w: Optional[npt.NDArray] = None,
):
    r"""Pool adjacent violators algorithm (PAVA).

    Uses the PAVA in an efficient O(n) verision according to `[Busing]`. Only the mean
    functional is implemented.

    Parameters
    ----------
    y : ndarray of shape (n_obs)
        Observed values of the response variable, already ordered according to some
        feature.
    w : ndarray of shape (n_obs) or None
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

    for all loss functions \(L\) that are strictly consistent for the mean functional.

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


def gpava(
    fun: Callable[[npt.NDArray, Optional[npt.NDArray]], tuple],
    y: npt.NDArray,
    w: Optional[npt.NDArray] = None,
):
    r"""Generalise Pool adjacent violators algorithm (PAVA).

    This is a generalisation of the PAVA, as implemented in function `pava`, that works
    for any identifiable functional like quantiles (and not only the mean).

    Parameters
    ----------
    fun : callable
        Function that calculates the functional at interest, e.g.
        ```py
        def median(x, w=None):
            return np.quantile(x, 0.5, method="inverted_cdf")
        ```
        or, to get the upper bound
        ```py
        def median(x, w=None):
            return -np.quantile(-x, 0.5, method="inverted_cdf")
        ```
    y : ndarray of shape (n_obs)
        Observed values of the response variable, already ordered according to some
        feature.
    w : ndarray of shape (n_obs) or None
        Case weights.

    Returns
    -------
    x : ndarray of shape (n_obs)
        The solution to the isotonic regression problem.
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

    for all loss functions \(L\) that are strictly consistent for the functional under
    consideration. The callable `fun` determines, which solution in case of set valued
    functionals like quantiles.

    References
    ----------
    `[Jordan]`

    :   Alexander I. Jordan, Anja Mühlemann, Johanna F. Ziegel (2021).
        "Characterizing the optimal solutions to the isotonic regression problem for
        identifiable functionals".
        Annals of the Institute of Statistical Mathematics (2022) 74:489-514
        https://doi.org/10.1007/s10463-021-00808-0
    """
    # Note: In an email exchange with the above authors, we got confirmed of a subtle
    # bug in the algorithm of Jordan et al (2021) when applied to quantiles. They
    # confirmed, however, that it is good and sound to apply the standard pava once
    # for the lower and once for the upper quantile.
    # Therefore, this routine applies pava to just one single valued functional as
    # specified by the callable fun, e.g. the upper quantile is a singleton.

    if w is None:
        y = np.asarray(y)
        w = np.ones_like(y, dtype=float)
    else:
        y, w = validate_2_arrays(y, w)
        w = w.astype(float)  # copies w
    n: int = y.shape[0]
    # Let us assume fun(x) = x, for a single data point x. Otherwise, we would need
    # for i in range(n):
    #     x[i] = fun(x[i:i+1], w[i:i+1])
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
    # wb_prev = w[0]  # 6: set previous block weight
    # for(i=1; i<n; ++i)               # 7: loop over elements
    i = 1
    while i < n:
        b += 1  # 8: increase number of blocks
        xb = x[i]  # 9: set current block value xb (i, not b)
        # wb = w[i]  # 10: set current block weight wb (i, not b)
        # sb = 0.0
        if xb_prev >= xb:  # 11: check for down violation of x
            b -= 1  # 12: decrease number of blocks
            # sb = wb_prev * xb_prev + wb * xb  # 13: set current weighted block sum
            # wb += wb_prev  # 14: set new current block weight
            # xb = sb / wb  # 15: set new current block value
            xb = fun(y[r[b] : r[b + 1] + 1], w[r[b] : r[b + 1] + 1])
            while i < n - 1 and xb >= x[i + 1]:  # 16: repair up violations
                i += 1
                # sb += w[i] * x[i]  # 18: set new current weighted block sum
                # wb += w[i]
                # xb = sb / wb
                xb = fun(y[r[b] : i + 1], w[r[b] : i + 1])
            while b >= 1 and x[b - 1] >= xb:  # 22: repair down violations
                b -= 1
                # sb += w[b] * x[b]
                # wb += w[b]
                # xb = sb / wb  # 26: set new current block value
                xb = fun(y[r[b] : i + 1], w[r[b] : i + 1])

        x[b] = xb_prev = xb  # 29: save block value
        # w[b] = wb_prev = wb  # 30: save block weight
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
    functional: str = "mean",
    level: float = 0.5,
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
    functional : str
        The functional that is induced by the identification function `V`. Options are:

        - `"mean"`. Argument `level` is neglected.
        - `"median"`. Argument `level` is neglected.
        - `"expectile"`
        - `"quantile"`

    level : float
        The level of the expectile of quantile. (Often called \(\alpha\).)
        It must be `0 < level < 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.

    Returns
    -------
    x : (n_obs,) ndarray
        Isotonic regression solution, i.e. an increasing (or decresing) array
        of the same length than y.
    r : (n_blocks+1,) ndarray
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
    allowed_functionals = ("mean", "median", "expectile", "quantile")
    if functional not in allowed_functionals:
        msg = (
            f"Argument functional must be one of {allowed_functionals}, got "
            f"{functional}."
        )
        raise ValueError(msg)
    if functional in ("expectile", "quantile") and (level <= 0 or level >= 1):
        msg = f"Argument level must fulfil 0 < level < 1, got {level}."
        raise ValueError(msg)
    if functional == "median":
        functional = "quantile"
        level = 0.5

    y = np.asarray(y)
    if weights is None:
        weights = np.ones_like(y)
    else:
        if functional in "quantile":
            msg = "Weighted quantiles are not yet implemented."
            raise NotImplementedError(msg)
        weights = np.asarray(weights)

        if not (y.ndim == weights.ndim and y.shape[0] == weights.shape[0]):
            msg = "Input arrays y and w must have one dimension of equal length."
            raise ValueError(msg)
        if np.any(weights <= 0):
            msg = "Weights w must be strictly positive."
            raise ValueError(msg)

    order = np.s_[:] if increasing else np.s_[::-1]  # type: ignore
    x = y[order]
    wx = weights[order]

    if functional == "mean":
        x, r = pava(x, wx)
    elif functional == "expectile":

        def expectile_fun(x, w):
            return expectile(x, alpha=level, weights=w)

        x, r = gpava(expectile_fun, x, wx)
    elif functional == "quantile":
        xl, rl = gpava(partial(quantile_lower, level=level), x, wx)
        # Applying gpava on fun=np.quantile(x, level, method="higher") does not work
        # for unknown reasons. What works is to calculate the upper quantile on the
        # blocks given by rl from the lower quantile.
        q = np.fromiter(
            (
                partial(quantile_upper, level=level)(x[rl[i] : rl[i + 1]])
                for i in range(len(rl) - 1)
            ),
            dtype=xl.dtype,
        )
        # Take mininum from the right.
        q = np.minimum.accumulate(q[::-1])[::-1]
        xu = np.repeat(q, np.diff(rl))
        # We are free to use any value in the interval [xl, xu], as long as it is
        # increasing. We choose the midpoint.
        x = 0.5 * (xl + xu)
        # We recompute r to make it of minimal length again.
        r = np.nonzero(np.diff(x))[0] + 1
        r = np.r_[0, r, len(x)]

    if not increasing:
        x = x[::-1]
        r = r[-1] - r[::-1]
    return x, r


class IsotonicRegression:
    """Isotonic regression model.

    Parameters
    ----------
    increasing : bool or 'auto', default=True
        Determines whether the predictions should be constrained to increase
        or decrease with `X`. 'auto' will decide based on the Spearman
        correlation estimate's sign.
    functional : str
        The functional that is induced by the identification function `V`. Options are:

        - `"mean"`. Argument `level` is neglected.
        - `"median"`. Argument `level` is neglected.
        - `"expectile"`
        - `"quantile"`

    level : float
        The level of the expectile or quantile. (Often called \\(\alpha\\).)
        It must be `0 <= level <= 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.

    Attributes
    ----------
    X_thresholds_ : ndarray of shape (n_thresholds,)
        Unique ascending `X` values used to interpolate
        the y = f(X) monotonic function.

    y_thresholds_ : ndarray of shape (n_thresholds,)
        De-duplicated `y` values suitable to interpolate the y = f(X)
        monotonic function.

    f_ : function
        The stepwise interpolating function that covers the input domain ``X``.

    increasing_ : bool
        Inferred value for ``increasing``.
    """

    def __init__(
        self,
        *,
        increasing: bool = True,
        functional: str = "mean",
        level: float = 0.5,
    ):
        self.increasing = increasing
        self.functional = functional
        self.level = level

    def fit(
        self,
        X: npt.ArrayLike,
        y: npt.ArrayLike,
        sample_weight: Optional[npt.ArrayLike] = None,
    ):
        """Fit the model using X, y as training data.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Training data.

        y : array-like of shape (n_samples,)
            Training target.

        sample_weight : array-like of shape (n_samples,), default=None
            Weights. If set to None, all weights will be set to 1 (equal
            weights).

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if (n_cols := length_of_second_dimension(X)) >= 2:
            msg = f"X must have only one colume, got X.shape[1] = {n_cols}."
            raise ValueError(msg)
        if n_cols == 1:
            X = np.asarray(X)[:, 0]
        df = pl.DataFrame({"_X": X, "_target_y": y})
        if sample_weight is not None:
            df = df.hstack([pl.Series(name="_weights", values=sample_weight)])
        # We deal with duplicate values in X, aka ties, by also sorting y, but in
        # reverse order.
        df = df.sort(by=["_X", "_target_y"], descending=[False, self.increasing])
        yy = df["_target_y"].to_numpy()
        wy = df["_weights"].to_numpy() if sample_weight is not None else None

        y_iso, r = isotonic_regression(
            y=yy,
            weights=wy,
            increasing=self.increasing,
            functional=self.functional,
            level=self.level,
        )

        X_sorted = df.get_column("_X")
        idx_list = [r[0]]
        for i in range(1, len(r) - 1):
            # Check previous block has more than one element.
            if r[i] - 1 - r[i - 1] >= 1:
                idx_list.append(r[i] - 1)
            idx_list.append(r[i])
        # FIXME: Older versions of polars don't allow numpy integers as indices.
        if (X_sorted[int(r[-1] - 1)] != X_sorted[int(r[-2])]) and (
            y_iso[0] == y_iso[-1] or r[-1] - 1 - r[-2] >= 1
        ):
            # In case all y values are the same, we include this index.
            idx_list.append(r[-1] - 1)
        idx = np.asarray(idx_list)

        # Almost the same, might have some duplicates:
        # idx = np.sort(np.r_[r[:-1], r[1:] - 1])

        self.X_thresholds_, self.y_thresholds_ = X_sorted[idx].to_numpy(), y_iso[idx]

        # Build the interpolation function
        self.f_ = interp1d(
            self.X_thresholds_,
            self.y_thresholds_,
            kind="linear",
            bounds_error=False,
            fill_value=(self.y_thresholds_[0], self.y_thresholds_[-1]),
        )
        return self

    def predict(self, X):
        """Predict new data by linear interpolation.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, 1)
            Data to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Transformed data.
        """
        return self.f_(X)
