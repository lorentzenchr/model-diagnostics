"""The scoring module provides scoring functions, also known as loss functions,
and a score decomposition.
Each scoring function is implemented as a class that needs to be instantiated
before calling the `__call__` methode, e.g. `SquaredError()(y_obs=[1], y_pred=[2])`.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import special
from scipy.stats import expectile
from sklearn.isotonic import IsotonicRegression as IsotonicRegression_skl

from model_diagnostics._utils.array import (
    get_second_dimension,
    get_sorted_array_names,
    length_of_second_dimension,
    validate_2_arrays,
    validate_same_first_dimension,
)
from model_diagnostics._utils.isotonic import (
    IsotonicRegression,
    quantile_lower,
    quantile_upper,
)
from model_diagnostics.calibration import identification_function


class _BaseScoringFunction(ABC):
    """A base class for scoring functions."""

    @property
    @abstractmethod
    def functional(self):
        pass

    def __call__(
        self,
        y_obs: npt.ArrayLike,
        y_pred: npt.ArrayLike,
        weights: Optional[npt.ArrayLike] = None,
    ) -> np.floating[Any]:
        """Mean or average score.

        Parameters
        ----------
        y_obs : array-like of shape (n_obs)
            Observed values of the response variable.
        y_pred : array-like of shape (n_obs)
            Predicted values of the `functional` of interest, e.g. the conditional
            expectation of the response, `E(Y|X)`.
        weights : array-like of shape (n_obs) or None
            Case weights.

        Returns
        -------
        score : float
            The average score.
        """
        return np.average(self.score_per_obs(y_obs, y_pred), weights=weights)

    @abstractmethod
    def score_per_obs(
        self,
        y_obs: npt.ArrayLike,
        y_pred: npt.ArrayLike,
    ) -> np.ndarray:
        """Score per observation."""


class HomogeneousExpectileScore(_BaseScoringFunction):
    r"""Homogeneous scoring function of degree h for expectiles.

    The smaller the better, minimum is zero.

    Up to a multiplicative constant, these are the only scoring functions that are
    strictly consistent for expectiles at level alpha and homogeneous functions.
    The possible additive constant is chosen such that the minimal function value
    equals zero.

    Note that the 1/2-expectile (level alpha=0.5) equals the mean.

    Parameters
    ----------
    degree : float
        Degree of homogeneity.
    level : float
        The level of the expectile. (Often called \(\alpha\).)
        It must be `0 < level < 1`.
        `level=0.5` gives the mean.

    Attributes
    ----------
    functional: str
        "mean" if `level=0.5`, else "expectile"

    Notes
    -----
    The homogeneous score of degree \(h\) is given by

    \[
    S_\alpha^h(y, z) = 2 |\mathbf{1}\{z \ge y\} - \alpha| \frac{2}{h(h-1)}
    \left(|y|^h - |z|^h - h \operatorname{sign}(z) |z|^{h-1} (y-z)\right)
    \]

    Note that the first term, \(2 |\mathbf{1}\{z \ge y\} - \alpha|\) equals 1 for
    \(\alpha=0.5\).
    There are important domain restrictions and limits:

    - \(h>1\): All real numbers \(y\) and \(z\) are allowed.

        Special case \(h=2, \alpha=\frac{1}{2}\) equals the squared error, aka Normal
        deviance \(S(y, z) = (y - z)^2\).

    - \(0 < h \leq 1\): Only \(y \geq 0\), \(z>0\) are allowed.

        Special case \(h=1, \alpha=\frac{1}{2}\) (by taking the limit) equals the
        Poisson deviance \(S(y, z) = 2(y\log\frac{y}{z} - y + z)\).

    - \(h \leq 0\): Only \(y>0\), \(z>0\) are allowed.

        Special case \(h=0, \alpha=\frac{1}{2}\) (by taking the limit) equals the Gamma
        deviance \(S(y, z) = 2(\frac{y}{z} -\log\frac{y}{z} - 1)\).

    For the common domains, \(S_{\frac{1}{2}}^h\) equals the
    [Tweedie deviance](https://en.wikipedia.org/wiki/Tweedie_distribution) with the
    following relation between the degree of homogeneity \(h\) and the Tweedie
    power \(p\): \(h = 2-p\).

    References
    ----------
    `[Gneiting2011]`

    :   T. Gneiting.
        "Making and Evaluating Point Forecasts". (2011)
        [doi:10.1198/jasa.2011.r10138](https://doi.org/10.1198/jasa.2011.r10138)
        [arxiv:0912.0902](https://arxiv.org/abs/0912.0902)

    Examples
    --------
    >>> hes = HomogeneousExpectileScore(degree=2, level=0.1)
    >>> hes(y_obs=[0, 0, 1, 1], y_pred=[-1, 1, 1 , 2])  # doctest: +SKIP
    0.95
    """  # FIXME: numpy 2.0.0, doctest skip should not be necessary.

    def __init__(self, degree: float = 2, level: float = 0.5) -> None:
        self.degree = degree
        if level <= 0 or level >= 1:
            msg = f"Argument level must fulfil 0 < level < 1, got {level}."
            raise ValueError(msg)
        self.level = level

    @property
    def functional(self):
        if self.level == 0.5:
            return "mean"
        else:
            return "expectile"

    def score_per_obs(
        self,
        y_obs: npt.ArrayLike,
        y_pred: npt.ArrayLike,
    ) -> np.ndarray:
        """Score per observation.

        Parameters
        ----------
        y_obs : array-like of shape (n_obs)
            Observed values of the response variable.
        y_pred : array-like of shape (n_obs)
            Predicted values of the `functional` of interest, e.g. the conditional
            expectation of the response, `E(Y|X)`.

        Returns
        -------
        score_per_obs : ndarray
            Values of the scoring function for each observation.
        """
        y: np.ndarray
        z: np.ndarray
        y, z = validate_2_arrays(y_obs, y_pred)

        if self.degree == 2:
            # Fast path
            score = np.square(z - y)
        elif self.degree > 1:
            z_abs = np.abs(z)
            score = 2 * (
                (np.power(np.abs(y), self.degree) - np.power(z_abs, self.degree))
                / (self.degree * (self.degree - 1))
                - np.sign(z)
                / (self.degree - 1)
                * np.power(z_abs, self.degree - 1)
                * (y - z)
            )
        elif self.degree == 1:
            # Domain: y >= 0 and z > 0
            if not np.all((y >= 0) & (z > 0)):
                msg = (
                    f"Valid domain for degree={self.degree} is y_obs >= 0 and "
                    "y_pred > 0."
                )
                raise ValueError(msg)
            score = 2 * (special.xlogy(y, y / z) - y + z)
        elif self.degree == 0:
            # Domain: y > 0 and z > 0.
            if not np.all((y > 0) & (z > 0)):
                msg = (
                    f"Valid domain for degree={self.degree} is "
                    "y_obs > 0 and y_pred > 0."
                )
                raise ValueError(msg)
            y_z = y / z
            score = 2 * (y_z - np.log(y_z) - 1)
        else:  # self.degree < 1
            # Domain: y >= 0 and z > 0 for 0 < self.degree < 1
            # Domain: y > 0  and z > 0 else
            if self.degree > 0:
                if not np.all((y >= 0) & (z > 0)):
                    msg = (
                        f"Valid domain for degree={self.degree} is "
                        "y_obs >= 0 and y_pred > 0."
                    )
                    raise ValueError(msg)
            elif not np.all((y > 0) & (z > 0)):
                msg = (
                    f"Valid domain for degree={self.degree} is "
                    "y_obs > 0 and y_pred > 0."
                )
                raise ValueError(msg)
            # Note: We add 0.0 to be sure we have floating points. Integers are not
            # allowerd to be raised to a negative power.
            score = 2 * (
                (np.power(y + 0.0, self.degree) - np.power(z + 0.0, self.degree))
                / (self.degree * (self.degree - 1))
                - 1 / (self.degree - 1) * np.power(z + 0.0, self.degree - 1) * (y - z)
            )

        if self.level == 0.5:
            return score
        else:
            return 2 * np.abs(np.greater_equal(z, y) - self.level) * score


class SquaredError(HomogeneousExpectileScore):
    r"""Squared error.

    The smaller the better, minimum is zero.

    The squared error is strictly consistent for the mean.
    It has a degree of homogeneity of 2.
    In the context of probabilistic classification, it is also known as Brier score.


    Attributes
    ----------
    functional: str
        "mean"

    Notes
    -----
    \(S(y, z) = (y - z)^2\)

    Examples
    --------
    >>> se = SquaredError()
    >>> se(y_obs=[0, 0, 1, 1], y_pred=[-1, 1, 1 , 2])  # doctest: +SKIP
    0.75
    """  # FIXME: numpy 2.0.0, doctest skip should not be necessary.

    def __init__(self) -> None:
        super().__init__(degree=2, level=0.5)


class PoissonDeviance(HomogeneousExpectileScore):
    r"""Poisson deviance.

    The smaller the better, minimum is zero.

    The Poisson deviance is strictly consistent for the mean.
    It has a degree of homogeneity of 1.

    Attributes
    ----------
    functional: str
        "mean"

    Notes
    -----
    \(S(y, z) = 2(y\log\frac{y}{z} - y + z)\)

    Examples
    --------
    >>> pd = PoissonDeviance()
    >>> pd(y_obs=[0, 0, 1, 1], y_pred=[2, 1, 1 , 2])  # doctest: +SKIP
    1.6534264097200273
    """  # FIXME: numpy 2.0.0, doctest skip should not be necessary.

    def __init__(self) -> None:
        super().__init__(degree=1, level=0.5)


class GammaDeviance(HomogeneousExpectileScore):
    r"""Gamma deviance.

    The smaller the better, minimum is zero.

    The Gamma deviance is strictly consistent for the mean.
    It has a degree of homogeneity of 0 and is therefore insensitive to a change of
    units or multiplication of `y_obs` and `y_pred` by the same positive constant.

    Attributes
    ----------
    functional: str
        "mean"

    Notes
    -----
    \(S(y, z) = 2(\frac{y}{z} -\log\frac{y}{z} - 1)\)

    Examples
    --------
    >>> gd = GammaDeviance()
    >>> gd(y_obs=[3, 2, 1, 1], y_pred=[2, 1, 1 , 2])  # doctest: +SKIP
    0.2972674459459178
    """  # FIXME: numpy 2.0.0, doctest skip should not be necessary.

    def __init__(self) -> None:
        super().__init__(degree=0, level=0.5)


class LogLoss(_BaseScoringFunction):
    r"""Log loss.

    The smaller the better, minimum is zero.

    The log loss is a strictly consistent scoring function for the mean for
    observations and predictions in the range 0 to 1.
    It is also referred to as (half the) Bernoulli deviance,
    (half the) Binomial log-likelihood, logistic loss and binary cross-entropy.
    Its minimal function value is zero.

    Attributes
    ----------
    functional: str
        "mean"

    Notes
    -----
    The log loss for \(y,z \in [0,1]\) is given by

    \[
    S(y, z) = - y \log\frac{z}{y} - (1 - y) \log\frac{1-z}{1-y}
    \]

    If one restricts to \(y\in \{0, 1\}\), this simplifies to

    \[
    S(y, z) = - y \log(z) - (1 - y) \log(1-z)
    \]

    Examples
    --------
    >>> ll = LogLoss()
    >>> ll(y_obs=[0, 0.5, 1, 1], y_pred=[0.1, 0.2, 0.8 , 0.9], weights=[1, 2, 1, 1])  # doctest: +SKIP
    0.17603033705165635
    """  # noqa: E501

    # FIXME: numpy 2.0.0, doctest skip should not be necessary.

    @property
    def functional(self):
        return "mean"

    def score_per_obs(
        self,
        y_obs: npt.ArrayLike,
        y_pred: npt.ArrayLike,
    ) -> np.ndarray:
        """Score per observation.

        Parameters
        ----------
        y_obs : array-like of shape (n_obs)
            Observed values of the response variable.
        y_pred : array-like of shape (n_obs)
            Predicted values of the `functional` of interest, e.g. the conditional
            expectation of the response, `E(Y|X)`.

        Returns
        -------
        score_per_obs : ndarray
            Values of the scoring function for each observation.
        """
        y: np.ndarray
        z: np.ndarray
        y, z = validate_2_arrays(y_obs, y_pred)

        score = -special.xlogy(y, z) - special.xlogy(1 - y, 1 - z)
        if np.any((y > 0) & (y < 1)):
            score += special.xlogy(y, y) + special.xlogy(1 - y, 1 - y)
        return score


class HomogeneousQuantileScore(_BaseScoringFunction):
    r"""Homogeneous scoring function of degree h for quantiles.

    The smaller the better, minimum is zero.

    Up to a multiplicative constant, these are the only scoring funtions that are
    strictly consistent for quantiles at level alpha and homogeneous functions.
    The possible additive constant is chosen such that the minimal function value
    equals zero.

    Note that the 1/2-quantile (level alpha=0.5) equals the median.

    Parameters
    ----------
    degree : float
        Degree of homogeneity.
    level : float
        The level of the quantile. (Often called \(\alpha\).)
        It must be `0 < level < 1`.
        `level=0.5` gives the median.

    Attributes
    ----------
    functional: str
        "quantile"

    Notes
    -----
    The homogeneous score of degree \(h\) is given by

    \[
    S_\alpha^h(y, z) = (\mathbf{1}\{z \ge y\} - \alpha) \frac{z^h - y^h}{h}
    \]

    There are important domain restrictions and limits:

    - \(h\) positive odd integer: All real numbers \(y\) and \(z\) are allowed.

        - Special case \(h=1\) equals the pinball loss,
          \(S(y, z) = (\mathbf{1}\{z \ge y\} - \alpha) (z - y)\).
        - Special case \(h=1, \alpha=\frac{1}{2}\) equals half the absolute error
          \(S(y, z) = \frac{1}{2}|z - y|\).

    - \(h\) real valued: Only \(y>0\), \(z>0\) are allowed.

        Special case \(h=0\) (by taking the limit) equals
        \(S(y, z) = |\mathbf{1}\{z \ge y\} - \alpha| \log\frac{z}{y}\).

    References
    ----------
    `[Gneiting2011]`

    :   T. Gneiting.
        "Making and Evaluating Point Forecasts". (2011)
        [doi:10.1198/jasa.2011.r10138](https://doi.org/10.1198/jasa.2011.r10138)
        [arxiv:0912.0902](https://arxiv.org/abs/0912.0902)

    Examples
    --------
    >>> hqs = HomogeneousQuantileScore(degree=3, level=0.1)
    >>> hqs(y_obs=[0, 0, 1, 1], y_pred=[-1, 1, 1 , 2])  # doctest: +SKIP
    0.6083333333333334
    """  # FIXME: numpy 2.0.0, doctest skip should not be necessary.

    def __init__(self, degree: float = 2, level: float = 0.5) -> None:
        self.degree = degree
        if level <= 0 or level >= 1:
            msg = f"Argument level must fulfil 0 < level < 1, got {level}."
            raise ValueError(msg)
        self.level = level

    @property
    def functional(self):
        return "quantile"

    def score_per_obs(
        self,
        y_obs: npt.ArrayLike,
        y_pred: npt.ArrayLike,
    ) -> np.ndarray:
        """Score per observation.

        Parameters
        ----------
        y_obs : array-like of shape (n_obs)
            Observed values of the response variable.
        y_pred : array-like of shape (n_obs)
            Predicted values of the `functional` of interest, e.g. the conditional
            expectation of the response, `E(Y|X)`.

        Returns
        -------
        score_per_obs : ndarray
            Values of the scoring function for each observation.
        """
        y: np.ndarray
        z: np.ndarray
        y, z = validate_2_arrays(y_obs, y_pred)

        if self.degree == 1:
            # Fast path
            score = z - y
        elif self.degree > 1 and self.degree % 2 == 1:
            # Odd positive degree
            score = (np.power(z, self.degree) - np.power(y, self.degree)) / self.degree
        elif self.degree == 0:
            # Domain: y > 0 and z > 0.
            if not np.all((y > 0) & (z > 0)):
                msg = (
                    f"Valid domain for degree={self.degree} is "
                    "y_obs > 0 and y_pred > 0."
                )
                raise ValueError(msg)
            score = np.log(z / y)
        else:
            # Domain: y > 0 and z > 0.
            if not np.all((y > 0) & (z > 0)):
                msg = (
                    f"Valid domain for degree={self.degree} is "
                    "y_obs > 0 and y_pred > 0."
                )
                raise ValueError(msg)
            score = (np.power(z, self.degree) - np.power(y, self.degree)) / self.degree

        if self.level == 0.5:
            return 0.5 * np.abs(score)
        else:
            return (np.greater_equal(z, y) - self.level) * score


class PinballLoss(HomogeneousQuantileScore):
    r"""Pinball loss.

    The smaller the better, minimum is zero.

    The pinball loss is strictly consistent for quantiles.

    Parameters
    ----------
    level : float
        The level of the quantile. (Often called \(\alpha\).)
        It must be `0 < level < 1`.
        `level=0.5` gives the median.

    Attributes
    ----------
    functional: str
        "quantile"

    Notes
    -----
    The pinball loss has degree of homogeneity 1 and is given by

    \[
    S_\alpha(y, z) = (\mathbf{1}\{z \ge y\} - \alpha) (z - y)
    \]

    The authors do not know where and when the term *pinball loss* was coined. It is
    most famously used in quantile regression.

    Examples
    --------
    >>> pl = PinballLoss(level=0.9)
    >>> pl(y_obs=[0, 0, 1, 1], y_pred=[-1, 1, 1 , 2])  # doctest: +SKIP
    0.275
    """  # FIXME: numpy 2.0.0, doctest skip should not be necessary.

    def __init__(self, level: float = 0.5) -> None:
        super().__init__(degree=1, level=level)


class ElementaryScore(_BaseScoringFunction):
    r"""Elementary scoring function.

    The smaller the better.

    The elementary scoring function is consistent for the specified `functional` for
    all values of `eta` and is the main ingredient for Murphy diagrams.
    See [Notes](#notes) for further details.

    Parameters
    ----------
    eta : float
        Free parameter.
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

    Notes
    -----
    [](){#notes}
    The elementary scoring or loss function is given by

    \[
    S_\eta(y, z) = (\mathbf{1}\{\eta \le z\} - \mathbf{1}\{\eta \le y\})
    V(y, \eta)
    \]

    with [identification functions]
    [model_diagnostics.calibration.identification_function]
    \(V\) for the given `functional` \(T\) . If allows for the mixture or Choquet
    representation

    \[
    S(y, z) = \int S_\eta(y, z) \,dH(\eta)
    \]

    for some locally finite measure \(H\). It follows that the scoring function \(S\)
    is consistent for \(T\).

    References
    ----------
    `[Jordan2022]`

    :   A.I. Jordan, A. Mühlemann, J.F. Ziegel.
        "Characterizing the optimal solutions to the isotonic regression problem for
        identifiable functionals". (2022)
        [doi:10.1007/s10463-021-00808-0](https://doi.org/10.1007/s10463-021-00808-0)

    `[GneitingResin2022]`

    :   T. Gneiting, J. Resin.
        "Regression Diagnostics meets Forecast Evaluation: Conditional Calibration,
        Reliability Diagrams, and Coefficient of Determination".
        [arxiv:2108.03210](https://arxiv.org/abs/2108.03210)

    Examples
    --------
    >>> el_score = ElementaryScore(eta=2, functional="mean")
    >>> el_score(y_obs=[1, 2, 2, 1], y_pred=[4, 1, 2, 3])  # doctest: +SKIP
    0.5
    """  # FIXME: numpy 2.0.0, doctest skip should not be necessary.

    def __init__(
        self, eta: float, functional: str = "mean", level: float = 0.5
    ) -> None:
        self.eta = eta
        self._functional = functional
        if level <= 0 or level >= 1:
            msg = f"Argument level must fulfil 0 < level < 1, got {level}."
            raise ValueError(msg)
        self.level = level

    @property
    def functional(self):
        return self._functional

    def score_per_obs(
        self,
        y_obs: npt.ArrayLike,
        y_pred: npt.ArrayLike,
    ) -> np.ndarray:
        """Score per observation.

        Parameters
        ----------
        y_obs : array-like of shape (n_obs)
            Observed values of the response variable.
        y_pred : array-like of shape (n_obs)
            Predicted values of the `functional` of interest, e.g. the conditional
            expectation of the response, `E(Y|X)`.

        Returns
        -------
        score_per_obs : ndarray
            Values of the scoring function for each observation.
        """
        y: np.ndarray
        z: np.ndarray
        y, z = validate_2_arrays(y_obs, y_pred)

        eta = self.eta
        eta_term = np.less_equal(eta, z).astype(float) - np.less_equal(eta, y)
        return eta_term * identification_function(
            y_obs=y_obs,
            y_pred=np.full(y.shape, fill_value=float(eta)),
            functional=self.functional,
            level=self.level,
        )


def decompose(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    weights: Optional[npt.ArrayLike] = None,
    *,
    scoring_function: Callable[..., Any],  # TODO: make type hint stricter
    functional: Optional[str] = None,
    level: Optional[float] = None,
) -> pl.DataFrame:
    r"""Additive decomposition of scores.

    The score is decomposed as
    `score = miscalibration - discrimination + uncertainty`.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
    y_pred : array-like of shape (n_obs) or (n_obs, n_models)
        Predicted values of the `functional` of interest, e.g. the conditional
        expectation of the response, `E(Y|X)`.
    weights : array-like of shape (n_obs) or None
        Case weights.
    scoring_function : callable
        A scoring function with signature roughly
        `fun(y_obs, y_pred, weights) -> float`.
    functional : str or None
        The target functional which `y_pred` aims to predict.
        If `None`, then it will be inferred from `scoring_function.functional`.
        Options are:

        - `"mean"`. Argument `level` is neglected.
        - `"median"`. Argument `level` is neglected.
        - `"expectile"`
        - `"quantile"`
    level : float or None
        Functionals like expectiles and quantiles have a level (often called alpha).
        If `None`, then it will be inferred from `scoring_function.level`.

    Returns
    -------
    decomposition : polars.DataFrame
        The resulting score decomposition as a dataframe with columns:

        - `miscalibration`
        - `discrimination`
        - `uncertainty`
        - `score`: the average score

    If `y_pred` contains several predictions, i.e. it is 2-dimension with shape
    `(n_obs, n_pred)` and `n_pred >1`, then there is the additional column:

        - `model`

    Notes
    -----
    To be precise, this function returns the decomposition of the score in terms of
    auto-miscalibration, auto-discrimination (or resolution) and uncertainy (or
    entropy), see `[FLM2022]` and references therein.
    The key element is to estimate the recalibrated predictions, i.e. \(T(Y|m(X))\) for
    the target functional \(T\) and model predictions \(m(X)\).
    This is accomplished by isotonic regression, `[Dimitriadis2021]` and
    `[Gneiting2021]`.

    References
    ----------
    `[FLM2022]`

    :   T. Fissler, C. Lorentzen, and M. Mayer.
        "Model Comparison and Calibration Assessment". (2022)
        [arxiv:2202.12780](https://arxiv.org/abs/2202.12780).

    `[Dimitriadis2021]`

    :   T. Dimitriadis, T. Gneiting, and A. I. Jordan.
        "Stable reliability diagrams for probabilistic classifiers". (2021)
        [doi:10.1073/pnas.2016191118](https://doi.org/10.1073/pnas.2016191118)

    `[Gneiting2021]`

    :   T. Gneiting and J. Resin.
        "Regression Diagnostics meets Forecast Evaluation: Conditional Calibration,
        Reliability Diagrams, and Coefficient of Determination". (2021).
        [arXiv:2108.03210](https://arxiv.org/abs/2108.03210).

    Examples
    --------
    >>> decompose(y_obs=[0, 0, 1, 1], y_pred=[-1, 1, 1, 2],
    ... scoring_function=SquaredError())
    shape: (1, 4)
    ┌────────────────┬────────────────┬─────────────┬───────┐
    │ miscalibration ┆ discrimination ┆ uncertainty ┆ score │
    │ ---            ┆ ---            ┆ ---         ┆ ---   │
    │ f64            ┆ f64            ┆ f64         ┆ f64   │
    ╞════════════════╪════════════════╪═════════════╪═══════╡
    │ 0.625          ┆ 0.125          ┆ 0.25        ┆ 0.75  │
    └────────────────┴────────────────┴─────────────┴───────┘
    """
    if functional is None:
        if hasattr(scoring_function, "functional"):
            functional = scoring_function.functional
        else:
            msg = (
                "You set functional=None, but scoring_function has no attribute "
                "functional."
            )
            raise ValueError(msg)
    if level is None:
        level = 0.5
        if functional in ("expectile", "quantile"):
            if hasattr(scoring_function, "level"):
                level = float(scoring_function.level)
            else:
                msg = (
                    "You set level=None, but scoring_function has no attribute "
                    "level."
                )
                raise ValueError(msg)

    allowed_functionals = ("mean", "median", "expectile", "quantile")
    if functional not in allowed_functionals:
        msg = (
            f"The functional must be one of {allowed_functionals}, got "
            f"{functional}."
        )
        raise ValueError(msg)
    if functional in ("expectile", "quantile") and (level <= 0 or level >= 1):
        msg = f"The level must fulfil 0 < level < 1, got {level}."
        raise ValueError(msg)

    validate_same_first_dimension(y_obs, y_pred)
    n_pred = length_of_second_dimension(y_pred)
    pred_names, _ = get_sorted_array_names(y_pred)
    y_o = np.asarray(y_obs)

    if weights is None:
        w = None
    else:
        validate_same_first_dimension(weights, y_o)
        w = np.asarray(weights)  # needed to satisfy mypy
        if w.ndim > 1:
            msg = f"The array weights must be 1-dimensional, got weights.ndim={w.ndim}."
            raise ValueError(msg)

    if functional == "mean":
        iso = IsotonicRegression_skl(y_min=None, y_max=None)
        marginal = np.average(y_o, weights=w)
    else:
        iso = IsotonicRegression(functional=functional, level=level)
        if functional == "expectile":
            marginal = expectile(y_o, alpha=level, weights=w)
        elif functional == "quantile":
            marginal = 0.5 * (
                quantile_lower(y_o, level=level) + quantile_upper(y_o, level=level)
            )

    if y_o[0] == marginal == y_o[-1]:
        # y_o is constant. We need to check if y_o is allowed as argument to y_pred.
        # For instance for the poisson deviance, y_o = 0 is allowed. But 0 is forbidden
        # as a prediction.
        try:
            scoring_function(y_o[0], marginal)
        except ValueError as exc:
            msg = (
                "Your y_obs is constant and lies outside the allowed range of y_pred "
                "of your scoring function. Therefore, the score decomposition cannot "
                "be applied."
            )
            raise ValueError(msg) from exc

    # The recalibrated versions, further down, could contain min(y_obs) and that could
    # be outside of the valid domain, e.g. y_pred = 0 for the Poisson deviance where
    # y_obs=0 is allowed. We detect that here:
    y_min = np.amin(y_o)
    y_min_allowed = True
    try:
        scoring_function(y_o[:1], np.array([y_min]), None if w is None else w[:1])
    except ValueError:
        y_min_allowed = False

    marginal = np.full_like(y_o, fill_value=marginal, dtype=float)
    score_marginal = scoring_function(y_o, marginal, w)

    df_list = []
    for i in range(len(pred_names)):
        # Loop over columns of y_pred.
        x = y_pred if n_pred == 0 else get_second_dimension(y_pred, i)
        iso.fit(x, y_o, sample_weight=w)
        recalibrated = np.squeeze(iso.predict(x))
        if not y_min_allowed and recalibrated[0] <= y_min:
            # Oh dear, this needs quite some extra work:
            # First index of value greater than y_min
            idx1 = np.argmax(recalibrated > y_min)
            val1 = recalibrated[idx1]
            # First index of value greater than the value at idx1.
            idx2 = np.argmax(recalibrated > val1)
            # Note that val1 may already be the largest value of the array => idx2 = 0.
            if idx2 == 0:
                idx2 = recalibrated.shape[0]
            # We merge the first 2 blocks of the isotonic regression as it violates
            # our domain requirements.
            re2 = recalibrated[:idx2]
            w2 = None if w is None else w[:idx2]
            if functional == "mean":
                recalibrated[:idx2] = np.average(re2, weights=w2)
            elif functional == "expectile":
                recalibrated[:idx2] = expectile(re2, alpha=level, weights=w2)
            elif functional == "quantile":
                # Note, no scoring function known that could end up here.
                lower = quantile_lower(re2, level=level)
                upper = quantile_upper(re2, level=level)
                recalibrated[:idx2] = 0.5 * (lower + upper)

        score = scoring_function(y_o, x, w)
        try:
            score_recalibrated = scoring_function(y_o, recalibrated, w)
        except ValueError as exc:
            msg = (
                "The recalibrated predictions obtained from isotonic regression are "
                "very likely outside the allowed range of y_pred of your scoring "
                "function. Therefore, the score decomposition cannot be applied."
            )
            raise ValueError(msg) from exc

        df = pl.DataFrame(
            {
                "model": pred_names[i],
                "miscalibration": score - score_recalibrated,
                "discrimination": score_marginal - score_recalibrated,
                "uncertainty": score_marginal,
                "score": score,
            }
        )
        df_list.append(df)

    df = pl.concat(df_list)

    # Remove column "model" for a single model.
    if n_pred <= 1:
        df = df.drop("model")

    return df
