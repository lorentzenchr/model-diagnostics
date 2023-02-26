from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt
import polars as pl
from scipy import special
from sklearn.isotonic import IsotonicRegression

from .._utils.array import validate_2_arrays, validate_same_first_dimension


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
        pass


class HomogeneousExpectileScore(_BaseScoringFunction):
    r"""Homogeneous scoring function of degree h for expectiles.

    The smaller the better.

    Up to a multiplicative constant, these are the only scoring funtions that are
    strictly consistent for expectiles at level ɑ and homogeneous functions.
    The possible additive constant is chosen such that the minimal function value
    equals zero.

    Note that the 1/2-expectile (level ɑ=0.5) equals the mean.

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
    following relation between the degree of homogeneity \(h\) and the with Tweedie
    power \(p\): \(h = 2-p\).

    References
    ----------
    `[Gneiting2011]`

    :   T. Gneiting.
        "Making and Evaluating Point Forecasts”. (2011)
        [doi:10.1198/jasa.2011.r10138](https://doi.org/10.1198/jasa.2011.r10138)
        [arxiv:0912.0902](https://arxiv.org/abs/0912.0902)
    """

    def __init__(self, degree: float = 2, level: float = 0.5) -> None:
        self.degree = degree
        if level <= 0 or level >= 1:
            raise ValueError(f"Argument level must fulfil 0 < level < 1, got {level}.")
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
                raise ValueError(
                    f"Valid domain for degree={self.degree} is "
                    "y_obs >= 0 and y_pred > 0."
                )
            score = 2 * (special.xlogy(y, y / z) - y + z)
        elif self.degree == 0:
            # Domain: y > 0 and z > 0.
            if not np.all((y > 0) & (z > 0)):
                raise ValueError(
                    f"Valid domain for degree={self.degree} is "
                    "y_obs > 0 and y_pred > 0."
                )
            y_z = y / z
            score = 2 * (y_z - np.log(y_z) - 1)
        else:  # self.degree < 1
            # Domain: y >= 0 and z > 0 for 0 < self.degree < 1
            # Domain: y > 0  and z > 0 else
            if self.degree > 0:
                if not np.all((y >= 0) & (z > 0)):
                    raise ValueError(
                        f"Valid domain for degree={self.degree} is "
                        "y_obs >= 0 and y_pred > 0."
                    )
            else:
                if not np.all((y > 0) & (z > 0)):
                    raise ValueError(
                        f"Valid domain for degree={self.degree} is "
                        "y_obs > 0 and y_pred > 0."
                    )
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
    """

    def __init__(self) -> None:
        super().__init__(degree=2, level=0.5)


class PoissonDeviance(HomogeneousExpectileScore):
    r"""Poisson deviance.

    The Poisson deviance is strictly consistent for the mean.
    It has a degree of homogeneity of 1.

    Attributes
    ----------
    functional: str
        "mean"

    Notes
    -----
    \(S(y, z) = 2(y\log\frac{y}{z} - y + z)\)
    """

    def __init__(self) -> None:
        super().__init__(degree=1, level=0.5)


class GammaDeviance(HomogeneousExpectileScore):
    r"""Gamma deviance.

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
    """

    def __init__(self) -> None:
        super().__init__(degree=0, level=0.5)


class LogLoss(_BaseScoringFunction):
    r"""Log loss.

    The log loss is a strictly consistent scoring function for the mean for the
    observations and predictions in the range 0 to 1.
    It is also referred to as Bernoulli deviance, Binomial log-likelihood, logistic
    loss and binary cross-entropy.
    It's minimal function value is zero.

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
    """

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
        if np.any((0 < y) & (y < 1)):
            score += special.xlogy(y, y) + special.xlogy(1 - y, 1 - y)
        return score


class HomogeneousQuantileScore(_BaseScoringFunction):
    r"""Homogeneous scoring function of degree h for quantiles.

    The smaller the better.

    Up to a multiplicative constant, these are the only scoring funtions that are
    strictly consistent for quantiles at level ɑ and homogeneous functions.
    The possible additive constant is chosen such that the minimal function value
    equals zero.

    Note that the 1/2-quantile (level ɑ=0.5) equals the median.

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
        "Making and Evaluating Point Forecasts”. (2011)
        [doi:10.1198/jasa.2011.r10138](https://doi.org/10.1198/jasa.2011.r10138)
        [arxiv:0912.0902](https://arxiv.org/abs/0912.0902)
    """

    def __init__(self, degree: float = 2, level: float = 0.5) -> None:
        self.degree = degree
        if level <= 0 or level >= 1:
            raise ValueError(f"Argument level must fulfil 0 < level < 1, got {level}.")
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
                raise ValueError(
                    f"Valid domain for degree={self.degree} is "
                    "y_obs > 0 and y_pred > 0."
                )
            score = np.log(z / y)
        else:
            # Domain: y > 0 and z > 0.
            if not np.all((y > 0) & (z > 0)):
                raise ValueError(
                    f"Valid domain for degree={self.degree} is "
                    "y_obs > 0 and y_pred > 0."
                )
            score = (np.power(z, self.degree) - np.power(y, self.degree)) / self.degree

        if self.level == 0.5:
            return 0.5 * np.abs(score)
        else:
            return (np.greater_equal(z, y) - self.level) * score


class PinballLoss(HomogeneousQuantileScore):
    r"""Pinball loss.

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
    """

    def __init__(self, level: float = 0.5) -> None:
        super().__init__(degree=1, level=level)


def decompose(
    scoring_function: Callable[..., Any],  # TODO: make type hint stricter
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    weights: Optional[npt.ArrayLike] = None,
    functional: Optional[str] = None,
    level: Optional[float] = None,
) -> pl.DataFrame:
    r"""Additive decomposition of scores.

    The score is decomposed as
    `score = miscalibration - discrimination + uncertainty`.

    Parameters
    ----------
    scoring_function : callable
        A scoring function with signature roughly
        `fun(y_obs, y_pred, weights) -> float`.
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
    y_pred : array-like of shape (n_obs)
        Predicted values of the `functional` of interest, e.g. the conditional
        expectation of the response, `E(Y|X)`.
    weights : array-like of shape (n_obs) or None
        Case weights.
    functional : str or None
        The target functionl which `y_pred` aims to predict.
        If `None`, then it will be inferred from `scoring_function.functional`.
    level : float or None
        Functionals like expectiles and quantiles have a level (often called ɑ).
        If `None`, then it will be inferred from `scoring_function.level`.

    Returns
    -------
    decomposition : polars.DataFrame
        The resulting score decomposition as a dataframe with columns:

        - `miscalibration`
        - `discrimination`
        - `uncertainty`
        - `score`: the average score

    Notes
    -----
    To be precise, this function returns the decomposition of the score in terms of
    auto-miscalibration, auto-discrimination (or resolution) and uncertainy (or
    entropy), see `[FLM2022]` and references therein.
    The key element is to esimate the recalibrated predictions, i.e. \(T(Y|m(X))\) for
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

    """
    if functional is None:
        if hasattr(scoring_function, "functional"):
            functional = scoring_function.functional
        else:
            raise ValueError(
                "You set functional=None, but scoring_function has no attribute "
                "functional."
            )
    if level is None:
        if functional == "mean":
            level = 0.5
        elif functional in ("expectile", "quantile"):
            if hasattr(scoring_function, "level"):
                level = scoring_function.level
            else:
                raise ValueError(
                    "You set level=None, but scoring_function has no attribute "
                    "level."
                )
    if not (functional == "mean" or (functional == "expectile" and level == 0.5)):
        raise ValueError(
            f"The given {functional=} and {level=} are not supported (yet)."
        )
    y: np.ndarray
    z: np.ndarray
    y, z = validate_2_arrays(y_obs, y_pred)
    if weights is None:
        w = None
    else:
        validate_same_first_dimension(weights, y)
        w = np.asarray(weights)  # needed to satisfy mypy

    iso = IsotonicRegression(y_min=None, y_max=None).fit(z, y, sample_weight=w)
    recalibrated = np.squeeze(iso.predict(z))
    marginal = np.full(shape=y.shape, fill_value=np.average(y, weights=w))

    score = scoring_function(y, z, w)
    score_recalibrated = scoring_function(y, recalibrated, w)
    score_marginal = scoring_function(y, marginal, w)

    df = pl.DataFrame(
        {
            "miscalibration": score - score_recalibrated,
            "discrimination": score_marginal - score_recalibrated,
            "uncertainty": score_marginal,
            "score": score,
        }
    )

    return df
