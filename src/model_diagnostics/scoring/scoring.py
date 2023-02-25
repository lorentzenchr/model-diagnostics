from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from scipy import special

from .._utils.array import validate_2_arrays


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
        """Mean or average score."""
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
        It must be `0 <= level <= 1`.
        `level=0.5` and `functional="expectile"` gives the mean.

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
        if level < 0 or level > 1:
            raise ValueError(
                f"Argument level must fulfil 0 <= level <= 1, got {level}."
            )
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
        """Score per observation."""
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
            # Should error when y<0 or x<=0.
            score = 2 * (special.xlogy(y, y / z) - y + z)
        elif self.degree == 0:
            # Should error when y<0 or x<0.
            y_z = y / z
            score = 2 * (y_z - np.log(y_z) - 1)
        else:  # self.degree < 1
            # Should error when y<0 or x<0.
            score = 2 * (
                (np.power(y, self.degree) - np.power(z, self.degree))
                / (self.degree * (self.degree - 1))
                - 1 / (self.degree - 1) * np.power(z, self.degree - 1) * (y - z)
            )

        if self.level == 0.5:
            return score
        else:
            return 2 * np.abs(np.greater_equal(z, y) - self.level) * score


class SquaredError(HomogeneousExpectileScore):
    """Squared error."""

    def __init__(self) -> None:
        super().__init__(degree=2, level=0.5)


class PoissonDeviance(HomogeneousExpectileScore):
    """Squared error."""

    def __init__(self) -> None:
        super().__init__(degree=1, level=0.5)


class GammaDeviance(HomogeneousExpectileScore):
    """Squared error."""

    def __init__(self) -> None:
        super().__init__(degree=0, level=0.5)
