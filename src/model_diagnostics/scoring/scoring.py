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
        """Score per observation."""
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
