from inspect import isclass

import numpy as np
import pytest

from model_diagnostics.scoring import (
    GammaDeviance,
    HomogeneousExpectileScore,
    PoissonDeviance,
    SquaredError,
)

SCORES = [
    HomogeneousExpectileScore(degree=-1, level=0.2),
    HomogeneousExpectileScore(degree=-1, level=0.5),
    HomogeneousExpectileScore(degree=-1, level=0.99),
    HomogeneousExpectileScore(degree=0, level=0.2),
    HomogeneousExpectileScore(degree=0, level=0.5),
    HomogeneousExpectileScore(degree=0, level=0.99),
    HomogeneousExpectileScore(degree=0.5, level=0.2),
    HomogeneousExpectileScore(degree=0.5, level=0.5),
    HomogeneousExpectileScore(degree=0.5, level=0.99),
    HomogeneousExpectileScore(degree=1, level=0.2),
    HomogeneousExpectileScore(degree=1, level=0.5),
    HomogeneousExpectileScore(degree=1, level=0.99),
    HomogeneousExpectileScore(degree=1.5, level=0.2),
    HomogeneousExpectileScore(degree=1.5, level=0.5),
    HomogeneousExpectileScore(degree=1.5, level=0.99),
    HomogeneousExpectileScore(degree=2, level=0.2),
    HomogeneousExpectileScore(degree=2, level=0.5),
    HomogeneousExpectileScore(degree=2, level=0.99),
    HomogeneousExpectileScore(degree=2.5, level=0.2),
    HomogeneousExpectileScore(degree=2.5, level=0.5),
    HomogeneousExpectileScore(degree=2.5, level=0.99),
]


def sf_name(sf):
    """Return name of a scoring function for id in pytest."""
    if isinstance(sf, HomogeneousExpectileScore):
        return f"HomogeneousExpectileScore(degree={sf.degree}, level={sf.level})"
    elif isclass(sf):
        return sf.__class__.__name__
    else:
        return sf.__name__


@pytest.mark.parametrize(
    "sf, level, msg",
    [
        (HomogeneousExpectileScore, -1, "Argument level must fulfil 0 <= level <= 1"),
        (HomogeneousExpectileScore, 1.1, "Argument level must fulfil 0 <= level <= 1"),
    ],
)
def test_scoring_function_raises(sf, level, msg):
    """Test that scoring function raises error for invalid input."""
    with pytest.raises(ValueError, match=msg):
        sf(level=level)


@pytest.mark.parametrize("sf", SCORES, ids=sf_name)
def test_scoring_function_equal_input(sf):
    """Test that S(x, x) is zero."""
    rng = np.random.default_rng(112358132134)
    n = 5
    y = np.abs(rng.normal(loc=-2, scale=2, size=n))  # common domain
    assert sf(y_obs=y, y_pred=y) == pytest.approx(0)


@pytest.mark.parametrize("sf", SCORES, ids=sf_name)
@pytest.mark.parametrize("weights", [None, True])
def test_scoring_function_score_per_obs(sf, weights):
    """Test equivalence of __call__ and average(score_per_obs)."""
    rng = np.random.default_rng(112358132134)
    n = 5
    y_obs = np.abs(rng.normal(loc=-2, scale=2, size=n))  # common domain
    y_pred = np.abs(rng.normal(loc=-2, scale=2, size=n))  # common domain
    if weights is not None:
        weights = np.abs(rng.normal(loc=-2, scale=2, size=n))

    assert np.average(
        sf.score_per_obs(y_obs=y_obs, y_pred=y_pred), weights=weights
    ) == pytest.approx(sf(y_obs=y_obs, y_pred=y_pred, weights=weights))


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_homogeneous_scoring_function_close_to_0_1_2(degree):
    """Test scoring function close to degree = 0, 1 and 2."""
    rng = np.random.default_rng(112358132134)
    n = 5
    y_obs = np.abs(rng.normal(loc=-2, scale=2, size=n))  # common domain
    y_pred = np.abs(rng.normal(loc=-2, scale=2, size=n))  # common domain

    sf_exact = HomogeneousExpectileScore(degree=degree)(y_obs=y_obs, y_pred=y_pred)
    sf_below = HomogeneousExpectileScore(degree=degree - 1e-10)(
        y_obs=y_obs, y_pred=y_pred
    )
    sf_above = HomogeneousExpectileScore(degree=degree + 1e-10)(
        y_obs=y_obs, y_pred=y_pred
    )

    assert sf_below == pytest.approx(sf_exact, rel=1e-5)
    assert sf_above == pytest.approx(sf_exact, rel=1e-5)


def test_homogeneous_scoring_function_against_precomputed_values():
    """Test scoring function close to degree = 0, 1 and 2."""
    y_obs = [1, 2]
    y_pred = [4, 1]

    sf = HomogeneousExpectileScore(degree=2, level=0.5)
    se = SquaredError()
    precomputed = (9 + 1) / 2
    assert sf(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)
    assert se(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)

    sf = HomogeneousExpectileScore(degree=1, level=0.5)
    pd = PoissonDeviance()
    precomputed = (-2 * np.log(4) + 6 + 4 * np.log(2) - 2) / 2
    assert sf(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)
    assert pd(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)

    sf = HomogeneousExpectileScore(degree=0, level=0.5)
    gd = GammaDeviance()
    precomputed = (1 / 2 + 2 * np.log(4) - 2 + 4 - 2 * np.log(2) - 2) / 2
    assert sf(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)
    assert gd(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)


def test_scoring_function_functional():
    """Test that HomogeneousExpectileScore returns the right functional."""
    sf = HomogeneousExpectileScore(level=0.5)
    assert sf.functional == "mean"
    sf = HomogeneousExpectileScore(level=0.5001)
    assert sf.functional == "expectile"
