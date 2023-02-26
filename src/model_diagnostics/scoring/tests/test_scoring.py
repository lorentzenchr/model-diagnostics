from inspect import isclass

import numpy as np
import polars as pl
import pytest
import scipy.stats
from numpy.testing import assert_allclose
from polars.testing import assert_frame_equal

from model_diagnostics.scoring import (
    GammaDeviance,
    HomogeneousExpectileScore,
    HomogeneousQuantileScore,
    LogLoss,
    PinballLoss,
    PoissonDeviance,
    SquaredError,
    decompose,
)

HES = HomogeneousExpectileScore
HQS = HomogeneousQuantileScore

SCORES = [
    HES(degree=-1, level=0.2),
    HES(degree=-1, level=0.5),
    HES(degree=-1, level=0.99),
    HES(degree=0, level=0.2),
    HES(degree=0, level=0.5),
    HES(degree=0, level=0.99),
    HES(degree=0.5, level=0.2),
    HES(degree=0.5, level=0.5),
    HES(degree=0.5, level=0.99),
    HES(degree=1, level=0.2),
    HES(degree=1, level=0.5),
    HES(degree=1, level=0.99),
    HES(degree=1.5, level=0.2),
    HES(degree=1.5, level=0.5),
    HES(degree=1.5, level=0.99),
    HES(degree=2, level=0.2),
    HES(degree=2, level=0.5),
    HES(degree=2, level=0.99),
    HES(degree=2.5, level=0.2),
    HES(degree=2.5, level=0.5),
    HES(degree=2.5, level=0.99),
    HQS(degree=-1, level=0.2),
    HQS(degree=-1, level=0.5),
    HQS(degree=-1, level=0.99),
    HQS(degree=0, level=0.2),
    HQS(degree=0, level=0.5),
    HQS(degree=0, level=0.99),
    HQS(degree=0.5, level=0.2),
    HQS(degree=0.5, level=0.5),
    HQS(degree=0.5, level=0.99),
    HQS(degree=1, level=0.2),
    HQS(degree=1, level=0.5),
    HQS(degree=1, level=0.99),
    HQS(degree=1.5, level=0.2),
    HQS(degree=1.5, level=0.5),
    HQS(degree=1.5, level=0.99),
    HQS(degree=2, level=0.2),
    HQS(degree=2, level=0.5),
    HQS(degree=2, level=0.99),
    HQS(degree=3, level=0.2),
    HQS(degree=3, level=0.5),
    HQS(degree=3, level=0.99),
    LogLoss(),
]


def sf_name(sf):
    """Return name of a scoring function for id in pytest."""
    if isinstance(sf, HomogeneousExpectileScore):
        return f"HomogeneousExpectileScore(degree={sf.degree}, level={sf.level})"
    elif isclass(sf):
        return sf.__class__.__name__
    elif hasattr(sf, "__name__"):
        return sf.__name__
    else:
        return sf


@pytest.mark.parametrize("sf", [HES, HQS])
@pytest.mark.parametrize(
    "level, msg",
    [
        (-1, "Argument level must fulfil 0 < level < 1"),
        (0, "Argument level must fulfil 0 < level < 1"),
        (1, "Argument level must fulfil 0 < level < 1"),
        (1.1, "Argument level must fulfil 0 < level < 1"),
    ],
)
def test_scoring_function_raises(sf, level, msg):
    """Test that scoring function raises error for invalid input."""
    with pytest.raises(ValueError, match=msg):
        sf(level=level)


@pytest.mark.parametrize(
    "sf, y_obs, y_pred, msg",
    [
        (HES(degree=1.1), -1, -2, None),
        (HES(degree=1), 0, 1, None),
        (HES(degree=1), 1, 0, "Valid domain .* y_obs >= 0 and y_pred > 0"),
        (HES(degree=0.5), 0, 1, None),
        (HES(degree=0.5), 1, 0, "Valid domain .* y_obs >= 0 and y_pred > 0"),
        (HES(degree=0), 1, 1, None),
        (HES(degree=0), 0, 1, "Valid domain .* y_obs > 0 and y_pred > 0"),
        (HES(degree=0), 1, 0, "Valid domain .* y_obs > 0 and y_pred > 0"),
        (HES(degree=-1), 1, 1, None),
        (HES(degree=-1), 0, 1, "Valid domain .* y_obs > 0 and y_pred > 0"),
        (HES(degree=-2), 1, 0, "Valid domain .* y_obs > 0 and y_pred > 0"),
        (HQS(degree=3), -1, -1, None),
        (HQS(degree=1), -1, -1, None),
        (HQS(degree=1.1), 0, 1, "Valid domain .* y_obs > 0 and y_pred > 0"),
        (HQS(degree=1.1), 1, 0, "Valid domain .* y_obs > 0 and y_pred > 0"),
        (HQS(degree=0), 0, 1, "Valid domain .* y_obs > 0 and y_pred > 0"),
        (HQS(degree=0), 1, 0, "Valid domain .* y_obs > 0 and y_pred > 0"),
    ],
)
def test_scoring_function_domain(sf, y_obs, y_pred, msg):
    """Test the valid domains of scoring functions."""
    if msg is None:
        # no error
        sf(y_obs=y_obs, y_pred=y_pred)
    else:
        with pytest.raises(ValueError, match=msg):
            sf(y_obs=y_obs, y_pred=y_pred)


@pytest.mark.parametrize("sf", SCORES, ids=sf_name)
def test_scoring_function_for_equal_input(sf):
    """Test that S(x, x) is zero."""
    rng = np.random.default_rng(112358132134)
    n = 5
    y = np.abs(rng.normal(loc=-2, scale=2, size=n))  # common domain
    if isinstance(sf, LogLoss):
        y /= np.amax(y)
    assert sf(y_obs=y, y_pred=y) == pytest.approx(0)


@pytest.mark.parametrize("sf", SCORES, ids=sf_name)
@pytest.mark.parametrize("weights", [None, True])
def test_scoring_function_score_per_obs(sf, weights):
    """Test equivalence of __call__ and average(score_per_obs)."""
    rng = np.random.default_rng(112358132134)
    n = 5
    y_obs = np.abs(rng.normal(loc=-2, scale=2, size=n))  # common domain
    y_pred = np.abs(rng.normal(loc=-2, scale=2, size=n))  # common domain
    if isinstance(sf, LogLoss):
        y_obs /= np.amax(y_obs)
        y_pred /= np.amax(y_pred)
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

    sf = HomogeneousQuantileScore(degree=1, level=0.5)
    pl = PinballLoss(level=0.5)
    precomputed = (3 / 2 + 1 / 2) / 2
    assert sf(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)
    assert pl(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)

    sf = HomogeneousQuantileScore(degree=2, level=1 / 4)
    precomputed = (3 / 4 * 15 / 2 + 1 / 4 * 3 / 2) / 2
    assert sf(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)


def test_log_loss_against_precomputed_values():
    """Test scoring function close to degree = 0, 1 and 2."""
    log05 = np.log(0.5)
    log15 = np.log(1.5)
    y_obs = [0, 0, 0.5, 0.5, 1, 1]
    y_pred = [0, 0.5, 0.5, 0.25, 0.5, 1]
    sf = LogLoss()
    precomputed = [0, -log05, 0, -0.5 * log05 - 0.5 * log15, -log05, 0]
    assert_allclose(sf.score_per_obs(y_obs=y_obs, y_pred=y_pred), precomputed)

    # bernoulli.logpmf can only handle y_obs in {0, 1}
    y_obs = [0, 0, 0, 1, 1, 1]
    precomputed = -scipy.stats.bernoulli.logpmf(y_obs, y_pred)
    assert_allclose(sf.score_per_obs(y_obs=y_obs, y_pred=y_pred), precomputed)


def test_scoring_function_functional():
    """Test that scoring function returns the right functional."""
    sf = HomogeneousExpectileScore(level=0.5)
    assert sf.functional == "mean"
    sf = HomogeneousExpectileScore(level=0.5001)
    assert sf.functional == "expectile"
    sf = HomogeneousQuantileScore()
    assert sf.functional == "quantile"


@pytest.mark.parametrize(
    "sf, functional, level, msg",
    [
        (SquaredError(), None, None, None),
        (PinballLoss(), None, None, "The given functional.* not supported \\(yet\\)."),
        (
            lambda y, z, w: np.mean((y - z) ** 2),
            None,
            None,
            "You set functional=None, but scoring_function has no attribute "
            "functional.",
        ),
        (
            lambda y, z, w: np.mean((y - z) ** 2),
            "expectile",
            None,
            "You set level=None, but scoring_function has no attribute " "level.",
        ),
        (lambda y, z, w: np.mean((y - z) ** 2), "mean", None, None),
    ],
)
def test_decompose_raises(sf, functional, level, msg):
    """Test that decompose raises errors."""
    y_obs = [0, 1]
    y_pred = [0.5, 0.5]
    if msg is None:
        # no error
        decompose(
            scoring_function=sf,
            y_obs=y_obs,
            y_pred=y_pred,
            functional=functional,
            level=level,
        )
    else:
        with pytest.raises(ValueError, match=msg):
            decompose(
                scoring_function=sf,
                y_obs=y_obs,
                y_pred=y_pred,
                functional=functional,
                level=level,
            )


def test_decompose_with_numbers():
    """Test decompose against R library reliabilitydiag"""
    # library(reliabilitydiag)
    # y <- c(0, 0, 0, 1, 1, 1)
    # z <- c(0.4, 0.3, 0.2, 0.1, 0.5, 0.9)
    # reldiag <- reliabilitydiag(pred=z, y=y)
    # summary(reldiag, score = "brier")
    #
    # 'brier' score decomposition (see also ?summary.reliabilitydiag)
    # # A tibble: 1 × 5
    #   forecast mean_score miscalibration discrimination uncertainty
    #   <chr>         <dbl>          <dbl>          <dbl>       <dbl>
    # 1 pred          0.227          0.102          0.125        0.25
    y = [0, 0, 0, 1, 1, 1]
    z = [0.4, 0.3, 0.2, 0.1, 0.5, 0.9]
    df = decompose(
        scoring_function=SquaredError(),
        y_obs=y,
        y_pred=z,
    )
    assert (df["miscalibration"] - df["discrimination"] + df["uncertainty"])[
        0
    ] == pytest.approx(df["score"][0])

    df_expected = pl.DataFrame(
        {
            "miscalibration": 0.1016667,
            "discrimination": 0.125,
            "uncertainty": 0.25,
            "score": 0.2266667,
        }
    )
    assert_frame_equal(df, df_expected, check_exact=False)

    # log_loss_score <- function(obs, pred){
    #   ifelse(((obs==0 & pred==0) | (obs==1 & pred==1)),
    #          0,
    #          -obs * log(pred) - (1 - obs) * log(1 - pred))
    # }
    # summary(reldiag, score = "log_loss_score")
    # 'log_loss_score' score decomposition (see also ?summary.reliabilitydiag)
    # # A tibble: 1 × 5
    #   forecast mean_score miscalibration discrimination uncertainty
    #   <chr>         <dbl>          <dbl>          <dbl>       <dbl>
    # 1 pred          0.699          0.324          0.318       0.693
    df = decompose(
        scoring_function=LogLoss(),
        y_obs=y,
        y_pred=z,
    )
    assert (df["miscalibration"] - df["discrimination"] + df["uncertainty"])[
        0
    ] == pytest.approx(df["score"][0])

    df_expected = pl.DataFrame(
        {
            "miscalibration": 0.3237327,
            "discrimination": 0.3182571,
            "uncertainty": 0.6931472,
            "score": 0.6986228,
        }
    )
    assert_frame_equal(df, df_expected, check_exact=False)
