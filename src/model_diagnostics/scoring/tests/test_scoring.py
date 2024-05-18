from decimal import Decimal
from inspect import isclass

import numpy as np
import polars as pl
import pytest
import scipy.stats
from numpy.testing import assert_allclose
from polars.testing import assert_frame_equal

from model_diagnostics._utils.isotonic import gpava
from model_diagnostics.scoring import (
    ElementaryScore,
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
    ElementaryScore(eta=0, functional="mean"),
    ElementaryScore(eta=0, functional="expectile", level=0.2),
    ElementaryScore(eta=0, functional="quantile", level=0.8),
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


@pytest.mark.parametrize("sf", [HES, HQS, ElementaryScore])
@pytest.mark.parametrize(
    ("level", "msg"),
    [
        (-1, "Argument level must fulfil 0 < level < 1"),
        (0, "Argument level must fulfil 0 < level < 1"),
        (1, "Argument level must fulfil 0 < level < 1"),
        (1.1, "Argument level must fulfil 0 < level < 1"),
    ],
)
def test_scoring_function_raises(sf, level, msg):
    """Test that scoring function raises error for invalid input."""
    if sf is ElementaryScore:
        with pytest.raises(ValueError, match=msg):
            sf(eta=0, functional="quantile", level=level)
    else:
        with pytest.raises(ValueError, match=msg):
            sf(level=level)


@pytest.mark.parametrize(
    ("sf", "y_obs", "y_pred", "msg"),
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
    """Test scoring function for precomputed values."""
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
    """Test log loss for precomputed values."""
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


def test_elemantary_scoring_function_against_precomputed_values():
    """Test elementary scoring function for precomputed values."""
    y_obs = [1, 2]
    y_pred = [4, 1]

    for functional in ("mean", "expectile", "quantile"):
        sf = ElementaryScore(eta=0, functional=functional)
        precomputed = 0
        assert sf(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)

    sf = ElementaryScore(eta=2, functional="mean")
    precomputed = 7 - 2
    assert sf(y_obs=[7], y_pred=[-1]) == pytest.approx(precomputed)

    sf = ElementaryScore(eta=2, functional="mean")
    precomputed = (1 + 0) / 2
    assert sf(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)

    sf = ElementaryScore(eta=1.75, functional="mean")
    precomputed = (0.75 + 0.25) / 2
    assert sf(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)

    sf = ElementaryScore(eta=1.75, functional="expectile", level=0.2)
    precomputed = 2 * ((1 - 0.2) * 0.75 + 0.2 * 0.25) / 2
    assert sf(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)

    sf = ElementaryScore(eta=1.75, functional="quantile", level=0.2)
    precomputed = ((1 - 0.2) + 0.2) / 2
    assert sf(y_obs=y_obs, y_pred=y_pred) == pytest.approx(precomputed)


def test_scoring_function_functional():
    """Test that scoring function returns the right functional."""
    sf = HomogeneousExpectileScore(level=0.5)
    assert sf.functional == "mean"
    sf = HomogeneousExpectileScore(level=0.5001)
    assert sf.functional == "expectile"
    sf = HomogeneousQuantileScore()
    assert sf.functional == "quantile"


class MockSFFunctional:
    functional = "XXX"
    level = 0.2

    def __call__(self, y_obs, y_pred, weights=None):  # pragma: no cover
        return y_obs


class MockSFLevel:
    functional = "quantile"
    level = 99

    def __call__(self, y_obs, y_pred, weights=None):  # pragma: no cover
        return y_obs


@pytest.mark.parametrize(
    ("sf", "functional", "level", "msg"),
    [
        (SquaredError(), None, None, None),
        (HES(level=0.1), None, None, None),
        (HQS(degree=1, level=0.1), None, None, None),
        (PinballLoss(), None, 99, "The level must fulfil 0 < level < 1, got 99."),
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
            "You set level=None, but scoring_function has no attribute level.",
        ),
        (lambda y, z, w: np.mean(np.subtract(y, z) ** 2), "mean", None, None),
        (MockSFFunctional(), None, None, "The functional must be one of.*, got XXX."),
        (MockSFLevel(), None, None, "The level must fulfil 0 < level < 1, got 99"),
    ],
)
def test_decompose_raises(sf, functional, level, msg):
    """Test that decompose raises errors."""
    y_obs = [0, 1]
    y_pred = [0.5, 0.5]
    if msg is None:
        # no error
        decompose(
            y_obs=y_obs,
            y_pred=y_pred,
            scoring_function=sf,
            functional=functional,
            level=level,
        )
    else:
        with pytest.raises(ValueError, match=msg):
            decompose(
                y_obs=y_obs,
                y_pred=y_pred,
                scoring_function=sf,
                functional=functional,
                level=level,
            )


@pytest.mark.parametrize(
    ("weights", "msg"),
    [
        ("not an array", "The two array-like objects don't have the same length"),
        ([1], "The two array-like objects don't have the same length"),
        (
            np.arange(2)[:, None],
            "The array weights must be 1-dimensional, got weights.ndim=2.",
        ),
    ],
)
def test_decompose_raises_for_wrong_weights(weights, msg):
    with pytest.raises(ValueError, match=msg):
        decompose(
            y_obs=[0, 1],
            y_pred=[1, 2],
            weights=weights,
            scoring_function=SquaredError(),
        )


def test_decompose_with_numbers():
    """Test decompose against R library reliabilitydiag."""
    # library(reliabilitydiag)
    # y <- c(0, 0, 0, 1, 1, 1)
    # z <- c(0.4, 0.3, 0.2, 0.1, 0.5, 0.9)
    # reldiag <- reliabilitydiag(pred=z, y=y)
    # summary(reldiag, score = "brier")
    #
    # 'brier' score decomposition (see also ?summary.reliabilitydiag)
    # # A tibble: 1 x 5
    #   forecast mean_score miscalibration discrimination uncertainty
    #   <chr>         <dbl>          <dbl>          <dbl>       <dbl>
    # 1 pred          0.227          0.102          0.125        0.25
    y = [0, 0, 0, 1, 1, 1]
    z = [0.4, 0.3, 0.2, 0.1, 0.5, 0.9]
    df = decompose(
        y_obs=y,
        y_pred=z,
        scoring_function=SquaredError(),
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
    # # A tibble: 1 x 5
    #   forecast mean_score miscalibration discrimination uncertainty
    #   <chr>         <dbl>          <dbl>          <dbl>       <dbl>
    # 1 pred          0.699          0.324          0.318       0.693
    df = decompose(
        y_obs=y,
        y_pred=z,
        scoring_function=LogLoss(),
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


def test_decompose_vs_gneiting_resin():
    """Test vs http://arxiv.org/abs/2108.03210v3

    R code: https://github.com/resinj/replication_GR21

    Fig. 6 in particular, quantile level alpha = 0.1.
    """
    # fmt: off
    # R code: > sim = setup_normal(400,1.5,0.7)
    #         > sim$y
    y_obs = np.array([
        -0.817710400,  2.002249554,  0.197870769,  0.254481365,  1.035358806,  0.529264370, -2.496082086,  0.422352759, # noqa: E501
        -2.461628084, -0.971057494, -0.363566706,  1.453791050, -1.674714766,  0.842758189,  0.321573172, -0.468919247, # noqa: E501
        -1.133096880,  1.252763901, -0.438577221,  2.553553069, -0.173333780,  2.407409150,  1.248972867,  0.426499010, # noqa: E501
        -0.962345074, -0.712661322,  0.512179729, -0.176548926, -2.249364664,  0.832801582,  0.880003686,  2.441098831, # noqa: E501
         0.907607719,  0.053145468, -2.427334901,  0.137738506,  1.286988773,  1.132407141,  0.382507949,  1.762277858, # noqa: E501
         1.220943750,  3.408725520, -0.491263988,  0.739924816,  0.412682480,  2.655246800,  0.618093815,  1.975216906, # noqa: E501
         1.050738041, -3.489485619, -0.702955493,  0.169577379,  0.880414798,  1.211770463, -2.609553073, -0.040927491, # noqa: E501
         0.251803128,  0.072975572,  3.184791486, -2.236510640,  1.262683473,  2.176113620,  0.070843247,  2.730072958, # noqa: E501
        -0.177642166,  0.662788537,  2.190276413,  0.160011073, -1.068880368,  1.081429316,  0.578282695, -1.144855209, # noqa: E501
        -0.917425766,  1.181246099, -2.012322283,  3.104999574, -0.085743124, -0.195199986,  1.509199940, -3.208354459, # noqa: E501
        -0.400375973, -0.941457179, -2.002917039, -2.199654113, -1.119893337,  0.070475211,  0.915897597,  2.104048387, # noqa: E501
         0.369569653, -0.037288909,  0.005033704,  1.608700503, -2.777167643,  1.185087290,  0.282081246,  2.410548481, # noqa: E501
        -1.719364994,  0.882543045,  0.522190893,  0.497251861, -0.155424094,  0.859388286, -0.364116299, -0.336278039, # noqa: E501
        -0.621096204, -0.139129723, -1.757325452,  1.863379778,  0.654904815, -1.013925878, -0.723842435, -0.274318259, # noqa: E501
        -0.355339585, -1.649524321,  1.209447090, -1.116863584,  2.271682049, -0.498903015, -0.925281313,  0.601020746, # noqa: E501
        -0.825628439, -0.473708646, -1.514820781,  1.666801571,  0.254701733, -1.147153809, -0.471079677, -0.235351337, # noqa: E501
        -0.663297093,  0.727374389, -0.274365707, -1.105326043, -0.031467639, -0.314687833, -0.580133330, -0.343397403, # noqa: E501
         0.639287366, -0.289454477, -2.542766281, -1.104338009,  1.118454605, -0.270605801, -1.667624901,  1.224073343, # noqa: E501
        -1.343452196, -1.697951663,  0.171662244,  0.011678582,  0.074112143, -0.540063812,  0.733237716,  1.085981377, # noqa: E501
        -1.530748410,  1.163118712, -1.025608727, -1.012049958, -0.535632306, -1.355848530, -0.534791407,  1.276686948, # noqa: E501
         0.494256832,  1.644273048,  0.925566265, -0.187911494,  0.839267908, -2.198958793,  1.065191705,  0.843728491, # noqa: E501
         0.178410001,  0.236729568,  0.743710157,  0.584094919, -1.041790340,  1.577030859,  0.803371987, -1.059124491, # noqa: E501
         0.043000190, -0.340139560,  2.059345646,  2.508096948, -0.295118165,  1.158666643,  0.382436081, -0.751777945, # noqa: E501
         0.930617515, -0.899657188, -0.990608598,  0.560498926, -3.047877800,  0.323905909, -0.128952547, -1.295074145, # noqa: E501
         1.663415427, -0.090679180,  0.836806226, -1.110644474,  0.560870044,  2.498556454, -2.256902933, -1.564490359, # noqa: E501
         1.125821909,  0.824333095,  1.440137056,  1.589537826,  2.154855898, -0.261186424, -0.414525873,  0.053897313, # noqa: E501
        -3.444949308, -0.042681009,  0.189051649, -0.617033857, -0.806989749, -0.931629343, -2.863511214,  1.660964842, # noqa: E501
        -0.532910992, -1.683707009,  0.575839473, -0.515970598, -1.478750607,  0.038702059,  0.582970464,  0.046439820, # noqa: E501
        -0.161971562, -0.353880904, -0.200281739, -1.365336551,  1.875461307, -1.102243082,  1.669789311, -0.183536237, # noqa: E501
        -0.117691971, -2.470378909, -1.724840702, -1.093912989,  0.730594789,  0.691135688,  0.327229415,  1.474196280, # noqa: E501
         0.042887627,  3.533912886, -0.955072900, -1.271631253, -0.004966362,  3.154439869, -1.553272310, -1.295168477, # noqa: E501
        -0.024049049,  2.978390371, -0.807773233, -1.998523695,  0.065449287,  1.581631991,  2.041067918, -1.708421866, # noqa: E501
        -1.906487545,  2.192254964, -0.526278850, -0.523458290,  0.383795138,  0.879489141, -1.007957216, -2.684035374, # noqa: E501
         0.845789588,  0.754199975,  1.790972794, -1.721416030,  1.002796717,  0.147738413,  1.909530489,  0.102885342, # noqa: E501
         1.572724597,  2.456900157,  0.657986040, -1.674069402,  0.490695312,  1.193869257,  2.984215665, -1.164616209, # noqa: E501
         0.372844613,  0.725204678,  1.783842485,  1.082936359, -0.844700307, -2.179425155,  1.299885048,  0.348120471, # noqa: E501
         1.501868753, -1.159301031,  0.837834383, -1.993436775, -0.400060074,  0.374118263, -2.790699891,  0.280421214, # noqa: E501
        -1.357423254,  2.327765003,  1.028187483, -0.632561573, -1.672190775, -2.435959871, -0.290373767,  0.237561019, # noqa: E501
        -0.311244542, -0.828080236,  0.477386385,  2.844868524, -1.553997732, -0.460788129, -1.599716300, -3.112394780, # noqa: E501
         0.605174506,  0.096196433, -1.657728356,  1.509294729,  0.802816211,  0.496522687, -0.196377696, -0.031303774, # noqa: E501
        -2.238577932, -0.590459633,  1.716180492,  1.406513241,  1.217538921,  1.188670688, -0.067447093,  0.838818734, # noqa: E501
        -0.490337699,  0.177471931, -1.101634586,  0.270608383, -0.740660027,  0.018113692,  0.626025989, -2.181118136, # noqa: E501
        -1.222660883, -1.586734358, -3.947223934, -1.694461004,  1.914963275, -2.644162705, -0.830749600,  0.232537346, # noqa: E501
        -1.042096000, -0.106506582, -0.777312949,  0.510699713,  1.531191854, -0.893598380, -0.797702626,  1.057727581, # noqa: E501
        -0.496431678, -0.563155043,  0.102889627,  0.118009141,  1.326586996,  1.074658810,  3.330077780, -3.071818855, # noqa: E501
        -1.445860985,  3.535542076,  1.789262330,  0.350618971, -0.041123171,  0.685890685, -2.854784791, -0.888359823, # noqa: E501
         0.181528952,  2.070160227,  1.054061046,  1.244581629, -0.330543429,  1.515676161, -0.260578045, -2.681030502, # noqa: E501
         0.582436027,  0.856023192, -1.948274674,  1.785214979,  0.211731342, -0.282405962, -0.389455378, -1.205423343, # noqa: E501
        -0.429954065,  1.807048695,  0.963408926, -0.640395525,  1.724298423, -0.479008714, -1.096766532,  0.552644126, # noqa: E501
         1.013451090, -1.501800622,  0.466960443,  0.565185735, -1.638667748, -1.142773348, -1.560527851, -3.008045454, # noqa: E501
    ])
    # R code: > sim$perf$quant(0.1)
    y_unfocused = np.array([
        -1.783743916, -1.150020400, -1.360468655, -0.394766756, -1.164580295, -0.962921478, -1.863342250, -0.567018855, # noqa: E501
        -2.106810991, -1.641413697, -1.191665422, -1.185277105, -1.483185518, -0.541711066, -1.158172064, -1.310868275, # noqa: E501
        -1.670405812, -0.770695308, -2.195365751,  1.028745257, -1.719641547, -0.517490949, -1.019590274, -0.508146969, # noqa: E501
        -2.095930690, -1.720002135, -2.001773116, -1.050607033, -2.439281028, -1.034475573, -1.372665127,  0.475824056, # noqa: E501
        -1.419481177, -1.392745061, -1.971565886, -1.503345796, -1.098643882, -0.864228279, -0.216149238, -0.311349548, # noqa: E501
        -1.383180804,  0.121651923, -3.058327198, -0.658684175, -1.803834917,  0.040679390, -1.644991892,  0.037514177, # noqa: E501
        -1.237772498, -3.160207448, -1.728613748, -3.020149513, -1.102686717,  0.615914135, -3.553477052, -0.301087427, # noqa: E501
        -2.680377182,  0.543320857,  0.099747164, -2.120403441, -1.543547340, -1.350395594, -1.660435122,  1.300407362, # noqa: E501
        -1.151717428, -1.994576546, -0.643557323, -1.079859974, -1.351468514, -1.374041441, -0.832648292, -2.345907236, # noqa: E501
        -2.443970888,  0.366970181, -3.343647585, -1.268801845, -2.369079915, -1.011012072, -0.273099692, -3.355956320, # noqa: E501
        -0.384729294, -1.331547332, -2.626900876, -3.212763100, -0.571969982, -1.439456597, -1.065183693, -0.464189490, # noqa: E501
         0.445624189, -1.385321858, -1.838673856,  0.146749864, -2.174508968, -2.439122806, -1.811848020,  1.164131192, # noqa: E501
        -2.114047364, -0.868031717, -2.460234706, -2.455586324, -1.614474917,  0.081562141, -1.750698905, -0.438675933, # noqa: E501
        -2.739545288, -1.681857486, -2.057968851, -1.650848077, -0.041450107, -1.388985374, -1.108958059, -1.026950297, # noqa: E501
        -1.896085395, -2.710766662, -1.612527000, -1.153165502, -0.263431573, -1.537125257, -1.584092576,  0.333639117, # noqa: E501
        -2.055264920, -0.857549164, -1.865498547, -0.866515887, -2.826813222, -1.800301070, -1.561343120, -0.274094183, # noqa: E501
        -1.751121519, -0.983654527, -1.699345999, -2.131932342, -0.592505371, -1.741747760,  0.066632812, -0.838480181, # noqa: E501
        -1.432477754, -0.826002709, -1.321706247, -0.825430522, -1.689976595, -3.418045421, -1.124729649, -0.621502664, # noqa: E501
        -2.263385979, -2.395195270, -1.718899242, -1.797662812, -0.862555575, -1.147396129, -0.246865111,  0.371951660, # noqa: E501
        -1.299498383, -1.305754886, -1.031304665, -1.618676102, -1.394905271, -1.380434480, -1.017464743, -1.142567880, # noqa: E501
        -1.523821064, -1.222520184, -1.458823434, -0.486871297, -1.274813779, -1.911341859, -1.534041348, -1.971973729, # noqa: E501
        -1.079009420, -0.435170128, -0.649477504, -1.080138041, -1.372622209, -0.992067440, -1.336236505, -3.323401419, # noqa: E501
        -0.923182324, -1.654152417, -0.013242725,  0.887048752, -2.521274408, -0.691677677, -1.157532277, -1.805259352, # noqa: E501
        -0.661323563, -0.573329981, -1.374749917, -1.576748266, -2.367366796, -1.906366622, -1.514558109, -1.532368429, # noqa: E501
        -0.327656229, -1.547524072,  0.613724381, -1.711542393,  0.293995429, -1.119610366, -2.367004472, -0.704614265, # noqa: E501
        -1.253379799, -1.638254971, -0.428925190, -0.768186317, -0.263348568, -2.303030650, -1.843219836, -2.294107644, # noqa: E501
        -4.302365865, -0.949201298, -0.041039999, -0.610201966, -2.611585677, -2.132131879, -3.070382308, -0.050029651, # noqa: E501
        -1.615951239, -2.248873667, -0.402181217, -1.535130255, -2.799427354, -1.305434821, -1.254892092, -1.117870365, # noqa: E501
        -0.871543626, -1.908914165, -0.644633562, -2.483486347,  0.064990847, -1.880318876, -0.839347128, -0.668025157, # noqa: E501
        -1.580647938, -2.882139661, -1.621018521, -1.862859312, -1.671913576, -0.436037210, -0.596824361, -1.043838482, # noqa: E501
        -1.992173452,  1.331767392, -2.908198917, -2.888857859, -0.941234120,  1.446336142, -1.608741605, -2.538448314, # noqa: E501
        -1.712949671,  1.267301682, -2.052959568, -1.788893919, -1.551574149, -0.533435002, -0.612957755, -1.736850226, # noqa: E501
        -3.209396790,  0.600425501, -2.531262695, -1.770896534, -1.066160280, -0.859442892, -2.502699844, -0.969323745, # noqa: E501
        -0.395089431, -0.806740873, -0.411710211, -2.168075362, -1.410207097, -0.185041071, -0.763522019, -0.883518390, # noqa: E501
        -0.214892155, -1.731091647, -0.655449581, -2.100387516, -1.545067774, -1.126976280, -0.713106016, -3.941711243, # noqa: E501
        -0.145019024, -0.859778790,  0.068531032, -0.177794680, -0.634505456, -1.105915755, -0.627993779, -1.347041869, # noqa: E501
        -0.636029393, -0.917257529, -0.438109997, -1.958030431, -1.672984469, -0.876610921, -2.388410261, -1.664851974, # noqa: E501
        -0.987531317,  0.511509103, -0.467519394, -2.601575962, -3.136996793, -2.300116280, -2.326662652, -1.932839219, # noqa: E501
        -0.898868912, -0.249651855, -1.150476592, -0.904359832, -3.169246648, -0.761332163, -0.422398404, -2.084040221, # noqa: E501
        -1.773512082, -1.492505883, -1.745228174, -0.982071998, -1.887025392, -1.720563700, -2.002305196, -0.500746535, # noqa: E501
        -2.503835841, -0.390357529, -1.027628723, -1.347367992, -1.080085533,  1.196148948, -0.809798771,  0.044646517, # noqa: E501
        -0.612953080, -1.418945979, -2.692723262, -0.081188461, -0.982461458, -0.825324683, -0.843473121, -1.496572704, # noqa: E501
        -1.593674042, -1.974181985, -3.677781249, -2.865826499,  0.477196668, -3.644438354, -0.888379498, -0.715613701, # noqa: E501
        -2.325541936, -0.258438273, -1.794215580, -1.233814891,  0.111325432, -1.714050691, -1.416719402,  0.016183585, # noqa: E501
        -1.352278049, -1.702379760, -0.140211208, -1.708678050,  0.114118041, -1.577738735, -0.698891955, -3.957780917, # noqa: E501
        -2.682230575,  2.022599545, -0.424774024, -0.120535120, -1.002614680, -1.295100061, -3.957930452, -1.160237111, # noqa: E501
        -1.782094808,  0.098126345, -1.687538224,  0.114380379, -0.542955410, -0.544937413, -2.014639519, -3.179947615, # noqa: E501
        -0.835672264, -0.825290703, -2.807748774, -1.362134183, -1.621562172, -1.227870767, -1.973186415, -1.585046461, # noqa: E501
        -2.476565614, -0.261815650, -1.759995788, -1.941043615, -0.602671913, -1.386906893, -1.735269238,  0.059887066, # noqa: E501
        -0.295594365, -2.477016341, -1.856410139, -0.299560203, -2.938934791, -2.401198498, -2.452375972, -2.420603177, # noqa: E501
    ])
    # Isotonic solution of y_obs on x=y_unfocused. This is sorted.
    # R code: > result_reldiag = reldiag(sim$perf$quant(0.1), sim$y, type = list("quantile",alpha = 0.1))  # noqa: E501
    #         > result_reldiag$x_rc
    y_recalibrated = np.array([
        -3.9472239, -3.9472239, -3.9472239, -3.9472239, -3.9472239, -3.2083545, -3.2083545, -3.2083545, -3.2083545, -3.2083545, # noqa: E501
        -3.2083545, -3.2083545, -3.2083545, -3.2083545, -3.2083545, -3.2083545, -2.8635112, -2.8635112, -2.4703789, -2.4703789, # noqa: E501
        -2.4703789, -2.4703789, -2.4703789, -2.4703789, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, # noqa: E501
        -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, # noqa: E501
        -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, # noqa: E501
        -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, # noqa: E501
        -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, # noqa: E501
        -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.4616281, -2.1989588, -2.1989588, -2.1989588, -2.1989588, # noqa: E501
        -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, # noqa: E501
        -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, -2.1989588, -1.6577284, # noqa: E501
        -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, # noqa: E501
        -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, -1.6577284, # noqa: E501
        -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, # noqa: E501
        -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, # noqa: E501
        -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, # noqa: E501
        -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, # noqa: E501
        -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.2226609, # noqa: E501
        -1.2226609, -1.2226609, -1.2226609, -1.2226609, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, # noqa: E501
        -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, # noqa: E501
        -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, # noqa: E501
        -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, # noqa: E501
        -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, # noqa: E501
        -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, # noqa: E501
        -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, # noqa: E501
        -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, # noqa: E501
        -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, -1.1168636, # noqa: E501
        -1.1168636, -1.1168636, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, # noqa: E501
        -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, # noqa: E501
        -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, # noqa: E501
        -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, # noqa: E501
        -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.8307496, # noqa: E501
        -0.8307496, -0.8307496, -0.8307496, -0.8307496, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, # noqa: E501
        -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, # noqa: E501
        -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.4003760, -0.2353513, -0.2353513, # noqa: E501
        -0.2353513, -0.2353513, -0.2353513, -0.2353513, -0.2353513, -0.2353513, -0.2353513, -0.2353513, -0.2353513, -0.2353513, # noqa: E501
         0.1028896,  0.1028896,  0.1028896,  0.1028896,  0.1028896,  0.1028896,  0.1028896,  0.1890516,  0.1890516,  0.1890516, # noqa: E501
         0.1890516,  0.1890516,  0.1890516,  0.1890516,  0.1890516,  0.1890516,  0.1890516,  0.1890516,  0.1890516,  0.1890516, # noqa: E501
         0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.3695697, # noqa: E501
         0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.3695697,  0.8368062,  0.8368062, # noqa: E501
         1.1886707,  1.1886707,  1.1886707,  1.1886707,  1.1886707,  2.7300730,  2.7300730,  3.1544399,  3.1544399,  3.5355421, # noqa: E501
    ])
    # fmt: on
    level = 0.1
    # R code > result_reldiag$decomp
    # This gives decomp = c(umcb,cmcb,mcb,dsc,unc)
    # Note that those numbers are twice our pinball loss => factor 0.5.
    decomp_unfocused = [
        2.847969e-05,
        3.099230e-02,
        3.102078e-02,
        1.803370e-01,
        5.004139e-01,
    ]
    mcb, dsc, unc = decomp_unfocused[2:]

    # Important note:
    # The R solution from gpava, package https://cran.r-project.org/package=isotone,
    # gives the lower quantile ("inverted_cdf", type=1 in R quantile), so we can't just
    # use our decompose which uses the midpoint between lower and upper quantile.
    # Therefore, we redo the computation here.
    df = pl.DataFrame({"_X": y_unfocused, "_target_y": y_obs})
    df = df.sort(by=["_X", "_target_y"], descending=[False, True])

    def quantile_lower(x, wx=None):
        return np.quantile(x, level, method="inverted_cdf")

    def quantile_upper(x, wx=None):
        return -np.quantile(
            -np.asarray(x), float(1 - Decimal(str(level))), method="higher"
        )

    xl, rl = gpava(quantile_lower, df["_target_y"])
    # Applying gpava on fun=np.quantile(x, level, method="higher") does not work
    # for unknown reasons. What works is to calculate the upper quantile on the
    # blocks given by rl from the lower quantile.
    y_sorted = df["_target_y"].to_numpy()
    q = np.fromiter(
        (quantile_upper(y_sorted[rl[i] : rl[i + 1]]) for i in range(len(rl) - 1)),
        dtype=xl.dtype,
    )
    # Take mininum from the right.
    q = np.minimum.accumulate(q[::-1])[::-1]
    np.repeat(q, np.diff(rl))
    assert_allclose(xl, y_recalibrated, rtol=5e-7)

    pbl = PinballLoss(level=level)

    marginal = np.full_like(
        y_obs, fill_value=np.quantile(y_obs, q=level, method="inverted_cdf")
    )
    score = pbl(y_obs=y_obs, y_pred=y_unfocused)
    score_recalibrated = pbl(y_obs=df["_target_y"], y_pred=xl)
    score_marginal = pbl(y_obs=df["_target_y"], y_pred=marginal)
    miscalibration = score - score_recalibrated
    discrimination = score_marginal - score_recalibrated
    uncertainty = score_marginal
    assert score == pytest.approx(0.5 * (mcb - dsc + unc))
    assert miscalibration == pytest.approx(0.5 * mcb)
    assert discrimination == pytest.approx(0.5 * dsc)
    assert uncertainty == pytest.approx(0.5 * unc)

    # The upper solution should have the same loss.
    assert pbl(y_obs=df["_target_y"], y_pred=xl) == pytest.approx(score_recalibrated)

    # And decompose should have the same loss, too.
    df = decompose(y_obs=y_obs, y_pred=y_unfocused, scoring_function=pbl)
    assert df.get_column("miscalibration")[0] == pytest.approx(0.5 * mcb)
    assert df.get_column("discrimination")[0] == pytest.approx(0.5 * dsc)
    assert df.get_column("uncertainty")[0] == pytest.approx(0.5 * unc)


@pytest.mark.parametrize(
    "sf", [SquaredError(), HES(degree=2, level=0.2), PinballLoss(level=0.8)]
)
def test_decompose_multiple_predictions(sf):
    """Test decompose for multiple predictions."""
    n_obs = 10
    y_obs = np.arange(n_obs)
    y_pred = pl.DataFrame({"model_1": y_obs, "model_2": 3 * y_obs})
    # isotonic regr aka recalibrated = y_obs

    df_decomp = decompose(
        y_obs=y_obs,
        y_pred=y_pred,
        scoring_function=sf,
    )
    if sf.functional == "mean":
        marginal = np.average(y_obs)
    elif sf.functional == "expectile":
        marginal = scipy.stats.expectile(y_obs, alpha=sf.level)
    elif sf.functional == "quantile":
        marginal = np.quantile(y_obs, q=sf.level, method="inverted_cdf")
    marginal = np.full_like(y_obs, fill_value=marginal, dtype=float)

    score_m1 = sf(y_obs, y_pred["model_1"])
    score_m2 = sf(y_obs, y_pred["model_2"])
    score_marginal = sf(y_obs, marginal)
    df_expected = pl.DataFrame(
        {
            "model": ["model_1", "model_2"],
            "miscalibration": [score_m1, score_m2],
            "discrimination": [score_marginal] * 2,
            "uncertainty": [score_marginal] * 2,
            "score": [score_m1, score_m2],
        }
    )
    assert_frame_equal(df_decomp, df_expected, check_exact=False)

    # Same for pure numpy input.
    df_decomp = decompose(
        y_obs=y_obs,
        y_pred=y_pred.to_numpy(),
        scoring_function=sf,
    )
    df_expected = df_expected.with_columns(pl.Series(["0", "1"]).alias("model"))
    assert_frame_equal(df_decomp, df_expected, check_exact=False)


@pytest.mark.parametrize(
    ("y", "sf"),
    [(0, PoissonDeviance()), (0, HomogeneousExpectileScore(degree=0.1, level=0.9))],
)
def test_decompose_constant_0_y_obs(y, sf):
    """Test decompose for y_obs = 0.

    This is a problem if y_pred = y_obs = 0 is not allowed like for Poisson deciance.
    """
    n = 4
    y_obs = np.full(n, fill_value=y)
    y_pred = np.ones(n)

    with pytest.raises(ValueError, match="Your y_obs is constant"):
        decompose(
            y_obs=y_obs,
            y_pred=y_pred,
            scoring_function=sf,
        )


@pytest.mark.parametrize(
    ("y_obs", "y_pred", "sf", "recalibrated"),
    [
        ([0, 1], [0.25, 0.75], PoissonDeviance(), [0.5] * 2),
        ([0, 0, 1, 1], [0.3, 0.2, 0.8, 0.7], PoissonDeviance(), [0.5] * 4),
        (
            [0, 0, 1, 1, 2],
            [0.3, 0.2, 0.8, 0.7, 7],
            PoissonDeviance(),
            [0.5, 0.5, 0.5, 0.5, 2],
        ),
        (
            [0, 0, 1, 1, 2],
            [0.3, 0.2, 0.8, 0.7, 7],
            HomogeneousExpectileScore(degree=1, level=0.25),
            [0.25, 0.25, 0.25, 0.25, 2],
        ),
    ],
)
def test_decompose_isotonic_outside_range(y_obs, y_pred, sf, recalibrated):
    """Test that decompose works if isotonic regression is outside calid domain."""
    df = decompose(
        y_obs=y_obs,
        y_pred=y_pred,
        scoring_function=sf,
    )
    score = sf(y_obs=y_obs, y_pred=y_pred)
    score_recalibrated = sf(y_obs=y_obs, y_pred=recalibrated)
    assert score - score_recalibrated == pytest.approx(df["miscalibration"][0])
