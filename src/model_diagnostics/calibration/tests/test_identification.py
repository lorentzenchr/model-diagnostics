from functools import partial

import numpy as np
import pytest
from scipy.stats import expectile

from model_diagnostics.calibration import identification_function


@pytest.mark.parametrize(
    ("functional", "est"),
    [
        ("mean", np.mean),
        ("median", np.median),
        ("expectile", expectile),
        ("quantile", np.quantile),
    ],
)
@pytest.mark.parametrize("level", [0.01, 0.4, 0.4999, 0.5, 0.50001, 0.8, 0.999])
def test_identification_function_equal_input(functional, est, level):
    rng = np.random.default_rng(112358132134)
    n = 1000
    if functional == "quantile":
        est = partial(est, q=level)
    elif functional == "expectile":
        est = partial(est, alpha=level)
    y_obs = rng.normal(loc=-2, scale=2, size=n) + rng.normal(loc=1, scale=0.1, size=n)
    y_pred = np.ones_like(y_obs) * est(y_obs)
    id_func = identification_function(y_obs, y_pred, functional=functional, level=level)
    assert np.mean(id_func) == pytest.approx(0, abs=1e-1 / n)


@pytest.mark.parametrize(
    ("functional", "level", "msg"),
    [
        ("no good functional", 0.5, "Argument functional must be one of"),
        ("quantile", 0, "Argument level must fulfil 0 < level < 1"),
        ("quantile", 1, "Argument level must fulfil 0 < level < 1"),
        ("quantile", 1.1, "Argument level must fulfil 0 < level < 1"),
    ],
)
def test_identification_function_raises(functional, level, msg):
    y_obs, y_pred = np.arange(5), np.arange(5)
    with pytest.raises(ValueError, match=msg):
        identification_function(y_obs, y_pred, functional=functional, level=level)
