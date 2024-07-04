# model-diagnostics

| | |
| --- | --- |
| CI/CD |[![CI - Test](https://github.com/lorentzenchr/model-diagnostics/actions/workflows/test.yml/badge.svg)](https://github.com/lorentzenchr/model-diagnostics/actions/workflows/test.yml) [![Coverage](https://codecov.io/github/lorentzenchr/model-diagnostics/coverage.svg?branch=main)](https://codecov.io/gh/lorentzenchr/model-diagnostics)
| Docs | [![Docs](https://github.com/lorentzenchr/model-diagnostics/actions/workflows/docs.yml/badge.svg)](https://github.com/lorentzenchr/model-diagnostics/actions/workflows/docs.yml)
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/model-diagnostics.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/model-diagnostics/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/model-diagnostics.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.org/project/model-diagnostics/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/model-diagnostics.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/model-diagnostics/) |
| Meta | [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![License - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/)

## Tools for diagnostics and assessment of (machine learning) models

Highlights:

- All common point predictions covered: mean, median, quantiles, expectiles.
- Assess model calibration with [identification functions][model_diagnostics.calibration.identification.identification_function] (generalized residuals), [compute_bias][model_diagnostics.calibration.identification.compute_bias] and [compute_marginal][model_diagnostics.calibration.identification.compute_marginal].
- Assess calibration and bias graphically
    - [reliability diagrams][model_diagnostics.calibration.plots.plot_reliability_diagram] for auto-calibration
    - [bias plots][model_diagnostics.calibration.plots.plot_bias] for conditional calibration
    - [marginal plots][model_diagnostics.calibration.plots.plot_marginal] for average `y_obs`, `y_pred` and partial dependence for one feature
- Assess the predictive performance of models
    - strictly consistent, homogeneous [scoring functions][model_diagnostics.scoring.scoring]
    - [score decomposition][model_diagnostics.scoring.decompose] into miscalibration, discrimination and uncertainty
- Choose your plot backend, either [matplotlib](https://matplotlib.org) or [plotly](https://plotly.com/python/), e.g., via [set_config][model_diagnostics.set_config].

:rocket: To our knowledge, this is the first python package to offer reliability diagrams for quantiles and expectiles and a score decomposition, both made available by an internal implementation of isotonic quantile/expectile regression. :rocket:

This package relies on the giant shoulders of, among others, [polars](https://pola.rs/), [matplotlib](https://matplotlib.org), [scipy](https://scipy.org) and [scikit-learn](https://scikit-learn.org).

## Installation

```
pip install model-diagnostics
```

## Contributions

Contributions are warmly welcome!
When contributing, you agree that your contributions will be subject to the [MIT License](https://github.com/lorentzenchr/model-diagnostics/blob/main/LICENSE).