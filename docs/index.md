# model-diagnostics

| | |
| --- | --- |
| CI/CD |[![CI - Test](https://github.com/lorentzenchr/model-diagnostics/actions/workflows/test.yml/badge.svg)](https://github.com/lorentzenchr/model-diagnostics/actions/workflows/test.yml) [![Coverage](https://codecov.io/github/lorentzenchr/model-diagnostics/coverage.svg?branch=main)](https://codecov.io/gh/lorentzenchr/model-diagnostics)
| Docs | [![Docs](https://github.com/lorentzenchr/model-diagnostics/actions/workflows/docs.yml/badge.svg)](https://github.com/lorentzenchr/model-diagnostics/actions/workflows/docs.yml)
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/model-diagnostics.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/model-diagnostics/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/model-diagnostics.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.org/project/model-diagnostics/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/model-diagnostics.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/model-diagnostics/) |
| Meta | [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff) [![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![License - MIT](https://img.shields.io/badge/license-MIT-9400d3.svg)](https://spdx.org/licenses/)

## Tools for diagnostics and assessment of (machine learning) models

Highlights:

- Assess model calibration with identification functions (generalized residuals).
- Assess calibration graphically
    - reliability diagrams for auto-calibration
    - bias plots for conditional calibration
- Assess the predictive performance of models
    - strictly consistent, homogeneous scoring functions
    - score decomposition into miscalibration, discrimination and uncertainty

## Installation

`pip install model-diagnostics`

## Contributions

Contributions are warmly welcome!
When contributing, you agree that your contributions will be subject to the [MIT License](https://github.com/lorentzenchr/model-diagnostics/blob/main/LICENSE).