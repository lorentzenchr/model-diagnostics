"""
Global configuration state and functions for management
To a large part taken from scikit-learn.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from importlib.util import find_spec
from typing import Optional

_global_config = {
    "plot_backend": "matplotlib",
}


def get_config() -> dict:
    """Retrieve current values for configuration set by :func:`set_config`.

    Returns
    -------
    config : dict
        A copy of the configuration dictionary. Keys are parameter names that can be
        passed to :func:`set_config`.

    See Also
    --------
    config_context : Context manager for global model-diagnostics configuration.
    set_config : Set global model-diagnostics configuration.

    Examples
    --------
    >>> import model_diagnostics
    >>> config = model_diagnostics.get_config()
    >>> config.keys()
    dict_keys([...])
    """
    # Return a copy of the global config so that users will
    # not be able to modify the configuration with the returned dict.
    return _global_config.copy()


def set_config(
    plot_backend: Optional[str] = None,
) -> None:
    """Set global model-diagnostics configuration.

    Parameters
    ----------
    plot_backend : bool, default=None
        The library used for plotting. Can be "matplotlib" or "plotly".
        If None, the existing value won't change. Global default: "matplotlib".

    See Also
    --------
    config_context : Context manager for global scikit-learn configuration.
    get_config : Retrieve current values of the global configuration.

    Examples
    --------
    >>> from model_diagnostics import set_config
    >>> set_config(plot_backend="plotly")  # doctest: +SKIP
    """
    if plot_backend not in (None, "matplotlib", "plotly"):
        msg = f"The plot_backend must be matplotlib or plotly, got {plot_backend}."
        raise ValueError(msg)
    if plot_backend == "plotly" and not find_spec("plotly"):
        msg = (
            "In order to set the plot backend to plotly, plotly must be installed, "
            "i.e. via `pip install plotly`."
        )
        raise ModuleNotFoundError(msg)

    if plot_backend is not None:
        _global_config["plot_backend"] = plot_backend


@contextmanager
def config_context(
    *,
    plot_backend: Optional[str] = None,
) -> Iterator[None]:
    """Context manager for global model-diagnostics configuration.

    Parameters
    ----------
    plot_backend : bool, default=None
        The library used for plotting. Can be "matplotlib" or "plotly".
        If None, the existing value won't change. Global default: "matplotlib".

    Yields
    ------
    None.

    See Also
    --------
    set_config : Set global model-diagnostics configuration.
    get_config : Retrieve current values of the global configuration.

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited.

    Examples
    --------
    >>> import model_diagnostics
    >>> from model_diagnostics.calibration import plot_reliability_diagram
    >>> with model_diagnostics.config_context(plot_backend="plotly"):  # doctest: +SKIP
    ...    plot_reliability_diagram(y_obs=[0, 1], y_pred=[0.3, 0.7])
    """
    old_config = get_config()
    set_config(
        plot_backend=plot_backend,
    )

    try:
        yield
    finally:
        set_config(**old_config)
