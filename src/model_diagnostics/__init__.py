from importlib.metadata import version

from packaging.version import parse

from ._config import config_context, get_config, set_config

polars_version = parse(version("polars"))

__all__ = [
    "get_config",
    "set_config",
    "config_context",
]
