from importlib.metadata import version

from packaging.version import parse

polars_version = parse(version("polars"))
