import polars as pl
from packaging.version import Version

# The polars.StringCache got deprecated in version 1.41.
if Version(pl.__version__) < Version("1.41.0"):
    StringCache = pl.StringCache
else:

    class StringCache:  # type: ignore
        # Could be used as context manager, therefore needs __enter__ and __exit__.
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
