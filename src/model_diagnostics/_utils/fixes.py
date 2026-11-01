from packaging.version import Version
import polars as pl

# The polars.StringCache got deprecated in version 1.41.
if Version(pl.__version__) < Version("1.41.0"):
    StringCache = pl.StringCache
else:
    class StringCache():
        # Could be used as context manager, therefore needs __enter__ and __exit__.
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
