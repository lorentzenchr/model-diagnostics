class SkipContainer:
    def __init__(self, *args, **kwargs):
        pass

    def __getitem__(self, *args):
        return self

    def column(self, *args):
        return self


pa_available = True
try:
    import pyarrow as pa

    pa_array = pa.array
    pa_DictionaryArray_from_arrays = pa.DictionaryArray.from_arrays
    pa_float64 = pa.float64
    # pa_chunked_array = pa.chunked_array
    pa_table = pa.table
except ImportError:
    pa_available = False
    pa_array = SkipContainer
    pa_DictionaryArray_from_arrays = SkipContainer
    pa_float64 = SkipContainer
    # pa_chunked_array = SkipContainer
    pa_table = SkipContainer


pd_available = True
try:
    import pandas as pd

    pd_DataFrame = pd.DataFrame
    pd_Series = pd.Series
except ImportError:
    pd_available = False
    pd_DataFrame = SkipContainer
    pd_Series = SkipContainer


__all__ = [
    "pa_available",
    "pa_array",
    "pd_available",
    "pa_DictionaryArray_from_arrays",
    "pa_float64",
    "pa_table",
    "pd_DataFrame",
    "pd_Series",
    "SkipContainer",
]
