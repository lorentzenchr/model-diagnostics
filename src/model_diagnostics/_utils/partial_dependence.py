import copy
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import polars as pl

from model_diagnostics._utils.array import (
    is_pandas_df,
    is_pyarrow_table,
    length_of_first_dimension,
    safe_assign_column,
    safe_index_rows,
)


def compute_partial_dependence(
    pred_fun: Callable,
    X: npt.ArrayLike,
    feature_index: int,
    grid: npt.ArrayLike,
    weights: Optional[npt.ArrayLike] = None,
    n_max: int = 1000,
    rng: Optional[Union[np.random.Generator, int]] = None,
):
    """Compute partial dependence.

    This is a fast brute force method to compute partial dependence values for the
    given grid.

    Parameters
    ----------
    pred_fun : callable
        Prediction function, such that `pred_fun(X)` gives predicted values.
     X : array-like of shape (n_obs, n_features)
        The dataframe or array of features to be passed to the model predict function.
    feature_index : int
        Index / Position of the feature in `X`.
    grid : pl.Series
        Values of the feature, specified by feature_index, for wich to compute partial
        dependence.
    weights : array-like of shape (n_obs) or None
        Case weights. If given, the bias is calculated as weighted average of the
        identification function with these weights.
    n_max : int or None
        The number of rows to subsample from X. This speeds up computation, in
        particular for slow predict functions.
    rng : np.random.Generator, int or None
        The random number generator. The used one will be `np.random.default_rng(rng)`.

    Returns
    -------
    np.ndarray : shape (n_grid,)
        Partial dependence values for the grid.
    """
    n = length_of_first_dimension(X)
    n_grid = length_of_first_dimension(grid)

    # Usually, the data is too large and we need subsampling.
    if n_max is not None and n > n_max:
        rng_ = np.random.default_rng(rng)
        row_indices = rng_.choice(n, size=n_max, replace=False)
        X = safe_index_rows(X, row_indices)
        if weights is not None:
            weights = safe_index_rows(weights, row_indices)
        n = n_max
    elif hasattr(X, "copy"):
        # pandas
        X = X.copy()
    elif is_pyarrow_table(X) or isinstance(X, pl.DataFrame):
        # Copy on Write
        pass
    else:
        X = copy.deepcopy(X)

    # X is stacked n_grid times, and grid column is replaced by replicated grid
    X_stacked = safe_index_rows(X, np.tile(np.arange(n), n_grid))
    grid_stacked = safe_index_rows(grid, np.repeat(np.arange(n_grid), n))

    if is_pandas_df(X):
        # pandas<2 does not allow "values" to have repeated indices
        X_stacked = X_stacked.reset_index(drop=True)
    X_stacked = safe_assign_column(
        X_stacked, values=grid_stacked, column_index=feature_index
    )

    y_pred = pred_fun(X_stacked)
    if hasattr(y_pred, "to_numpy"):
        # pandas.Series, polars.Series, pyarrow.Array, pyarrow.ChunkedArray
        y_pred = y_pred.to_numpy()

    # Partial dependences are averages per grid block
    pd_values = np.average(
        y_pred.reshape(n_grid, y_pred.shape[0] // n_grid),
        axis=1,
        weights=weights,
    )

    return pd_values
