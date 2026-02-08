from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import polars as pl

from model_diagnostics._utils.array import (
    get_column_names,
    get_second_dimension,
    is_pyarrow_array,
    length_of_first_dimension,
    safe_assign_column,
    safe_copy,
    safe_index_rows,
)
from model_diagnostics.scoring import SquaredError


def _rearrange_rows_of_some_columns(X, columns, row_indices):
    """Rearrange values in specific columns according to provided indices.

    This function creates a copy of the input data and rearranges the values
    in the specified columns according to the provided row indices. It supports
    various data container formats (numpy arrays, pandas DataFrames, polars
    DataFrames, PyArrow Tables).

    Parameters
    ----------
    X : array-like
        The input data which can be a numpy array, pandas DataFrame,
        polars DataFrame, PyArrow Table, or other similar data container.
    columns : str, int, or list of str or int
        Column name(s) or index/indices of the column(s) to be rearranged.
    row_indices : array-like
        Indices specifying the new order of rows for the specified column(s).

    Returns
    -------
    array-like
        A copy of X with rearranged values in the specified columns.
    """
    X = safe_copy(X)  # Important
    if isinstance(columns, (str, int)):
        columns = [columns]
    all_columns = get_column_names(X)

    for v in columns:
        column_index = all_columns.index(v) if isinstance(v, str) else v
        x = get_second_dimension(X, column_index)
        x_shuffled = safe_index_rows(x, row_indices)
        X = safe_assign_column(X, values=x_shuffled, column_index=column_index)

    return X


def compute_permutation_importance(
    pred_fun: Callable,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    features: Optional[Union[list, tuple, set, dict]] = None,
    scoring_function: Callable = SquaredError(),
    weights: Optional[npt.ArrayLike] = None,
    n_repeats: int = 5,
    n_max: int = 10_000,
    scoring_orientation: str = "smaller_is_better",
    rng: Optional[Union[np.random.Generator, int]] = None,
):
    """Compute permutation feature importance.

    This function calculates permutation feature importance for features and/or
    feature groups according to the idea in `[Breiman]` and `[Fisher]`.

    For each feature (group), permutation importance measures how much the model
    performance worsenes when shuffling the values of that feature (group) before
    calculating predictions. The idea is that if a feature is important,
    then shuffling its values will lead to a large drop in model performance.
    Shuffling is done `n_repeats` times, and mean differences and mean ratios are
    returned along with their standard errors.

    Note that the model is never retrained during this process.

    Parameters
    ----------

    pred_fun : callable
        A callable to get predictions, i.e. `pred_fun(X)`.
    X : array-like of shape (n_obs, n_features)
        The dataframe or array of features to be passed to the model predict function.
    y : npt.ArrayLike
        1D array of shape (n_observations,) containing the target values.
    features: list, tuple, dict, default=None
        Iterable of feature names/indices of features in `X`. The default None
        will use all features in `X`. Can also be a dictionary with lists of feature
        names/indices as values. The keys of the dictionary are used as feature group
        names. Example: `{"x1": ["x1"], "x2": ["x2"], "size": ["x1", "x2"]}`.
        Passing a dictionary is also useful if you want to represent feature indices
        of a numpy array as strings. Example: `{"area": 0, "age": 1}`.
    scoring_function : callable, default=SquaredError()
        A scoring function with signature roughly
        `fun(y_obs, y_pred, weights) -> float`.
    weights : array-like of shape (n_obs) or None, default=None
        Case weights passed to the scoring_function.
    n_repeats : int, default=5
        Number of times to repeat the permutation for each feature group.
    n_max : int or None, default=10_000
        Maximum number of observations used. If the number of observations is greater
        than `n_max`, a random subset of size `n_max` will be drawn from `X`, `y`, (and
        `weights`). Pass None for no subsampling.
    scoring_orientation : str, default="smaller_is_better"
        Direction of scoring function. Use "smaller_is_better" if smaller values are
        better (e.g., average losses), or "greater_is_better" if greater values are
        better (e.g., R-squared).
    rng : np.random.Generator, int or None, default=None
        The random number generator used for shuffling values and for subsampling
        `n_max` rows. The input is internally wrapped by `np.random.default_rng(rng)`.

    Returns
    -------
    df : polars.DataFrame
        A DataFrame with one row per feature (group) and the following columns:

        - `feature`: Feature name or feature group name.
        - `difference_mean`: Mean of the score differences.
        - `difference_stderr`: Standard error, i.e. standard deviation of
          `difference_mean`. (None if `n_repeats = 1`.)
        - `ratio_mean`: Mean of the score ratios.
        - `ratio_stderr`: Standard error, i.e. standard deviation of `ratio_mean`.
          (None if `n_repeats = 1`.)

    References
    ----------
    `[Breiman]`

    :   Breiman, L. (2001).
        "Random Forests".
        Machine Learning, 45(1), 5-32.
        https://doi.org/10.1023/A:1010933404324

    `[Fisher]`

    :   Fisher, A. and Rudin, C. and Dominici F. (2019).
        "All Models Are Wrong, but Many Are Useful: Learning a Variable's Importance
        by Studying an Entire Class of Prediction Models Simultaneously".
        Journal of Machine Learning Research, 20(177), 1-81.

    Examples
    --------
    >>> import numpy as np
    >>> import polars as pl
    >>> from sklearn.linear_model import LinearRegression
    >>> # Create a synthetic dataset
    >>> rng = np.random.default_rng(1)
    >>> n = 1000
    >>> X = pl.DataFrame(
    ...     {
    ...         "rooms": rng.choice([2.5, 3.5, 4.5], n),
    ...         "area": rng.uniform(30, 120, n),
    ...         "age": rng.uniform(0, 100, n),
    ...     }
    ... )
    >>> y = X["area"] + 20 * X["rooms"] + rng.normal(0, 10, n)
    >>> model = LinearRegression()
    >>> _ = model.fit(X, y)
    >>> perm_importance = compute_permutation_importance(
    ...     pred_fun=model.predict,
    ...     X=X,
    ...     y=y,
    ...     rng=1,
    ... )
    >>> perm_importance
    shape: (3, 5)
    ┌─────────┬─────────────────┬───────────────────┬────────────┬──────────────┐
    │ feature ┆ difference_mean ┆ difference_stderr ┆ ratio_mean ┆ ratio_stderr │
    │ ---     ┆ ---             ┆ ---               ┆ ---        ┆ ---          │
    │ str     ┆ f64             ┆ f64               ┆ f64        ┆ f64          │
    ╞═════════╪═════════════════╪═══════════════════╪════════════╪══════════════╡
    │ rooms   ┆ 524.213195      ┆ 8.813555          ┆ 6.263515   ┆ 0.088495     │
    │ area    ┆ 1328.885114     ┆ 15.924463         ┆ 14.343058  ┆ 0.159894     │
    │ age     ┆ 0.174047        ┆ 0.090023          ┆ 1.001748   ┆ 0.000904     │
    └─────────┴─────────────────┴───────────────────┴────────────┴──────────────┘

    Using feature subsets
    >>> perm_importance = compute_permutation_importance(
    ...     pred_fun=model.predict,
    ...     X=X,
    ...     y=y,
    ...     features=["area", "age"],
    ...     rng=1,
    ... )

    Using feature groups
    >>> perm_importance = compute_permutation_importance(
    ...     pred_fun=model.predict,
    ...     X=X,
    ...     y=y,
    ...     features={"size": ["area", "rooms"], "age": "age"},
    ...     rng=1,
    ... )
    """
    if n_repeats < 1:
        msg = f"Argument n_repeats must be >= 1, got {n_repeats}."
        raise ValueError(msg)

    if scoring_orientation not in ("smaller_is_better", "greater_is_better"):
        msg = (
            f"Argument scoring_orientation must be 'smaller_is_better' or "
            f"'greater_is_better', got {scoring_orientation}."
        )
        raise ValueError(msg)

    # Turn features into form {"x1": ["x1"], "x2": ["x2"], "group": ["x1", "x2"]}.
    # While looking verbose, it is the most flexible way to handle all cases.
    if features is None:
        features = get_column_names(X)
    if not isinstance(features, dict):
        features = {v: [v] for v in features}

    # Sometimes, the data is too large and we need subsampling.
    n = length_of_first_dimension(X)
    rng_ = np.random.default_rng(rng)
    if n_max is not None and n > n_max:
        row_indices = rng_.choice(n, size=n_max, replace=False)
        X = safe_index_rows(X, row_indices)
        y = safe_index_rows(y, row_indices)
        if weights is not None:
            weights = safe_index_rows(weights, row_indices)
        n = n_max
    else:
        X = safe_copy(X)

    # Calculate pre-shuffle score before stacking X
    base_score = scoring_function(y, pred_fun(X), weights=weights)

    # Stack X per repetition
    if n_repeats >= 2:
        X = safe_index_rows(X, np.tile(np.arange(n), n_repeats))

    # Loop over feature groups
    feature_scores = []
    feature_groups = features.keys()

    for feature_group in feature_groups:
        shuffle_indices = np.concatenate(
            [rng_.permutation(n) for _ in range(n_repeats)]
        )
        X_shuffled = _rearrange_rows_of_some_columns(
            X, features[feature_group], shuffle_indices
        )

        predictions = pred_fun(X_shuffled)

        if is_pyarrow_array(predictions):  # np.split() does not work on pyarrow arrays
            predictions = predictions.to_numpy()
        scores_per_repetition = [
            scoring_function(y, pred, weights=weights)
            for pred in np.split(predictions, n_repeats, axis=0)
        ]
        feature_scores.append(pl.Series(scores_per_repetition))

    # Differences and ratios wrt base score
    direction = 1 if scoring_orientation == "smaller_is_better" else -1
    difference_scores = [direction * (s - base_score) for s in feature_scores]
    ratio_scores = [(s / base_score) ** direction for s in feature_scores]

    # Aggregate over repetitions
    difference_mean = [s.mean() for s in difference_scores]
    ratio_mean = [s.mean() for s in ratio_scores]
    if n_repeats >= 2:
        difference_stderr = [s.std() for s in difference_scores] / np.sqrt(n_repeats)
        ratio_stderr = [s.std() for s in ratio_scores] / np.sqrt(n_repeats)
    else:
        difference_stderr, ratio_stderr = None, None

    out = pl.DataFrame(
        {
            "feature": feature_groups,
            "difference_mean": difference_mean,
            "difference_stderr": difference_stderr,
            "ratio_mean": ratio_mean,
            "ratio_stderr": ratio_stderr,
        }
    )

    return out
