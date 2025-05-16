import copy
from typing import Callable, Optional, Union

import numpy as np
import numpy.typing as npt
import polars as pl

from model_diagnostics._utils.array import (
    is_pandas_df,
    is_pandas_series,
    is_pyarrow_table,
    length_of_first_dimension,
    safe_assign_column,
    safe_index_rows,
)
from model_diagnostics.scoring import SquaredError


def safe_copy(X):
    if hasattr(X, "copy"):
        # pandas
        X = X.copy()
    elif is_pyarrow_table(X) or isinstance(X, pl.DataFrame):
        # Copy on Write
        pass
    else:
        X = copy.deepcopy(X)
    return X


def safe_column_names(X):
    """If we have column names, return them. Otherwise, return indices."""
    if is_pyarrow_table(X):
        return X.column_names
    elif is_pandas_df(X):
        return X.columns.to_list()
    elif hasattr(X, "columns"):
        # polars
        return X.columns
    else:
        # numpy
        return list(range(X.shape[1]))


def safe_index_rows_1d(x, row_indices):
    # safe_index_rows() does not work for pandas series
    if is_pandas_series(x):
        return x.iloc[row_indices]
    return safe_index_rows(x, row_indices)


def safe_select_column(X, index):
    if hasattr(X, "iloc"):
        # pandas
        out = X.iloc[:, index]
    elif is_pyarrow_table(X):
        out = X.column(index)
    elif hasattr(X, "select"):
        # polars
        all_columns = safe_column_names(X)
        out = X.select(all_columns[index])
    else:
        # numpy
        out = X[:, index]

    return out


def safe_shuffle_cols(X, columns, row_indices):
    X = safe_copy(X)  # Important
    if isinstance(columns, (str, int)):
        columns = [columns]
    all_columns = safe_column_names(X)

    for v in columns:
        column_index = all_columns.index(v) if isinstance(v, str) else v
        x = safe_select_column(X, column_index)
        x_shuffled = safe_index_rows_1d(x, row_indices)
        X = safe_assign_column(X, values=x_shuffled, column_index=column_index)

    return X


def compute_permutation_importance(
    predict_function: Callable,
    X: npt.ArrayLike,
    y: npt.ArrayLike,
    features: Optional[Union[list, tuple, set, dict]] = None,
    scoring_function: Callable = SquaredError(),
    weights: Optional[npt.ArrayLike] = None,
    n_repeats: Optional[int] = 5,
    n_max: Optional[int] = 10_000,
    method: Optional[str] = "difference",
    smaller_is_better: Optional[bool] = True,
    rng: Optional[Union[np.random.Generator, int]] = None,
):
    """Compute permutation feature importance.

    This function calculates permutation feature importance for features and/or
    feature groups according to the idea in `[Breiman]` and `[Fisher]`.

    For each feature (group), permutation importance measures how much the model
    performance worsenes when shuffling the values of that feature (group) before
    calculating predictions. The idea is that if a feature is important,
    then shuffling its values will lead to a large drop in model performance.
    Note that the model is never retrained during this process.

    Parameters
    ----------

    predict_function : callable
        A callable to get predictions, i.e. `predict_function(X)`.
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
    n_repeats : int or None, default=5
        Number of times to repeat the permutation for each feature group. None means
        only one repetition.
    n_max : int or None, default=10_000
        Maximum number of observations used. If the number of observations is greater
        than `n_max`, a random subset of size `n_max` will be drawn from `X`, `y`, (and
        `weights`). Pass None for no subsampling.
    method : str, default="difference"
        Normalization method for the importance scores. The options are: "difference",
        "ratio", and "raw" (no normalization).
    smaller_is_better : bool, default=True
        If True, smaller values of the scoring_function are better.
        If False, the role of shuffled scores and base_score is reversed.
    rng : np.random.Generator, int or None, default=None
        The random number generator used for shuffling values and for subsampling
        `n_max` rows. The input is internally wrapped by `np.random.default_rng(rng)`.

    Returns
    -------
    df : polars.DataFrame
        A DataFrame with the following columns:

        - `feature`: Feature name or feature group name.
        - `importance`: Sample mean of the importance scores.
        - `standard_deviation`: Sample standard deviation of the importance scores
          (None if `n_repeats = 1`).

        The values are sorted in decreasing order of importance.

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
    >>>
    >>> # Create a synthetic dataset
    >>> rng = np.random.default_rng(1)
    >>> n = 1000
    >>>
    >>> X = pl.DataFrame(
    ...     {
    ...         "area": rng.uniform(30, 120, n),
    ...         "rooms": rng.choice([2.5, 3.5, 4.5], n),
    ...         "age": rng.uniform(0, 100, n),
    ...     }
    ... )
    >>>
    >>> y = X["area"] + 20 * X["rooms"] + rng.normal(0, 1, n)
    >>>
    >>> model = LinearRegression()
    >>> _ = model.fit(X, y)
    >>>
    >>> perm_importance = compute_permutation_importance(
    ...     predict_function=model.predict,
    ...     X=X,
    ...     y=y,
    ...     rng=1,
    ... )
    >>> perm_importance
    shape: (3, 5)
    ┌─────────┬─────────────┬────────────────────┐
    │ feature ┆ importance  ┆ standard_deviation │
    │ ---     ┆ ---         ┆ ---                │
    │ str     ┆ f64         ┆ f64                │
    ╞═════════╪═════════════╪════════════════════╡
    │ area    ┆ 1352.856052 ┆ 36.695011          │
    │ rooms   ┆ 515.038303  ┆ 19.899192          │
    │ age     ┆ 0.001373    ┆ 0.001787           │
    └─────────┴─────────────┴────────────────────┘
    >>>
    >>> # Using feature subsets
    >>> perm_importance = compute_permutation_importance(
    ...     predict_function=model.predict,
    ...     X=X,
    ...     y=y,
    ...     features=["area", "age"],
    ...     rng=1,
    ... )
    >>>
    >>> # Using feature groups
    >>> perm_importance = compute_permutation_importance(
    ...     predict_function=model.predict,
    ...     X=X,
    ...     y=y,
    ...     features={"size": ["area", "rooms"], "age": "age"},
    ...     rng=1,
    ... )
    """
    if n_repeats is None or n_repeats < 1:
        n_repeats = 1

    if method not in ("difference", "ratio", "raw"):
        msg = f"Unknown normalization method: {method}"
        raise ValueError(msg)

    # Turn features into form {"x1": ["x1"], "x2": ["x2"], "group": ["x1", "x2"]}
    # While looking verbose, it is the most flexible way to handle all cases
    if features is None:
        features = safe_column_names(X)
    if not isinstance(features, dict):
        features = {v: [v] for v in features}

    # Usually, the data is too large and we need subsampling
    n = length_of_first_dimension(X)
    rng_ = np.random.default_rng(rng)  # we need it later for shuffling
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
    if method in ("difference", "ratio"):
        base_score = scoring_function(y, predict_function(X), weights=weights)

    # Stack X per repetition
    if n_repeats >= 2:
        # Do we need to worry about pandas 1?
        X = safe_index_rows(X, np.tile(np.arange(n), n_repeats))

    # Loop over feature groups
    scores = []
    feature_groups = features.keys()

    for feature_group in feature_groups:
        shuffle_indices = np.concatenate(
            [rng_.permutation(n) for _ in range(n_repeats)]
        )
        X_shuffled = safe_shuffle_cols(X, features[feature_group], shuffle_indices)

        # Note: np.split() also works on Series and DataFrames
        predictions = predict_function(X_shuffled)
        scores_per_repetition = [
            scoring_function(y, pred, weights=weights)
            for pred in np.split(predictions, n_repeats, axis=0)
        ]
        scores.append(pl.Series(scores_per_repetition))

    # Remove base score
    if method in ("difference", "ratio"):
        direction = 1 if smaller_is_better else -1

        if method == "difference":
            scores = [direction * (s - base_score) for s in scores]
        elif method == "ratio":
            scores = [(s / base_score) ** direction for s in scores]

    # Aggregate over repetitions
    importance = pl.Series([s.mean() for s in scores])
    std = pl.Series([s.std() for s in scores]) if n_repeats >= 2 else None

    out = pl.DataFrame(
        {
            "feature": feature_groups,
            "importance": importance,
            "standard_deviation": std,
        }
    ).sort("importance", descending=True)

    return out
