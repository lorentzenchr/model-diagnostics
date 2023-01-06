from typing import Optional

import numpy as np
import numpy.typing as npt
import pyarrow as pa
import pyarrow.compute as pc
from scipy import special

from .._utils.array import (
    array_name,
    get_second_dimension,
    length_of_second_dimension,
    validate_2_arrays,
    validate_same_first_dimension,
)


def identification_function(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    *,
    functional: str = "mean",
    level: float = 0.5,
) -> np.ndarray:
    r"""Canonical identification function.

    Identification functions act as generalised residuals. See Notes for further
    details.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs)
        Predicted values of the `functional`, e.g. the conditional expectation of
        the response, `E(Y|X)`.
    functional : str
        The functional that is induced by the identification function `V`. Options are:
        - `"mean"`. Argument `level` is neglected.
        - `"median"`. Argument `level` is neglected.
        - `"expectile"`
        - `"quantile"`
    level : float
        The level of the expectile of quantile. (Often called \(\alpha\).)
        It must be `0 <= level <= 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.

    Returns
    -------
    V : ndarray of shape (n_obs)
        Values of the identification function.

    Notes
    -----
    The function \(V(y, z)\) for observation \(y=y_{pred}\) and prediction
    \(z=y_{pred}\) is a strict identification function for the functional \(T\), or
    induces the functional \(T\) as:

    \[
    E[V(Y, z)] = 0\quad \Rightarrow\quad z=T(F) \quad \forall \text{ distributions } F
    \]

    Functional \(T\) can be the mean, median, an expectile or a quantile.

    | functional | strict identification function \(V(y, z)\)           |
    | ---------- | ---------------------------------------------------- |
    | mean       | \(z - y\)                                            |
    | median     | \(\mathbf{1}\{z \ge y\} - \frac{1}{2}\)              |
    | expectile  | \(2 \mid\mathbf{1}\{z \ge y\} - \alpha\mid (z - y)\) |
    | quantile   | \(\mathbf{1}\{z \ge y\} - \alpha\)                   |

    For `level` \(\alpha\).

    References
    ----------

    Examples
    --------

    """
    y_o: np.ndarray
    y_p: np.ndarray
    y_o, y_p = validate_2_arrays(y_obs, y_pred)

    if functional in ("expectile", "quantile") and (level < 0 or level > 1):
        raise ValueError(f"Argument level must fulfil 0 <= level <= 1, got {level}.")

    if functional == "mean":
        return y_p - y_o
    elif functional == "median":
        return np.greater_equal(y_p, y_o) - 0.5
    elif functional == "expectile":
        return 2 * np.abs(np.greater_equal(y_p, y_o) - level) * (y_p - y_o)
    elif functional == "quantile":
        return np.greater_equal(y_p, y_o) - level
    else:
        allowed_functionals = ("mean", "median", "expectile", "quantile")
        raise ValueError(
            f"Argument functional must be one of {allowed_functionals}, got "
            f"{functional}."
        )


def compute_bias(
    y_obs: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    feature: Optional[npt.ArrayLike] = None,
    *,
    functional: str = "mean",
    level: float = 0.5,
    n_bins: int = 10,
):
    r"""Compute generalised bias conditional on a feature.

    This function computes and aggregates the generalised bias, i.e. the values of the
    canonical identification function, versus (grouped by) a feature.
    This is a good way to assess whether a model is conditionally calibrated or not.
    Well calibrated models have bias terms around zero.
    For the mean functional, the generalised bias is the negative residual
    `y_pred - y_obs`.
    See Notes for further details.

    Parameters
    ----------
    y_obs : array-like of shape (n_obs)
        Observed values of the response variable.
        For binary classification, y_obs is expected to be in the interval [0, 1].
    y_pred : array-like of shape (n_obs) or (n_obs, n_models)
        Predicted values of the conditional expectation of Y, \(E(Y|X)\).
    feature : array-like of shape (n_obs) or None
        Some feature column.
    functional : str
        The functional that is induced by the identification function `V`. Options are:
        - `"mean"`. Argument `level` is neglected.
        - `"median"`. Argument `level` is neglected.
        - `"expectile"`
        - `"quantile"`
    level : float
        The level of the expectile of quantile. (Often called \(\alpha\).)
        It must be `0 <= level <= 1`.
        `level=0.5` and `functional="expectile"` gives the mean.
        `level=0.5` and `functional="quantile"` gives the median.
    n_bins : int
        The number of bins for numerical features and the maximal number of (most
        frequent) categories shown for categorical features.

    Returns
    -------
    df : pyarrow Table
        The result table contains at least the columns:

        - `bias_mean`: Mean of the bias
        - `bias_cout`: Number of data rows
        - `bias_stderr`: Standard error, i.e. standard deviation of `bias_mean`

    Notes
    -----
    A model \(m(X)\) is conditionally calibrated iff \(E(V(m(X), Y))=0\) a.s. with
    canonical identification function \(V\). The empirical version, given some data,
    reads \(\frac{1}{n}\sum_i V(m(x_i), y_i)\).
    This generalises the classical residual (up to a minus sign) for target functionals
    other than the mean. See `[FLM2022]`.

    References
    ----------
    `[FLM2022]`

    :   T. Fissler, C. Lorentzen, and M. Mayer.
        "Model Comparison and Calibration Assessment". (2022)
        [arxiv:https://arxiv.org/abs/2202.12780](https://arxiv.org/abs/2202.12780).
    """
    validate_same_first_dimension(y_obs, y_pred)
    if feature is not None:
        validate_same_first_dimension(y_obs, feature)
    n_pred = length_of_second_dimension(y_pred)
    if n_pred == 0:
        pred_names = [""]
    else:
        pred_names = []
        for i in range(n_pred):
            x = get_second_dimension(y_pred, i)
            pred_names.append(array_name(x, default=str(i)))

    df_list = []
    for i in range(len(pred_names)):
        # Loop over columns of y_pred.
        if n_pred == 0:
            x = y_pred
        else:
            x = get_second_dimension(y_pred, i)

        bias = identification_function(
            y_obs=y_obs,
            y_pred=x,
            functional=functional,
            level=level,
        )

        if feature is None:
            feature_name = None
            df = pa.table(
                {
                    "bias_mean": [np.mean(bias)],
                    "bias_count": [bias.shape[0]],
                    "bias_stddev": [np.std(bias, ddof=1)],
                }
            )
        else:
            feature_name = array_name(feature, default="feature")
            df = pa.table(
                {
                    "y_obs": y_obs,
                    "y_pred": x,
                    feature_name: feature,
                    "bias": bias,
                }
            )

            is_categorical = False
            # is_numeric = False
            is_string = False
            if pa.types.is_dictionary(df.column(feature_name).type):
                is_categorical = True
            elif pa.types.is_string(df.column(feature_name).type):
                # We could convert strings to categoricals.
                is_string = True
            # else:
            #     is_numeric = True

            agg = (
                ("bias", "mean"),
                ("bias", "count"),
                ("bias", "stddev", pc.VarianceOptions(ddof=1)),
            )
            if is_categorical or is_string:
                df = df.group_by([feature_name]).aggregate([*agg])
                n_bins = min(n_bins, df.num_rows)
                df = df.sort_by("bias_count").take(np.arange(n_bins))
                if is_categorical:
                    # Pyarrow does not yet support sorting dictionary type arrays, see
                    # https://issues.apache.org/jira/browse/ARROW-14314
                    # We resort to pandas instead.
                    df = df.to_pandas().sort_values(feature_name)
                    df = pa.Table.from_pandas(df)
                else:
                    df = df.sort_by(feature_name)
            else:
                # binning
                q = np.quantile(
                    feature,
                    q=np.linspace(0 + 1 / n_bins, 1 - 1 / n_bins, n_bins - 1),
                    method="lower",  # "linear" would not reduce with np.unique below
                )  # type: ignore
                q = np.unique(q)
                # bins[i-1] < x <= bins[i]
                f_binned = np.digitize(feature, bins=q, right=True)  # type: ignore
                df = df.append_column("bin", pa.array(f_binned))
                df = (
                    df.group_by(["bin"])
                    .aggregate(
                        [
                            *agg,
                            (feature_name, "mean"),
                        ]
                    )
                    .sort_by("bin")
                    .drop(["bin"])
                )
                cnames = df.column_names
                cnames[-1] = feature_name
                df = df.rename_columns(cnames)

        # Add column standard error.
        df = df.append_column(
            "bias_stderr",
            pc.divide(df.column("bias_stddev"), pc.sqrt(df.column("bias_count"))),
        )

        # Add column with p-value of 2-sided t-test.
        s_ = df.column("bias_stddev").to_numpy()
        p_value = np.full_like(s_, fill_value=np.nan)
        mask: npt.ArrayLike = s_ > 0
        x = df.column("bias_mean").to_numpy()[mask]
        n = df.column("bias_count").to_numpy()[mask]
        s = s_[mask]
        # t-statistic t (-|t| and factor of 2 because of 2-sided test)
        p_value[mask] = 2 * special.stdtr(
            n - 1,  # degrees of freedom
            -np.abs(x / s * np.sqrt(n)),
        )
        df = df.append_column("p_value", pa.array(p_value))

        # Add column "model".
        if n_pred > 0:
            if feature_name == "model":
                model_col_name = "model_"
            else:
                model_col_name = "model"
            df = df.append_column(
                model_col_name, pa.array([pred_names[i]] * df.num_rows)
            )

        # Select the columns in the correct order.
        col_selection = []
        if n_pred > 0:
            col_selection.append(model_col_name)
        if feature_name is not None and feature_name in df.column_names:
            col_selection.append(feature_name)
        col_selection += ["bias_mean", "bias_count", "bias_stderr", "p_value"]
        df_list.append(df.select(col_selection))

    return pa.concat_tables(df_list)
