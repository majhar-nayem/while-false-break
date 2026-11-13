"""Ranking with ties, and percentiles.

The smallest value gets rank 1. Ranks are returned as floats, because
"average" ranks can be halves and NaN values get rank NaN.
"""

import numpy as np

RANK_METHODS = ("average", "dense", "ordinal")
INTERPOLATIONS = ("linear", "lower", "higher", "midpoint")


def _check_numeric(a, name):
    """Convert a to an array and raise TypeError if it is not numeric."""
    a = np.asarray(a)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be numeric, got dtype {a.dtype}")
    return a


def _average_ranks(is_new_group):
    """
    Average rank of each sorted value, given where each group of ties starts.

    A group that starts at position `start` (0-based) and has `size` values
    covers ranks start + 1 ... start + size, whose mean is start + (size + 1) / 2.
    """
    group_id = np.cumsum(is_new_group) - 1  # which group each sorted value is in
    group_start = np.flatnonzero(is_new_group)
    group_size = np.bincount(group_id)
    group_rank = group_start + (group_size + 1) / 2
    return group_rank[group_id]


def rank(data, method="average"):
    """
    Rank the values of a 1-D array, handling ties.

    For [10, 20, 20, 30] the methods give:
        average: mean of the tied positions -> [1, 2.5, 2.5, 4]
        dense:   ties share a rank, no gaps -> [1, 2, 2, 3]
        ordinal: ties broken by position   -> [1, 2, 3, 4]

    Parameters:
        data : array_like, shape (n,), numeric
        method : "average", "dense" or "ordinal"

    Returns:
        ndarray of float, shape (n,); NaN values get rank NaN and the other
        values are ranked among themselves

    Raises:
        TypeError if data is not numeric.
        ValueError if data is not 1-D or method is unknown.

    Notes:
        After one stable sort, a new group of ties starts wherever the sorted
        value changes. The ranks are worked out per group and written back to
        the original positions.

    Complexity:
        O(n log n) time, O(n) space.
    """
    if method not in RANK_METHODS:
        raise ValueError(f"method must be 'average', 'dense' or 'ordinal', got {method!r}")
    x = _check_numeric(data, "data")
    if x.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {x.shape}")

    result = np.full(x.shape, np.nan)
    is_nan = np.isnan(x)
    values = x[~is_nan]
    n = values.size
    if n == 0:
        return result

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    is_new_group = np.concatenate([[True], sorted_values[1:] != sorted_values[:-1]])

    if method == "ordinal":
        sorted_ranks = np.arange(1, n + 1)
    elif method == "dense":
        sorted_ranks = np.cumsum(is_new_group)
    else:
        sorted_ranks = _average_ranks(is_new_group)

    # Put each rank back at the original position of its value.
    ranks = np.empty(n)
    ranks[order] = sorted_ranks
    result[~is_nan] = ranks
    return result


def percentile(data, q, interpolation="linear"):
    """
    Percentiles of the data, with q between 0 and 100.

    Parameters:
        data : array_like, numeric, any shape (all values are used); NaN is ignored
        q : float or array_like of floats in [0, 100]
        interpolation : "linear", "lower", "higher" or "midpoint"

    Returns:
        float if q is a scalar, otherwise ndarray with the same shape as q

    Raises:
        TypeError if data is not numeric.
        ValueError if q is outside [0, 100], interpolation is unknown, or
            data has no non-NaN values.

    Complexity:
        O(n log n) time for the sort, O(n) space.
    """
    if interpolation not in INTERPOLATIONS:
        raise ValueError(
            "interpolation must be 'linear', 'lower', 'higher' or 'midpoint', "
            f"got {interpolation!r}"
        )
    x = _check_numeric(data, "data").ravel().astype(float)
    x = np.sort(x[~np.isnan(x)])
    if x.size == 0:
        raise ValueError("data has no non-NaN values")
    q = np.asarray(q, dtype=float)
    if np.isnan(q).any() or np.any((q < 0) | (q > 100)):
        raise ValueError("q must be in the range [0, 100]")

    # Position of each percentile in the sorted data, and its two neighbours.
    position = q / 100 * (x.size - 1)
    lower_index = np.floor(position).astype(int)
    upper_index = np.ceil(position).astype(int)
    fraction = position - lower_index
    lower_value = x[lower_index]
    upper_value = x[upper_index]

    if interpolation == "linear":
        result = lower_value + (upper_value - lower_value) * fraction
    elif interpolation == "lower":
        result = lower_value
    elif interpolation == "higher":
        result = upper_value
    else:
        result = (lower_value + upper_value) / 2

    if result.ndim == 0:
        return float(result)
    return result
