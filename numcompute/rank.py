"""Ranking with ties, and percentiles.

Ranks are built such that smallest ones get rank 1 and returned as float,
because "average" ranks can be halves and NaN values get rank NaN.
"""

import numpy as np


def _check_numeric(a, name):
    a = np.asarray(a)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be numeric, got dtype {a.dtype}")
    return a


def rank(data, method="average"):
    """Rank the values of a 1-D array, handling ties.

    For [10, 20, 20, 30] the methods give:
        average: mean of the tied positions -> [1, 2.5, 2.5, 4]
        dense:   ties share a rank, no gaps -> [1, 2, 2, 3]
        ordinal: ties broken by position   -> [1, 2, 3, 4]

    Parameters:
        data: array_like, shape (n,), numeric
        method: "average", "dense" or "ordinal"

    Returns:
        ndarray of float, shape (n,); NaN values get rank NaN and the other
        values are ranked among themselves

    Exceptions:
        TypeError: if data is not numeric
        ValueError: if data is not 1-D or method is unknown

    After one stable sort, a new group of ties starts wherever the sorted
    value changes. The ranks are worked out per group and written back to
    the original positions.
    Complexity: O(n log n) time, O(n) space.
    """
    if method not in ("average", "dense", "ordinal"):
        raise ValueError(f"method must be 'average', 'dense' or 'ordinal', got {method!r}")
    x = _check_numeric(data, "data")
    if x.ndim != 1:
        raise ValueError(f"data must be 1-D, got shape {x.shape}")

    out = np.full(x.shape, np.nan)
    is_nan = np.isnan(x)
    vals = x[~is_nan]
    n = vals.size
    if n == 0:
        return out

    order = np.argsort(vals, kind="stable")
    s = vals[order]
    new_group = np.concatenate([[True], s[1:] != s[:-1]])

    if method == "ordinal":
        sorted_ranks = np.arange(1, n + 1)
    elif method == "dense":
        sorted_ranks = np.cumsum(new_group)
    else:
        group = np.cumsum(new_group) - 1          
        start = np.flatnonzero(new_group)        
        size = np.bincount(group)                 
        sorted_ranks = (start + (size + 1) / 2)[group]

    ranks = np.empty(n)
    ranks[order] = sorted_ranks
    out[~is_nan] = ranks
    return out


def percentile(data, q, interpolation="linear"):
    """Percentiles of the data, with q between 0 and 100.
    
    Parameters:
        data: array_like, numeric, any shape (all values are used); NaN is ignored
        q: float or array_like of floats in [0, 100]
        interpolation: "linear", "lower", "higher" or "midpoint"

    Returns:
        float if q is a scalar, otherwise ndarray with the same shape as q

    Exceptions:
        TypeError: if data is not numeric
        ValueError: if q is outside [0, 100], interpolation is unknown, or
            data has no non-NaN values

    Complexity: O(n log n) time for the sort, O(n) space.
    """
    if interpolation not in ("linear", "lower", "higher", "midpoint"):
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

    pos = q / 100 * (x.size - 1)
    i = np.floor(pos).astype(int)
    j = np.ceil(pos).astype(int)
    frac = pos - i

    if interpolation == "linear":
        out = x[i] + (x[j] - x[i]) * frac
    elif interpolation == "lower":
        out = x[i]
    elif interpolation == "higher":
        out = x[j]
    else:
        out = (x[i] + x[j]) / 2

    return float(out) if out.ndim == 0 else out
