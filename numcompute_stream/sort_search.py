"""Sorting, top-k (argpartition), quickselect and binary search."""

import numpy as np


def _check_numeric_1d(a, name):
    """Convert a to an array and check that it is 1-D and numeric."""
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be numeric, got dtype {a.dtype}")
    return a


def _check_int(k):
    """Raise TypeError if k is not an integer."""
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int, got {type(k).__name__}")


def stable_sort(a, axis=-1):
    """
    Sort an array along one axis, keeping equal values in their original order.

    Parameters:
        a : array_like, ndim >= 1
        axis : int, axis to sort along

    Returns:
        ndarray, same shape as a

    Raises:
        ValueError if a is 0-D.

    Complexity:
        O(n log n) time, O(n) space.
    """
    a = np.asarray(a)
    if a.ndim == 0:
        raise ValueError("a must have at least one dimension")
    return np.sort(a, axis=axis, kind="stable")


def _topk_positions(values, k, largest):
    """
    Positions of the k best values (no NaN allowed), in their original order.

    np.argpartition finds the k-th best value in O(n). Every value strictly
    better than it is chosen, and the earliest values equal to it fill the
    remaining places, so ties are broken by position.
    """
    m = values.size
    if largest:
        kth_value = values[np.argpartition(values, m - k)[m - k]]
        better = np.flatnonzero(values > kth_value)
    else:
        kth_value = values[np.argpartition(values, k - 1)[k - 1]]
        better = np.flatnonzero(values < kth_value)
    n_ties_needed = k - better.size
    ties = np.flatnonzero(values == kth_value)[:n_ties_needed]
    return np.sort(np.concatenate([better, ties]))


def topk(values, k, largest=True, return_indices=True):
    """
    Return the k largest (or smallest) values of a 1-D array.

    Parameters:
        values : array_like, shape (n,), numeric; NaN values are never selected
        k : int, 0 <= k <= number of non-NaN values
        largest : bool, pick the largest (True) or smallest (False)
        return_indices : bool, also return their positions in values

    Returns:
        top : ndarray, shape (k,), best first
        idx : ndarray of int, shape (k,) (only if return_indices)

    Raises:
        TypeError if values is not numeric or k is not an int.
        ValueError if values is not 1-D or k is out of range.

    Complexity:
        O(n + k log k) time, O(n) space.
    """
    v = _check_numeric_1d(values, "values")
    _check_int(k)
    valid_positions = np.flatnonzero(~np.isnan(v))
    valid_values = v[valid_positions]
    n_valid = valid_values.size
    if not 0 <= k <= n_valid:
        raise ValueError(f"k must be in [0, {n_valid}] (number of non-NaN values), got {k}")

    if k == 0:
        empty_values = valid_values[:0]
        empty_indices = valid_positions[:0]
        if return_indices:
            return empty_values, empty_indices
        return empty_values

    chosen = _topk_positions(valid_values, k, largest)

    # Sort only the k chosen values, best first (stable, so ties keep their order).
    key = valid_values[chosen].astype(float)
    if largest:
        key = -key
    order = np.argsort(key, kind="stable")
    idx = valid_positions[chosen[order]]

    if return_indices:
        return v[idx], idx
    return v[idx]


def quickselect(a, k):
    """
    Return the k-th smallest value without sorting the whole array.

    Parameters:
        a : array_like, shape (n,), numeric, no NaN
        k : int, 0 <= k < n

    Returns:
        scalar equal to np.sort(a)[k]

    Raises:
        TypeError if a is not numeric or k is not an int.
        ValueError if a is empty, not 1-D, contains NaN, or k is out of range.

    Notes:
        Each round picks a random pivot and splits the values into smaller,
        equal and larger groups (with vectorised comparisons). Only the group
        that contains the answer is kept for the next round.

    Complexity:
        Expected O(n) time, worst case O(n^2); O(n) space.
    """
    a = _check_numeric_1d(a, "a")
    _check_int(k)
    if a.size == 0:
        raise ValueError("a must not be empty")
    if np.isnan(a).any():
        raise ValueError("a contains NaN")
    if not 0 <= k < a.size:
        raise ValueError(f"k must be in [0, {a.size - 1}], got {k}")

    rng = np.random.default_rng()
    while True:
        pivot = a[rng.integers(a.size)]
        smaller = a[a < pivot]
        n_equal = np.count_nonzero(a == pivot)

        if k < smaller.size:
            a = smaller  # the answer is among the smaller values
        elif k < smaller.size + n_equal:
            return pivot  # the answer is the pivot itself
        else:
            k -= smaller.size + n_equal  # the answer is among the larger values
            a = a[a > pivot]


def binary_search(sorted_array, x):
    """
    Find where each query belongs in a sorted array, and whether it is there.

    Parameters:
        sorted_array : array_like, shape (n,), ascending, numeric (NaN at the end)
        x : scalar or array_like of queries

    Returns:
        index : int or ndarray of int, same shape as x
            First position i with sorted_array[i] >= x (same as np.searchsorted).
        found : bool or ndarray of bool, same shape as x
            True if x occurs in sorted_array (always False for NaN).

    Raises:
        TypeError if the inputs are not numeric.
        ValueError if sorted_array is not 1-D.

    Notes:
        All queries are searched at the same time: lo and hi are arrays, and
        every loop iteration halves the search range of each query.

    Complexity:
        O(q log n) time for q queries, O(q) space.
    """
    a = _check_numeric_1d(sorted_array, "sorted_array")
    queries = np.asarray(x)
    if queries.dtype.kind not in "biuf":
        raise TypeError(f"x must be numeric, got dtype {queries.dtype}")
    n = a.size

    lo = np.zeros(queries.shape, dtype=np.intp)
    hi = np.full(queries.shape, n, dtype=np.intp)
    while np.any(lo < hi):
        active = lo < hi
        mid = (lo + hi) // 2
        mid_value = a[np.minimum(mid, n - 1)]
        go_right = active & (mid_value < queries)  # NaN in a acts like +inf
        lo = np.where(go_right, mid + 1, lo)
        hi = np.where(active & ~go_right, mid, hi)

    # A NaN query goes after all the numbers.
    n_numbers = np.count_nonzero(~np.isnan(a))
    lo = np.where(np.isnan(queries), n_numbers, lo)

    if n > 0:
        in_range = lo < n
        found = in_range & (a[np.minimum(lo, n - 1)] == queries)
    else:
        found = np.zeros(queries.shape, dtype=bool)

    if queries.ndim == 0:
        return int(lo), bool(found)
    return lo, found
