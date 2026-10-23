#Sorting, top-k (argpartition), quickselect and binary search.

import numpy as np


def _check_numeric_1d(a, name):
    a = np.asarray(a)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be numeric, got dtype {a.dtype}")
    return a


def stable_sort(a, axis=-1):
    """Stable sort along an axis: equal values keep their original order.

    Parameters
    ----------
    a : array_like, ndim >= 1
    axis : int, axis to sort along

    Returns
    -------
    ndarray, same shape as a

    Raises
    ------
    ValueError if a is 0-D.

    Complexity: O(n log n) time, O(n) space.
    """
    a = np.asarray(a)
    if a.ndim == 0:
        raise ValueError("a must have at least one dimension")
    return np.sort(a, axis=axis, kind="stable")


def multikey_sort(X, keys):
    """Sort the rows of a 2-D table by several columns, primary key first.

    Parameters
    ----------
    X : array_like, shape (n, d)
    keys : list of int, column indices, primary key first

    Returns
    -------
    ndarray, shape (n, d), rows sorted ascending by the keys

    Raises
    ------
    ValueError if X is not 2-D or a key is out of range.

    np.lexsort sorts by the LAST key first, so the keys are reversed.
    Complexity: O(k * n log n) time for k keys, O(n) extra space.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}")
    for k in keys:
        if not 0 <= k < X.shape[1]:
            raise ValueError(f"key column {k} out of range for {X.shape[1]} columns")
    order = np.lexsort([X[:, k] for k in reversed(keys)])
    return X[order]


def topk(values, k, largest=True, return_indices=True):
    """Return the k largest (or smallest) values of a 1-D array, best first.

    Parameters
    ----------
    values : array_like, shape (n,), numeric; NaN values are never selected
    k : int, 0 <= k <= number of non-NaN values
    largest : bool, pick the largest (True) or smallest (False)
    return_indices : bool, also return their positions in values

    Returns
    -------
    top : ndarray, shape (k,)
    idx : ndarray of int, shape (k,) (only if return_indices)

    Raises
    ------
    TypeError if values is not numeric or k is not an int.
    ValueError if values is not 1-D or k is out of range.

    np.argpartition finds the k-th best value in O(n) without a full sort.
    Ties are deterministic: among equal values the lower index wins.
    Complexity: O(n + k log k) time, O(n) space.
    """
    v = _check_numeric_1d(values, "values")
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    valid = np.flatnonzero(~np.isnan(v))  # positions of non-NaN values
    w = v[valid]
    m = w.size
    if not 0 <= k <= m:
        raise ValueError(f"k must be in [0, {m}] (number of non-NaN values), got {k}")
    if k == 0:
        return (w[:0], valid[:0]) if return_indices else w[:0]

    if largest:
        kth = w[np.argpartition(w, m - k)[m - k]]  # k-th largest value
        better = np.flatnonzero(w > kth)
    else:
        kth = w[np.argpartition(w, k - 1)[k - 1]]  # k-th smallest value
        better = np.flatnonzero(w < kth)
    ties = np.flatnonzero(w == kth)[: k - better.size]  # earliest ties fill the rest
    chosen = np.sort(np.concatenate([better, ties]))

    key = w[chosen].astype(float)
    order = np.argsort(-key if largest else key, kind="stable")
    idx = valid[chosen[order]]
    return (v[idx], idx) if return_indices else v[idx]


def quickselect(a, k):
    """Return the k-th smallest value (0-based) without sorting the whole array.

    Educational version of Hoare's quickselect: pick a random pivot, split the
    values into < pivot, == pivot and > pivot with boolean masks, and keep
    only the part that contains rank k. The three-way split means duplicates
    (or all-equal input) do not slow it down.

    Parameters
    ----------
    a : array_like, shape (n,), numeric, no NaN
    k : int, 0 <= k < n

    Returns
    -------
    scalar equal to np.sort(a)[k]

    Raises
    ------
    TypeError if a is not numeric or k is not an int.
    ValueError if a is empty, not 1-D, contains NaN, or k is out of range.

    The while loop runs over rounds (about log n), not over elements.
    Complexity: expected O(n) time, worst case O(n^2); O(n) space.
    """
    a = _check_numeric_1d(a, "a")
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if a.size == 0:
        raise ValueError("a must not be empty")
    if np.isnan(a).any():
        raise ValueError("a contains NaN")
    if not 0 <= k < a.size:
        raise ValueError(f"k must be in [0, {a.size - 1}], got {k}")

    rng = np.random.default_rng()
    while True:
        pivot = a[rng.integers(a.size)]
        lower = a[a < pivot]
        n_equal = np.count_nonzero(a == pivot)
        if k < lower.size:
            a = lower
        elif k < lower.size + n_equal:
            return pivot
        else:
            k -= lower.size + n_equal
            a = a[a > pivot]


def binary_search(sorted_array, x):
    """Find where x belongs in an ascending array, and whether it is there.

    All queries are searched at the same time: every loop iteration halves
    each query's [lo, hi) interval using NumPy masks.

    Parameters
    ----------
    sorted_array : array_like, shape (n,), ascending, numeric (NaN at the end)
    x : scalar or array_like of queries

    Returns
    -------
    index : int or ndarray of int, same shape as x
        First position i with sorted_array[i] >= x (same as np.searchsorted).
    found : bool or ndarray of bool, same shape as x
        True if x occurs in sorted_array (always False for NaN).

    Raises
    ------
    TypeError if the inputs are not numeric.
    ValueError if sorted_array is not 1-D.

    The array must already be sorted; this is not checked (that would cost O(n)).
    Complexity: O(q log n) time for q queries, O(q) space.
    """
    a = _check_numeric_1d(sorted_array, "sorted_array")
    q = np.asarray(x)
    if q.dtype.kind not in "biuf":
        raise TypeError(f"x must be numeric, got dtype {q.dtype}")
    n = a.size

    lo = np.zeros(q.shape, dtype=np.intp)
    hi = np.full(q.shape, n, dtype=np.intp)
    while np.any(lo < hi):
        active = lo < hi
        mid = (lo + hi) // 2
        go_right = active & (a[np.minimum(mid, n - 1)] < q)  # NaN in a acts like +inf
        lo = np.where(go_right, mid + 1, lo)
        hi = np.where(active & ~go_right, mid, hi)

    lo = np.where(np.isnan(q), np.count_nonzero(~np.isnan(a)), lo)  # NaN goes after numbers
    found = (lo < n) & (a[np.minimum(lo, n - 1)] == q) if n > 0 else np.zeros(q.shape, dtype=bool)

    if q.ndim == 0:
        return int(lo), bool(found)
    return lo, found
