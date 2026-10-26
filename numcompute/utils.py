"""Helpers: distances, activations, logsumexp, top-k and batching."""

import numpy as np


def _check_numeric(a, name):
    """Convert a to a float array, raising TypeError if it is not numeric."""
    a = np.asarray(a)
    if a.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be numeric, got dtype {a.dtype}")
    return a.astype(float)


def _check_point_sets(A, B):
    """Check A and B (B defaults to A) are 2-D numeric arrays with the same number of columns."""
    A = _check_numeric(A, "A")
    if B is None:
        B = A
    else:
        B = _check_numeric(B, "B")
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError(f"A and B must be 2-D, got shapes {A.shape} and {B.shape}")
    if A.shape[1] != B.shape[1]:
        raise ValueError(f"A has {A.shape[1]} columns but B has {B.shape[1]}")
    return A, B


def euclidean_distances(A, B=None):
    """
    Pairwise Euclidean distances between the rows of A and the rows of B.

    Parameters:
        A : array_like, shape (n, d)
        B : array_like, shape (m, d), optional (default: B = A)

    Returns:
        ndarray, shape (n, m); entry [i, j] is the distance from A[i] to B[j]

    Raises:
        TypeError if A or B is not numeric.
        ValueError if A or B is not 2-D or they have a different number of columns.

    Notes:
        Uses ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b, so the whole matrix comes
        from one matrix product instead of a double loop. Round-off can make the
        squared distance slightly negative, so it is clipped at 0, and when B is
        not given the diagonal (each point to itself) is set to exactly 0.

    Complexity:
        O(n * m * d) time, O(n * m) space.
    """
    B_was_given = B is not None
    A, B = _check_point_sets(A, B)

    A_squared = (A**2).sum(axis=1)[:, None]  # shape (n, 1)
    B_squared = (B**2).sum(axis=1)[None, :]  # shape (1, m)
    squared_distances = A_squared + B_squared - 2 * A @ B.T
    distances = np.sqrt(np.maximum(squared_distances, 0))
    if not B_was_given:
        np.fill_diagonal(distances, 0)  # the formula leaves ~1e-8 instead of exactly 0
    return distances


def manhattan_distances(A, B=None):
    """
    Pairwise Manhattan (L1) distances between the rows of A and the rows of B.

    Parameters:
        A : array_like, shape (n, d)
        B : array_like, shape (m, d), optional (default: B = A)

    Returns:
        ndarray, shape (n, m)

    Raises:
        TypeError if A or B is not numeric.
        ValueError if A or B is not 2-D or they have a different number of columns.

    Complexity:
        O(n * m * d) time and space (broadcasting builds an (n, m, d) array).
    """
    A, B = _check_point_sets(A, B)
    differences = A[:, None, :] - B[None, :, :]  # shape (n, m, d)
    return np.abs(differences).sum(axis=2)


def sigmoid(x):
    """
    Logistic function 1 / (1 + e^-x), safe for very large or small x.

    Parameters:
        x : array_like, numeric

    Returns:
        ndarray, same shape as x, values in [0, 1]

    Raises:
        TypeError if x is not numeric.

    Notes:
        For x < 0 we use e^x / (1 + e^x) instead, so e^-x never overflows.

    Complexity:
        O(n) time and space.
    """
    x = _check_numeric(x, "x")
    exp_neg_abs = np.exp(-np.abs(x))  # always in (0, 1], never overflows
    positive_branch = 1 / (1 + exp_neg_abs)  # = 1 / (1 + e^-x) for x >= 0
    negative_branch = exp_neg_abs / (1 + exp_neg_abs)  # = e^x / (1 + e^x) for x < 0
    return np.where(x >= 0, positive_branch, negative_branch)


def relu(x):
    """
    Rectified linear unit: max(0, x).

    Parameters:
        x : array_like, numeric

    Returns:
        ndarray, same shape as x

    Raises:
        TypeError if x is not numeric.

    Complexity:
        O(n) time and space.
    """
    return np.maximum(_check_numeric(x, "x"), 0)


def logsumexp(x, axis=None):
    """
    Compute log(sum(exp(x))) without overflow.

    Parameters:
        x : array_like, numeric
        axis : int or None, axis to sum over (default: all values)

    Returns:
        float if axis is None, otherwise ndarray with that axis removed

    Raises:
        TypeError if x is not numeric.
        ValueError if x is empty.

    Notes:
        Uses log(sum(exp(x))) = m + log(sum(exp(x - m))) with m = max(x), so the
        largest term is exp(0) = 1 and nothing overflows.

    Complexity:
        O(n) time and space.
    """
    x = _check_numeric(x, "x")
    if x.size == 0:
        raise ValueError("x is empty")
    shift = np.max(x, axis=axis, keepdims=True)
    shift = np.where(np.isfinite(shift), shift, 0)  # shifting by +-inf would give NaN
    with np.errstate(divide="ignore"):  # log(0) = -inf when every value is -inf
        sum_exp = np.sum(np.exp(x - shift), axis=axis, keepdims=True)
        result = np.log(sum_exp) + shift

    if axis is None:
        result = result.squeeze()
    else:
        result = result.squeeze(axis=axis)
    if result.ndim == 0:
        return float(result)
    return result


def softmax(x, axis=-1):
    """
    Softmax along an axis: exp(x) / sum(exp(x)), numerically stable.

    Parameters:
        x : array_like, numeric, ndim >= 1
        axis : int, axis that should sum to 1 (default: last)

    Returns:
        ndarray, same shape as x, values in [0, 1] summing to 1 along axis

    Raises:
        TypeError if x is not numeric.
        ValueError if x is 0-D.

    Notes:
        The max is subtracted first (softmax does not change when every value is
        shifted by the same amount), so exp never overflows.

    Complexity:
        O(n) time and space.
    """
    x = _check_numeric(x, "x")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=axis, keepdims=True)


def argtopk(x, k, axis=-1, largest=True):
    """
    Indices of the k largest (or smallest) values along an axis, best first.

    Parameters:
        x : array_like, numeric, ndim >= 1
        k : int, 1 <= k <= size of x along axis
        axis : int, axis to search along (default: last)
        largest : bool, largest (True) or smallest (False)

    Returns:
        ndarray of int, same shape as x except size k along axis

    Raises:
        TypeError if x is not numeric or k is not an int.
        ValueError if x is 0-D or k is out of range.

    Notes:
        np.argpartition picks the k candidates in O(n), then only those k are
        sorted. Use sort_search.topk for 1-D data with NaN and tie rules.

    Complexity:
        O(n + k log k) time per slice, O(n) space.
    """
    x = _check_numeric(x, "x")
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    if not isinstance(k, (int, np.integer)):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    n = x.shape[axis]
    if not 1 <= k <= n:
        raise ValueError(f"k must be in [1, {n}], got {k}")
    # Negate for "largest" so that, in both cases, smaller key = better.
    if largest:
        key = -x
    else:
        key = x

    candidates = np.argpartition(key, k - 1, axis=axis)
    candidates = np.take(candidates, np.arange(k), axis=axis)  # the k best, unordered
    candidate_keys = np.take_along_axis(key, candidates, axis=axis)
    order = np.argsort(candidate_keys, axis=axis, kind="stable")
    return np.take_along_axis(candidates, order, axis=axis)


def batch_iterator(X, batch_size, shuffle=False, seed=None):
    """
    Yield the rows of X in mini-batches.

    Parameters:
        X : array_like, shape (n, ...)
        batch_size : int >= 1
        shuffle : bool, shuffle the row order first
        seed : int, optional, random seed used when shuffle=True

    Yields:
        ndarray, shape (<= batch_size, ...); the last batch can be smaller

    Raises:
        ValueError if batch_size < 1 or X is 0-D.

    Complexity:
        O(n) time in total; each batch is a copy of at most batch_size rows.
    """
    X = np.asarray(X)
    if X.ndim == 0:
        raise ValueError("X must have at least one dimension")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
    n_rows = len(X)
    if shuffle:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_rows)
    else:
        order = np.arange(n_rows)
    for start in range(0, n_rows, batch_size):
        batch_rows = order[start : start + batch_size]
        yield X[batch_rows]
