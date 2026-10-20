"""Shared numerical helpers: validation, distances, activations, top-k, batching.

Every other module imports its input validation from here so that error types
and messages are identical across the package (see CONTRIBUTING.md):

* ``TypeError``  - wrong type (e.g. strings where numbers are expected)
* ``ValueError`` - wrong shape / ndim / length, or an invalid parameter value
* :class:`NotFittedError` - ``transform``/``predict`` called before ``fit``
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np

__all__ = [
    "NotFittedError",
    "NAN_POLICIES",
    "check_array",
    "check_nan_policy",
    "check_consistent_length",
    "euclidean_distances",
    "manhattan_distances",
    "cosine_similarity",
    "sigmoid",
    "relu",
    "softmax",
    "log_softmax",
    "logsumexp",
    "argtopk",
    "batch_iterator",
]

NAN_POLICIES = ("propagate", "omit", "raise")


class NotFittedError(RuntimeError):
    """Raised when ``transform``/``predict`` is called on an unfitted object."""


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------
def check_array(
    X,
    *,
    name: str = "X",
    ndim: int | tuple[int, ...] | None = None,
    numeric: bool = True,
    to_float: bool = False,
    allow_empty: bool = True,
    allow_nan: bool = True,
) -> np.ndarray:
    """Convert ``X`` to an ndarray and validate it.

    Parameters
    ----------
    X : array_like
        Input data. Never modified in place.
    name : str, default="X"
        Argument name used in error messages.
    ndim : int or tuple of int, optional
        Allowed number of dimensions. ``None`` accepts any.
    numeric : bool, default=True
        If True the result must have a boolean, integer or floating dtype.
        Object arrays holding numbers are converted to ``float64``.
    to_float : bool, default=False
        If True the result is cast to ``float64``.
    allow_empty : bool, default=True
        If False an array with ``size == 0`` raises ``ValueError``.
    allow_nan : bool, default=True
        If False an array containing NaN raises ``ValueError``.

    Returns
    -------
    arr : ndarray
        Validated array (a view of ``X`` when no conversion was needed).

    Raises
    ------
    TypeError
        If ``numeric`` and the data is not numeric.
    ValueError
        If the dimensionality, emptiness or NaN checks fail.

    Notes
    -----
    Complexity: O(n) time for the NaN check / casts, O(1) extra space otherwise.
    """
    arr = np.asarray(X)
    if numeric:
        if arr.dtype.kind == "O":
            try:
                arr = arr.astype(np.float64)
            except (TypeError, ValueError):
                raise TypeError(
                    f"{name} must be numeric; got an object array with "
                    "non-numeric entries"
                ) from None
        elif arr.dtype.kind not in "biuf":
            raise TypeError(f"{name} must be numeric, got dtype {arr.dtype}")
    if to_float and arr.dtype != np.float64:
        arr = arr.astype(np.float64)
    if ndim is not None:
        allowed = (ndim,) if isinstance(ndim, int) else tuple(ndim)
        if arr.ndim not in allowed:
            want = " or ".join(f"{d}-D" for d in allowed)
            hint = ""
            if 2 in allowed and arr.ndim == 1:
                hint = "; use x.reshape(-1, 1) for a single feature"
            raise ValueError(
                f"{name} must be {want}, got {arr.ndim}-D array with shape "
                f"{arr.shape}{hint}"
            )
    if not allow_empty and arr.size == 0:
        raise ValueError(f"{name} is empty (shape {arr.shape})")
    if not allow_nan and arr.dtype.kind == "f" and np.isnan(arr).any():
        raise ValueError(f"{name} contains NaN")
    return arr


def check_nan_policy(nan_policy: str) -> str:
    """Validate a ``nan_policy`` argument.

    Parameters
    ----------
    nan_policy : {"propagate", "omit", "raise"}
        The policy to check.

    Returns
    -------
    str
        The same policy.

    Raises
    ------
    ValueError
        If the policy is not one of the allowed values.

    Notes
    -----
    Complexity: O(1).
    """
    if nan_policy not in NAN_POLICIES:
        raise ValueError(
            f"nan_policy must be one of {NAN_POLICIES}, got {nan_policy!r}"
        )
    return nan_policy


def check_consistent_length(*arrays: np.ndarray, names: tuple[str, ...] = ()) -> int:
    """Check that all arrays share the same first dimension.

    Parameters
    ----------
    *arrays : ndarray
        Arrays to compare (``len(a)`` is used).
    names : tuple of str, optional
        Argument names for the error message.

    Returns
    -------
    int
        The common length.

    Raises
    ------
    ValueError
        If the lengths differ.

    Notes
    -----
    Complexity: O(k) for k arrays.
    """
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        labels = names or tuple(f"array {i}" for i in range(len(arrays)))
        detail = ", ".join(f"{n}={ln}" for n, ln in zip(labels, lengths, strict=True))
        raise ValueError(f"inconsistent number of samples: {detail}")
    return lengths[0] if lengths else 0


def _pair(A, B, name_a: str = "A", name_b: str = "B"):
    A = check_array(A, name=name_a, ndim=2, to_float=True)
    B = A if B is None else check_array(B, name=name_b, ndim=2, to_float=True)
    if A.shape[1] != B.shape[1]:
        raise ValueError(
            f"{name_a} and {name_b} must have the same number of features, "
            f"got {A.shape[1]} and {B.shape[1]}"
        )
    return A, B


# --------------------------------------------------------------------------
# Distances
# --------------------------------------------------------------------------
def euclidean_distances(A, B=None, *, squared: bool = False) -> np.ndarray:
    """Pairwise Euclidean distances between the rows of ``A`` and ``B``.

    Uses the expansion ``||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b`` so the work
    is a single matrix product. Round-off can make tiny squared distances
    negative, so they are clipped at 0.

    Parameters
    ----------
    A : array_like, shape (n, d)
    B : array_like, shape (m, d), optional
        Defaults to ``A``.
    squared : bool, default=False
        Return squared distances.

    Returns
    -------
    D : ndarray, shape (n, m)

    Raises
    ------
    ValueError
        If the inputs are not 2-D or the feature counts differ.

    Notes
    -----
    Complexity: O(n m d) time, O(n m) space.
    """
    same = B is None
    A, B = _pair(A, B)
    a2 = np.einsum("ij,ij->i", A, A)[:, None]
    b2 = np.einsum("ij,ij->i", B, B)[None, :]
    D = a2 + b2 - 2.0 * (A @ B.T)
    np.maximum(D, 0.0, out=D)
    if same:
        np.fill_diagonal(D, 0.0)
    return D if squared else np.sqrt(D)


def manhattan_distances(A, B=None) -> np.ndarray:
    """Pairwise L1 (city-block) distances between rows of ``A`` and ``B``.

    Parameters
    ----------
    A : array_like, shape (n, d)
    B : array_like, shape (m, d), optional
        Defaults to ``A``.

    Returns
    -------
    D : ndarray, shape (n, m)

    Raises
    ------
    ValueError
        If the inputs are not 2-D or the feature counts differ.

    Notes
    -----
    Complexity: O(n m d) time and O(n m d) temporary space (broadcasting).
    """
    A, B = _pair(A, B)
    return np.abs(A[:, None, :] - B[None, :, :]).sum(axis=-1)


def cosine_similarity(A, B=None, *, eps: float = 1e-12) -> np.ndarray:
    """Pairwise cosine similarity between rows of ``A`` and ``B``.

    Parameters
    ----------
    A : array_like, shape (n, d)
    B : array_like, shape (m, d), optional
        Defaults to ``A``.
    eps : float, default=1e-12
        Norms below ``eps`` are replaced by ``eps``; a zero vector therefore
        has similarity 0 with everything instead of producing NaN.

    Returns
    -------
    S : ndarray, shape (n, m)
        Values in ``[-1, 1]``.

    Raises
    ------
    ValueError
        If the inputs are not 2-D or the feature counts differ.

    Notes
    -----
    Complexity: O(n m d) time, O(n m) space.
    """
    A, B = _pair(A, B)
    na = np.maximum(np.linalg.norm(A, axis=1, keepdims=True), eps)
    nb = np.maximum(np.linalg.norm(B, axis=1, keepdims=True), eps)
    return np.clip((A / na) @ (B / nb).T, -1.0, 1.0)


# --------------------------------------------------------------------------
# Activations (numerically stable)
# --------------------------------------------------------------------------
def sigmoid(x) -> np.ndarray:
    """Logistic sigmoid ``1 / (1 + exp(-x))`` without overflow.

    For ``x >= 0`` it evaluates ``1 / (1 + exp(-x))``; for ``x < 0`` the
    algebraically equal ``exp(x) / (1 + exp(x))``. ``exp`` is therefore only
    ever called on non-positive numbers and cannot overflow.

    Parameters
    ----------
    x : array_like
        Any shape.

    Returns
    -------
    ndarray
        Same shape as ``x``, values in ``[0, 1]``.

    Notes
    -----
    Complexity: O(n) time and space.
    """
    x = check_array(x, name="x", to_float=True)
    z = np.exp(-np.abs(x))  # in (0, 1], never overflows
    return np.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))


def relu(x) -> np.ndarray:
    """Rectified linear unit ``max(x, 0)``.

    Parameters
    ----------
    x : array_like
        Any shape.

    Returns
    -------
    ndarray
        Same shape as ``x``.

    Notes
    -----
    Complexity: O(n) time and space.
    """
    x = check_array(x, name="x", to_float=True)
    return np.maximum(x, 0.0)


def logsumexp(x, axis: int | None = None, keepdims: bool = False) -> np.ndarray:
    """Compute ``log(sum(exp(x)))`` stably.

    Uses ``m + log(sum(exp(x - m)))`` with ``m = max(x)``, so the largest
    exponent is ``exp(0) = 1`` and nothing overflows. Slices that are all
    ``-inf`` return ``-inf``; any ``+inf`` returns ``+inf``; NaN propagates.

    Parameters
    ----------
    x : array_like
        Input values.
    axis : int, optional
        Axis to reduce. ``None`` reduces over all elements.
    keepdims : bool, default=False
        Keep the reduced axis with length 1.

    Returns
    -------
    ndarray or float
        Shape of ``x`` with ``axis`` removed (or kept as 1).

    Raises
    ------
    ValueError
        If ``x`` (or the reduced axis) is empty.

    Notes
    -----
    Complexity: O(n) time and space.
    """
    x = check_array(x, name="x", to_float=True)
    if x.size == 0 or (axis is not None and x.shape[axis] == 0):
        raise ValueError("logsumexp of an empty array is undefined")
    m = np.max(x, axis=axis, keepdims=True)
    # If the max is +-inf (or NaN), shifting by it would create inf - inf.
    m_safe = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):  # log(0) -> -inf for all -inf slices
        out = np.log(np.sum(np.exp(x - m_safe), axis=axis, keepdims=True)) + m_safe
    if not keepdims:
        out = np.squeeze(out, axis=axis) if axis is not None else out.reshape(())
    return out[()] if out.ndim == 0 else out


def softmax(x, axis: int = -1) -> np.ndarray:
    """Max-shifted softmax ``exp(x - max) / sum(exp(x - max))``.

    Parameters
    ----------
    x : array_like
        Logits.
    axis : int, default=-1
        Axis along which the probabilities sum to 1.

    Returns
    -------
    ndarray
        Same shape as ``x``; each slice along ``axis`` sums to 1.

    Raises
    ------
    ValueError
        If ``x`` is empty.

    Notes
    -----
    Shifting by the maximum leaves the result unchanged but bounds every
    exponent by 0, so logits like ``1000`` do not overflow.
    Complexity: O(n) time and space.
    """
    return np.exp(log_softmax(x, axis=axis))


def log_softmax(x, axis: int = -1) -> np.ndarray:
    """Stable ``log(softmax(x))`` computed as ``x - logsumexp(x)``.

    Parameters
    ----------
    x : array_like
        Logits.
    axis : int, default=-1
        Normalisation axis.

    Returns
    -------
    ndarray
        Same shape as ``x``.

    Raises
    ------
    ValueError
        If ``x`` is empty.

    Notes
    -----
    Complexity: O(n) time and space.
    """
    x = check_array(x, name="x", to_float=True)
    return x - logsumexp(x, axis=axis, keepdims=True)


# --------------------------------------------------------------------------
# Top-k
# --------------------------------------------------------------------------
def argtopk(x, k: int, axis: int = -1, largest: bool = True) -> np.ndarray:
    """Return indices of the ``k`` largest (or smallest) entries along an axis.

    A batched helper (e.g. top-k classes per row of a score matrix). Results
    are ordered best-first. For exact, deterministic tie-breaking on 1-D data
    use :func:`numcompute.sort_search.topk`.

    Parameters
    ----------
    x : array_like
        Numeric input, NaN not allowed.
    k : int
        ``0 <= k <= x.shape[axis]``.
    axis : int, default=-1
        Axis to select along.
    largest : bool, default=True
        Select the largest (True) or smallest (False) entries.

    Returns
    -------
    idx : ndarray of int
        Shape of ``x`` with ``x.shape[axis]`` replaced by ``k``.

    Raises
    ------
    ValueError
        If ``k`` is out of range or ``x`` contains NaN.

    Notes
    -----
    Complexity: O(n + k log k) per slice via ``np.argpartition``.
    """
    x = check_array(x, name="x", allow_nan=False)
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension")
    n = x.shape[axis]
    if not isinstance(k, (int, np.integer)) or isinstance(k, bool):
        raise TypeError(f"k must be an int, got {type(k).__name__}")
    if not 0 <= k <= n:
        raise ValueError(f"k must be in [0, {n}], got {k}")
    key = -x.astype(np.float64) if largest else x
    if k == 0:
        shape = list(x.shape)
        shape[axis] = 0
        return np.empty(shape, dtype=np.intp)
    part = np.argpartition(key, k - 1, axis=axis) if k < n else np.argsort(key, axis)
    part = np.take(part, np.arange(k), axis=axis)
    vals = np.take_along_axis(key, part, axis=axis)
    order = np.argsort(vals, axis=axis, kind="stable")
    return np.take_along_axis(part, order, axis=axis)


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------
def batch_iterator(
    *arrays,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    seed: int | None = None,
) -> Iterator[tuple[np.ndarray, ...] | np.ndarray]:
    """Yield aligned mini-batches from one or more arrays.

    Parameters
    ----------
    *arrays : array_like
        Arrays sharing the same first dimension ``n``.
    batch_size : int
        Rows per batch (the last batch may be smaller).
    shuffle : bool, default=False
        Shuffle rows once before batching.
    drop_last : bool, default=False
        Skip a final batch smaller than ``batch_size``.
    seed : int, optional
        Seed for the shuffle, for reproducibility.

    Yields
    ------
    ndarray or tuple of ndarray
        One array per input (a bare array when a single input is given).

    Raises
    ------
    ValueError
        If no arrays are given, lengths differ or ``batch_size < 1``.

    Notes
    -----
    Complexity: O(n) total time; each batch is a view (no shuffle) or a copy
    of ``batch_size`` rows (shuffle).
    """
    if not arrays:
        raise ValueError("batch_iterator needs at least one array")
    if not isinstance(batch_size, (int, np.integer)) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive int, got {batch_size!r}")
    arrs = [np.asarray(a) for a in arrays]
    n = check_consistent_length(*arrs)
    order = np.random.default_rng(seed).permutation(n) if shuffle else None
    stop = n - n % batch_size if drop_last else n
    for start in range(0, stop, batch_size):
        sl = slice(start, min(start + batch_size, n))
        batch = tuple(a[order[sl]] if order is not None else a[sl] for a in arrs)
        yield batch[0] if len(batch) == 1 else batch