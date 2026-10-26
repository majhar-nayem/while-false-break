"""Finite-difference gradients and Jacobians.

central: (f(x + h) - f(x - h)) / 2h, error O(h^2), 2n calls to f
forward: (f(x + h) - f(x)) / h,      error O(h),   n + 1 calls to f

Choosing h: a smaller h reduces the truncation error above, but f(x + h)
and f(x) then agree in more digits, so round-off error (about eps / h)
grows. The default h = 1e-5 is a good balance for the central method.
"""

import numpy as np

METHODS = ("central", "forward")


def _setup(x, h, method):
    """
    Check the arguments and build the step matrix.

    Returns (x as a float array, steps), where row i of steps is h in
    position i and 0 elsewhere, so x + steps[i] moves only x[i].
    """
    if method not in METHODS:
        raise ValueError(f"method must be 'central' or 'forward', got {method!r}")
    if not np.isscalar(h) or not h > 0:
        raise ValueError(f"h must be a positive number, got {h!r}")

    x = np.asarray(x)
    if x.dtype.kind not in "biuf":
        raise TypeError(f"x must be numeric, got dtype {x.dtype}")
    x = x.astype(float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"x must be a non-empty 1-D array, got shape {x.shape}")

    steps = np.eye(x.size) * h
    return x, steps


def _finite_differences(f, x, fx, steps, h, method):
    """Apply the central or forward formula; row i is the change caused by moving x[i]."""
    f_plus = np.array([f(x + step) for step in steps], dtype=float)
    if method == "central":
        f_minus = np.array([f(x - step) for step in steps], dtype=float)
        return (f_plus - f_minus) / (2 * h)
    return (f_plus - fx) / h


def grad(f, x, h=1e-5, method="central"):
    """
    Estimate the gradient of a scalar function f: R^n -> R.

    Parameters:
        f : callable, takes a 1-D array of length n and returns a number
        x : array_like, shape (n,), the point to evaluate at
        h : float > 0, step size
        method : "central" or "forward"

    Returns:
        ndarray, shape (n,), the estimated partial derivatives df/dx_i

    Raises:
        TypeError if x is not numeric.
        ValueError if x is not a non-empty 1-D array, h <= 0, method is
            unknown, or f does not return a single number.

    Notes:
        f is any Python function, so it has to be called once per step; the loop
        is over the n coordinates, and everything else is array arithmetic.

    Complexity:
        2n (central) or n + 1 (forward) calls to f, O(n^2) memory for the steps.
    """
    x, steps = _setup(x, h, method)
    fx = np.asarray(f(x), dtype=float)
    if fx.ndim != 0:
        raise ValueError(f"f must return a single number, got shape {fx.shape}; use jacobian()")
    return _finite_differences(f, x, fx, steps, h, method)


def jacobian(F, x, h=1e-5, method="central"):
    """
    Estimate the Jacobian matrix of a vector function F: R^n -> R^m.

    Parameters:
        F : callable, takes a 1-D array of length n and returns a 1-D array of length m
        x : array_like, shape (n,), the point to evaluate at
        h : float > 0, step size
        method : "central" or "forward"

    Returns:
        ndarray, shape (m, n); entry [i, j] is dF_i / dx_j

    Raises:
        TypeError if x is not numeric.
        ValueError if x is not a non-empty 1-D array, h <= 0, method is
            unknown, or F does not return a 1-D array.

    Notes:
        Row j of the finite differences is the change in all outputs when x[j]
        moves, i.e. column j of the Jacobian, so the result is transposed at the end.

    Complexity:
        2n (central) or n + 1 (forward) calls to F, O(n * m) memory.
    """
    x, steps = _setup(x, h, method)
    Fx = np.asarray(F(x), dtype=float)
    if Fx.ndim != 1:
        raise ValueError(f"F must return a 1-D array, got shape {Fx.shape}; use grad()")
    differences = _finite_differences(F, x, Fx, steps, h, method)  # shape (n, m)
    return differences.T
