
from __future__ import annotations

from collections.abc import Callable

import numpy as np

from numcompute.utils import check_array

__all__ = ["grad", "jacobian", "check_grad", "backtracking_line_search"]

METHODS = ("central", "forward", "backward")


def _setup(x, h, method):
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    if not np.isscalar(h) or not h > 0:
        raise ValueError(f"h must be a positive scalar, got {h!r}")
    x = check_array(x, name="x", to_float=True, allow_nan=False)
    if x.size == 0:
        raise ValueError("x is empty")
    flat = x.ravel()
    steps = (flat + h) - flat  # exactly representable step per coordinate
    return x, flat, steps


def _perturbed(flat, steps, method):
    """Rows are the points at which f must be evaluated, shape (k*n, n)."""
    E = np.diag(steps)
    if method == "central":
        return np.vstack([flat + E, flat - E])
    if method == "forward":
        return np.vstack([flat[None, :], flat + E])
    return np.vstack([flat[None, :], flat - E])


def _evaluate(F, points, shape, vectorized):
    if vectorized:
        vals = np.asarray(F(points.reshape((-1,) + shape)), dtype=np.float64)
        return vals.reshape(points.shape[0], -1)
    out = [np.asarray(F(p.reshape(shape)), dtype=np.float64).ravel() for p in points]
    return np.stack(out)


def _differences(vals, steps, method, n):
    if method == "central":
        return (vals[:n] - vals[n:]) / (2.0 * steps[:, None])
    if method == "forward":
        return (vals[1:] - vals[0]) / steps[:, None]
    return (vals[0] - vals[1:]) / steps[:, None]


def grad(
    f: Callable,
    x,
    h: float = 1e-5,
    method: str = "central",
    vectorized: bool = False,
) -> np.ndarray:
    
    x, flat, steps = _setup(x, h, method)
    vals = _evaluate(f, _perturbed(flat, steps, method), x.shape, vectorized)
    if vals.shape[1] != 1:
        raise ValueError(
            f"f must return a scalar, got output with {vals.shape[1]} values; "
            "use jacobian() for vector-valued functions"
        )
    return _differences(vals, steps, method, flat.size)[:, 0].reshape(x.shape)


def jacobian(
    F: Callable,
    x,
    h: float = 1e-5,
    method: str = "central",
    vectorized: bool = False,
) -> np.ndarray:
    
    x, flat, steps = _setup(x, h, method)
    out_shape = np.shape(F(x)) if not vectorized else np.shape(F(x[None, ...]))[1:]
    vals = _evaluate(F, _perturbed(flat, steps, method), x.shape, vectorized)
    D = _differences(vals, steps, method, flat.size)  # (n_in, n_out)
    return D.T.reshape(tuple(out_shape) + x.shape)


def check_grad(f: Callable, grad_f: Callable, x, h: float = 1e-5) -> float:
    
    g_num = grad(f, x, h=h, method="central")
    g_ana = np.asarray(grad_f(np.asarray(x, dtype=np.float64)), dtype=np.float64)
    if g_ana.shape != g_num.shape:
        raise ValueError(f"grad_f returned shape {g_ana.shape}, expected {g_num.shape}")
    num = np.linalg.norm(g_ana - g_num)
    den = max(1.0, np.linalg.norm(g_ana) + np.linalg.norm(g_num))
    return float(num / den)


def backtracking_line_search(
    f: Callable,
    x,
    direction,
    grad_x=None,
    alpha0: float = 1.0,
    rho: float = 0.5,
    c: float = 1e-4,
    max_iter: int = 50,
) -> float:
    
    if not 0 < rho < 1 or not 0 < c < 1 or alpha0 <= 0:
        raise ValueError("need alpha0 > 0, 0 < rho < 1 and 0 < c < 1")
    x = check_array(x, name="x", to_float=True)
    d = check_array(direction, name="direction", to_float=True)
    if d.shape != x.shape:
        raise ValueError(f"direction has shape {d.shape}, expected {x.shape}")
    g = (
        grad(f, x)
        if grad_x is None
        else check_array(grad_x, name="grad_x", to_float=True)
    )
    slope = float(np.sum(g * d))
    if slope >= 0:
        raise ValueError(
            f"direction is not a descent direction (grad . d = {slope:.3g})"
        )
    fx = float(f(x))
    alpha = float(alpha0)
    for _ in range(max_iter):
        if float(f(x + alpha * d)) <= fx + c * alpha * slope:
            return alpha
        alpha *= rho
    return alpha
