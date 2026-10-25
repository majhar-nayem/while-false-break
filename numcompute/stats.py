
from __future__ import annotations

import numpy as np

from numcompute.utils import check_array, check_nan_policy

__all__ = [
    "mean",
    "var",
    "std",
    "median",
    "amin",
    "amax",
    "quantile",
    "histogram",
    "describe",
    "RunningStats",
]

QUANTILE_METHODS = ("linear", "lower", "higher", "midpoint", "nearest")


def _prepare(x, axis, nan_policy, name="x"):
    """Validate input, apply the NaN policy, and reject empty reductions."""
    x = check_array(x, name=name, to_float=True)
    check_nan_policy(nan_policy)
    if x.ndim == 0:
        x = x.reshape(1)
    if axis is not None:
        if not -x.ndim <= axis < x.ndim:
            raise ValueError(f"axis {axis} is out of bounds for {x.ndim}-D input")
        if x.shape[axis] == 0:
            raise ValueError(f"cannot reduce over empty axis {axis} (shape {x.shape})")
    elif x.size == 0:
        raise ValueError(f"{name} is empty")
    nan = np.isnan(x)
    if nan_policy == "raise" and nan.any():
        raise ValueError(f"{name} contains NaN (nan_policy='raise')")
    return x, nan


def _finish(out, keepdims, axis, x):
    """Squeeze the reduced axes unless ``keepdims`` and unwrap 0-D results."""
    if not keepdims:
        out = out.reshape(()) if axis is None else np.squeeze(out, axis=axis)
    elif axis is None:
        out = out.reshape((1,) * x.ndim)
    return out[()] if isinstance(out, np.ndarray) and out.ndim == 0 else out


def _count_sum(x, nan, axis, nan_policy):
    if nan_policy == "omit":
        count = np.sum(~nan, axis=axis, keepdims=True)
        total = np.sum(np.where(nan, 0.0, x), axis=axis, keepdims=True)
    else:
        total = np.sum(x, axis=axis, keepdims=True)
        count = np.full(total.shape, x.size if axis is None else x.shape[axis])
    return count, total


def _safe_div(num, den):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def mean(
    x, axis: int | None = None, keepdims: bool = False, nan_policy: str = "propagate"
):
   
    x, nan = _prepare(x, axis, nan_policy)
    count, total = _count_sum(x, nan, axis, nan_policy)
    return _finish(_safe_div(total, count), keepdims, axis, x)


def var(
    x,
    axis: int | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    nan_policy: str = "propagate",
):
    
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof}")
    x, nan = _prepare(x, axis, nan_policy)
    count, total = _count_sum(x, nan, axis, nan_policy)
    mu = _safe_div(total, count)
    dev = x - mu
    sq = dev * dev
    if nan_policy == "omit":
        sq = np.where(nan, 0.0, sq)
    ss = np.sum(sq, axis=axis, keepdims=True)
    return _finish(_safe_div(ss, count - ddof), keepdims, axis, x)


def std(
    x,
    axis: int | None = None,
    ddof: int = 0,
    keepdims: bool = False,
    nan_policy: str = "propagate",
):
    
    return np.sqrt(
        var(x, axis=axis, ddof=ddof, keepdims=keepdims, nan_policy=nan_policy)
    )


def _extreme(x, axis, keepdims, nan_policy, fn, fill):
    x, nan = _prepare(x, axis, nan_policy)
    if nan_policy == "omit":
        out = fn(np.where(nan, fill, x), axis=axis, keepdims=True)
        all_nan = np.all(nan, axis=axis, keepdims=True)
        out = np.where(all_nan, np.nan, out)
    else:
        out = fn(x, axis=axis, keepdims=True)  # NaN propagates in np.min/max
    return _finish(out, keepdims, axis, x)


def amin(
    x, axis: int | None = None, keepdims: bool = False, nan_policy: str = "propagate"
):
   
    return _extreme(x, axis, keepdims, nan_policy, np.min, np.inf)


def amax(
    x, axis: int | None = None, keepdims: bool = False, nan_policy: str = "propagate"
):
  
    return _extreme(x, axis, keepdims, nan_policy, np.max, -np.inf)


def quantile(
    x,
    q,
    axis: int | None = None,
    method: str = "linear",
    keepdims: bool = False,
    nan_policy: str = "propagate",
):
   
    if method not in QUANTILE_METHODS:
        raise ValueError(f"method must be one of {QUANTILE_METHODS}, got {method!r}")
    qa = check_array(q, name="q", to_float=True)
    if np.isnan(qa).any() or np.any((qa < 0) | (qa > 1)):
        raise ValueError("quantiles must be in the range [0, 1]")
    x, nan = _prepare(x, axis, nan_policy)
    in_ndim = x.ndim
    if axis is None:
        x, nan = x.reshape(-1), nan.reshape(-1)
        ax = 0
    else:
        ax = axis % x.ndim
    xs = np.moveaxis(np.sort(x, axis=ax), ax, -1)  # (..., n), NaN last
    n = xs.shape[-1]
    nan_count = np.sum(np.moveaxis(nan, ax, -1), axis=-1)  # (...)
    m = n - nan_count if nan_policy == "omit" else np.full_like(nan_count, n)

    qb = qa.reshape(qa.shape + (1,) * (xs.ndim - 1))  # broadcast vs slices
    pos = qb * np.maximum(m - 1, 0)
    lo_i = np.floor(pos).astype(np.intp)
    hi_i = np.ceil(pos).astype(np.intp)
    frac = pos - lo_i
    if method == "nearest":
        # round half to even, as numpy does
        lo_i = hi_i = np.around(pos).astype(np.intp)

    def gather(idx):
        idx = np.broadcast_to(idx, qa.shape + xs.shape[:-1])[..., None]
        src = np.broadcast_to(xs, qa.shape + xs.shape)
        return np.take_along_axis(src, idx, axis=-1)[..., 0]

    lo, hi = gather(lo_i), gather(hi_i)
    with np.errstate(invalid="ignore"):
        if method == "linear":
            out = np.where(lo == hi, lo, lo + (hi - lo) * frac)
        elif method in ("lower", "nearest"):
            out = lo
        elif method == "higher":
            out = hi
        else:
            out = np.where(lo == hi, lo, (lo + hi) / 2.0)
    out = np.asarray(out, dtype=np.float64)
    if nan_policy == "propagate":
        out = np.where(nan_count > 0, np.nan, out)
    else:
        out = np.where(m == 0, np.nan, out)
    if keepdims:
        if axis is None:
            out = out.reshape(qa.shape + (1,) * in_ndim)
        else:
            out = np.expand_dims(out, qa.ndim + ax)
    return out[()] if out.ndim == 0 else out


def median(
    x, axis: int | None = None, keepdims: bool = False, nan_policy: str = "propagate"
):
   
    return quantile(x, 0.5, axis=axis, keepdims=keepdims, nan_policy=nan_policy)


def histogram(
    x,
    bins=10,
    range: tuple[float, float] | None = None,  # noqa: A002 - mirrors numpy
    density: bool = False,
    nan_policy: str = "omit",
):
    
    x = check_array(x, name="x", to_float=True).ravel()
    check_nan_policy(nan_policy)
    nan = np.isnan(x)
    if nan.any():
        if nan_policy != "omit":
            raise ValueError("x contains NaN; use nan_policy='omit' to drop it")
        x = x[~nan]
    if np.isinf(x).any():
        raise ValueError("x contains infinite values, which cannot be binned")

    if np.ndim(bins) == 0:
        if not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError(f"bins must be a positive int, got {bins!r}")
        if range is not None:
            lo, hi = map(float, range)
            if not (np.isfinite(lo) and np.isfinite(hi)) or lo > hi:
                raise ValueError(f"range must be finite with min <= max, got {range}")
        elif x.size:
            lo, hi = float(x.min()), float(x.max())
        else:
            lo, hi = 0.0, 1.0
        if lo == hi:
            lo, hi = lo - 0.5, hi + 0.5
        edges = np.linspace(lo, hi, bins + 1)
        keep = (x >= lo) & (x <= hi)
        xk = x[keep]
        idx = ((xk - lo) * (bins / (hi - lo))).astype(np.intp)
        idx[idx == bins] -= 1
        # Round-off correction (identical to numpy's implementation).
        idx[xk < edges[idx]] -= 1
        inc = (xk >= edges[idx + 1]) & (idx != bins - 1)
        idx[inc] += 1
        counts = np.bincount(idx, minlength=bins)
    else:
        edges = check_array(bins, name="bins", ndim=1, to_float=True)
        if edges.size < 2 or np.any(np.diff(edges) <= 0):
            raise ValueError("bin edges must be strictly increasing with >= 2 edges")
        nb = edges.size - 1
        idx = np.searchsorted(edges, x, side="right") - 1
        idx[x == edges[-1]] = nb - 1  # last bin is closed on the right
        keep = (idx >= 0) & (idx < nb)
        counts = np.bincount(idx[keep], minlength=nb)

    if density:
        total = counts.sum()
        widths = np.diff(edges)
        with np.errstate(invalid="ignore", divide="ignore"):
            return counts / (total * widths) if total else counts * np.nan, edges
    return counts, edges


def describe(x, axis: int | None = 0, ddof: int = 0, nan_policy: str = "omit") -> dict:
    
    arr, nan = _prepare(x, axis, nan_policy)
    kw = {"axis": axis, "nan_policy": nan_policy}
    q25, q50, q75 = quantile(arr, [0.25, 0.5, 0.75], **kw)
    return {
        "count": np.sum(~nan, axis=axis),
        "nan_count": np.sum(nan, axis=axis),
        "mean": mean(arr, **kw),
        "std": std(arr, ddof=ddof, **kw),
        "min": amin(arr, **kw),
        "q25": q25,
        "median": q50,
        "q75": q75,
        "max": amax(arr, **kw),
    }


class RunningStats:
   

    def __init__(self, nan_policy: str = "omit"):
        self.nan_policy = check_nan_policy(nan_policy)
        self._shape: tuple[int, ...] | None = None

    def _init(self, shape):
        self._shape = shape
        self.count = np.zeros(shape, dtype=np.int64)
        self._mean = np.zeros(shape)
        self._m2 = np.zeros(shape)
        self._min = np.full(shape, np.inf)
        self._max = np.full(shape, -np.inf)

    def update(self, X) -> RunningStats:
        
        X = check_array(X, name="X", to_float=True)
        if X.ndim == 0:
            X = X.reshape(1)
        if self._shape is None:
            self._init(X.shape[1:])
        elif X.shape[1:] != self._shape:
            raise ValueError(
                f"batch feature shape {X.shape[1:]} does not match stream shape "
                f"{self._shape}"
            )
        if X.shape[0] == 0:
            return self
        nan = np.isnan(X)
        if self.nan_policy == "raise" and nan.any():
            raise ValueError("X contains NaN (nan_policy='raise')")
        if self.nan_policy == "omit":
            n_b = np.sum(~nan, axis=0)
            mean_b = _safe_div(np.sum(np.where(nan, 0.0, X), axis=0), n_b)
            dev = np.where(nan, 0.0, X - mean_b)
            mins = np.min(np.where(nan, np.inf, X), axis=0)
            maxs = np.max(np.where(nan, -np.inf, X), axis=0)
        else:
            n_b = np.full(self._shape, X.shape[0], dtype=np.int64)
            mean_b = np.mean(X, axis=0)
            dev = X - mean_b
            mins, maxs = np.min(X, axis=0), np.max(X, axis=0)
        m2_b = np.sum(dev * dev, axis=0)

        n_a = self.count
        n = n_a + n_b
        with np.errstate(invalid="ignore"):
            delta = np.where(n_b > 0, mean_b, 0.0) - self._mean
            w = _safe_div(n_b, n)
            w = np.where(n > 0, w, 0.0)
            self._mean = np.where(n_b > 0, self._mean + delta * w, self._mean)
            self._m2 = np.where(
                n_b > 0, self._m2 + m2_b + delta * delta * n_a * w, self._m2
            )
        self._min = np.minimum(self._min, mins)
        self._max = np.maximum(self._max, maxs)
        self.count = n
        return self

    def push(self, x) -> RunningStats:
       
        return self.update(np.asarray(x, dtype=np.float64)[None, ...])

    def merge(self, other: RunningStats) -> RunningStats:
        
        if other._shape is None:
            return self
        if self._shape is None:
            self._init(other._shape)
        if other._shape != self._shape:
            raise ValueError(f"cannot merge shapes {self._shape} and {other._shape}")
        n = self.count + other.count
        with np.errstate(invalid="ignore"):
            delta = other._mean - self._mean
            w = np.where(n > 0, _safe_div(other.count, n), 0.0)
            self._mean = self._mean + delta * w
            self._m2 = self._m2 + other._m2 + delta * delta * self.count * w
        self._min = np.minimum(self._min, other._min)
        self._max = np.maximum(self._max, other._max)
        self.count = n
        return self

    def _need(self):
        if self._shape is None:
            raise ValueError("no data has been added yet")

    @property
    def mean(self) -> np.ndarray:
        
        self._need()
        return np.where(self.count > 0, self._mean, np.nan)

    def var(self, ddof: int = 0) -> np.ndarray:
        
        self._need()
        return _safe_div(self._m2, self.count - ddof)

    def std(self, ddof: int = 0) -> np.ndarray:
        
        return np.sqrt(self.var(ddof))

    @property
    def min(self) -> np.ndarray:
        
        self._need()
        return np.where(self.count > 0, self._min, np.nan)

    @property
    def max(self) -> np.ndarray:
        
        self._need()
        return np.where(self.count > 0, self._max, np.nan)

    def __repr__(self) -> str:
        shape = "unset" if self._shape is None else self._shape
        return f"RunningStats(shape={shape}, nan_policy={self.nan_policy!r})"



