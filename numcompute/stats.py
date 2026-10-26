"""Descriptive statistics, histogram, quantiles and streaming (Welford) stats.

Axis rules follow NumPy: axis=None uses every value and returns a float,
axis=0 gives one value per column, axis=1 one value per row.
With ignore_nan=False (default) any NaN makes the result NaN; with
ignore_nan=True NaN values are left out.
"""

import numpy as np


def _prepare(x, axis):
    """Convert x to a float array and check that axis is valid and not empty."""
    x = np.asarray(x)
    if x.dtype.kind not in "biuf":
        raise TypeError(f"x must be numeric, got dtype {x.dtype}")
    x = x.astype(float)
    if axis is None:
        if x.size == 0:
            raise ValueError("x is empty")
    else:
        if not -x.ndim <= axis < x.ndim:
            raise ValueError(f"axis {axis} is out of range for a {x.ndim}-D array")
        if x.shape[axis] == 0:
            raise ValueError(f"cannot reduce over an empty axis (shape {x.shape})")
    return x


def _result(out):
    """Return a plain float for 0-D results."""
    return float(out) if np.ndim(out) == 0 else out


def _sum_and_count(x, axis, ignore_nan):
    """Return the sum along axis and how many values went into it."""
    if ignore_nan:
        valid = ~np.isnan(x)
        total = np.where(valid, x, 0).sum(axis=axis)
        count = valid.sum(axis=axis)
        return total, count
    if axis is None:
        count = x.size
    else:
        count = x.shape[axis]
    return x.sum(axis=axis), count


def mean(x, axis=None, ignore_nan=False):
    """
    Arithmetic mean.

    Parameters:
        x : array_like, numeric
        axis : int or None, axis to average over (default: all values)
        ignore_nan : bool, leave NaN values out

    Returns:
        float if axis is None, otherwise ndarray with that axis removed;
        NaN where there is nothing to average

    Raises:
        TypeError if x is not numeric.
        ValueError if x is empty or axis is out of range.

    Complexity:
        O(n) time, O(n) space.
    """
    x = _prepare(x, axis)
    total, count = _sum_and_count(x, axis, ignore_nan)
    with np.errstate(invalid="ignore", divide="ignore"):  # 0 / 0 -> NaN
        return _result(total / count)


def var(x, axis=None, ddof=0, ignore_nan=False):
    """
    Variance: mean squared distance from the mean.

    Parameters:
        x : array_like, numeric
        axis : int or None
        ddof : int >= 0, divide by (count - ddof); 0 = population, 1 = sample
        ignore_nan : bool, leave NaN values out

    Returns:
        float if axis is None, otherwise ndarray with that axis removed

    Raises:
        TypeError if x is not numeric.
        ValueError if x is empty, axis is out of range or ddof < 0.

    Notes:
        Two-pass formula (subtract the mean first, then square). This is more
        accurate than E[x^2] - E[x]^2, which can lose all precision when the mean
        is large compared with the spread.

    Complexity:
        O(n) time, O(n) space.
    """
    if ddof < 0:
        raise ValueError(f"ddof must be >= 0, got {ddof}")
    x = _prepare(x, axis)
    x_mean = mean(x, axis=axis, ignore_nan=ignore_nan)
    if axis is not None:
        x_mean = np.expand_dims(x_mean, axis)  # so it broadcasts against x

    squared_dev = (x - x_mean) ** 2
    if ignore_nan:
        squared_dev = np.where(np.isnan(x), 0, squared_dev)

    _, count = _sum_and_count(x, axis, ignore_nan)
    divisor = np.asarray(count - ddof)
    with np.errstate(invalid="ignore", divide="ignore"):
        variance = squared_dev.sum(axis=axis) / divisor
    # Not enough values for this ddof: the variance is undefined.
    return _result(np.where(divisor > 0, variance, np.nan))


def std(x, axis=None, ddof=0, ignore_nan=False):
    """
    Standard deviation, the square root of var.

    Parameters, Returns and Raises are the same as var.

    Complexity:
        O(n) time, O(n) space.
    """
    return _result(np.sqrt(var(x, axis=axis, ddof=ddof, ignore_nan=ignore_nan)))


def _min_or_max(x, axis, ignore_nan, reduce_func, nan_replacement):
    """
    Shared code for amin and amax.

    reduce_func is np.min or np.max. With ignore_nan, NaN is replaced by a
    value that can never win (inf for min, -inf for max).
    """
    x = _prepare(x, axis)
    if not ignore_nan:
        return _result(reduce_func(x, axis=axis))  # NaN propagates through np.min / np.max

    is_nan = np.isnan(x)
    result = reduce_func(np.where(is_nan, nan_replacement, x), axis=axis)
    all_nan = is_nan.all(axis=axis)
    return _result(np.where(all_nan, np.nan, result))


def amin(x, axis=None, ignore_nan=False):
    """
    Minimum value.

    Parameters:
        x : array_like, numeric
        axis : int or None
        ignore_nan : bool, leave NaN values out

    Returns:
        float if axis is None, otherwise ndarray with that axis removed

    Raises:
        TypeError if x is not numeric.
        ValueError if x is empty or axis is out of range.

    Complexity:
        O(n) time, O(n) space.
    """
    return _min_or_max(x, axis, ignore_nan, np.min, np.inf)


def amax(x, axis=None, ignore_nan=False):
    """
    Maximum value. Same parameters, returns and errors as amin.

    Complexity:
        O(n) time, O(n) space.
    """
    return _min_or_max(x, axis, ignore_nan, np.max, -np.inf)


def quantile(x, q, axis=None, ignore_nan=False):
    """
    Quantiles with linear interpolation (same as np.quantile).

    Parameters:
        x : array_like, numeric
        q : float or array_like of floats in [0, 1]
        axis : int or None
        ignore_nan : bool, leave NaN values out

    Returns:
        float if q is a scalar and axis is None, otherwise ndarray of shape
        q.shape + (shape of x without axis)

    Raises:
        TypeError if x is not numeric.
        ValueError if q is outside [0, 1], x is empty or axis is out of range.

    Notes:
        Each slice is sorted (NaN goes to the end). With m valid values the
        quantile q sits at position q * (m - 1); if that falls between two
        values we interpolate linearly. All slices and all q are done at once.

    Complexity:
        O(n log n) time for the sort, O(n * len(q)) space.
    """
    x = _prepare(x, axis)
    q = np.asarray(q, dtype=float)
    if np.isnan(q).any() or np.any((q < 0) | (q > 1)):
        raise ValueError("q must be in the range [0, 1]")
    if axis is None:
        x = x.ravel()
        axis = 0

    # Sort each slice and move the sorted axis to the end: shape (..., n), NaN last.
    sorted_x = np.moveaxis(np.sort(x, axis=axis), axis, -1)
    n_total = sorted_x.shape[-1]
    n_nan = np.isnan(sorted_x).sum(axis=-1)
    if ignore_nan:
        n_valid = n_total - n_nan
    else:
        n_valid = np.full(n_nan.shape, n_total)

    # Position of each quantile in every slice, shape q.shape + (...).
    position = np.multiply.outer(q, np.maximum(n_valid - 1, 0))
    lower_index = np.floor(position).astype(int)
    upper_index = np.ceil(position).astype(int)
    fraction = position - lower_index

    source = np.broadcast_to(sorted_x, q.shape + sorted_x.shape)
    lower_value = np.take_along_axis(source, lower_index[..., None], axis=-1)[..., 0]
    upper_value = np.take_along_axis(source, upper_index[..., None], axis=-1)[..., 0]
    result = lower_value + (upper_value - lower_value) * fraction

    if ignore_nan:
        no_answer = n_valid == 0
    else:
        no_answer = n_nan > 0
    return _result(np.where(no_answer, np.nan, result))


def median(x, axis=None, ignore_nan=False):
    """
    Median, the 0.5 quantile. Same parameters and errors as mean.

    Complexity:
        O(n log n) time, O(n) space.
    """
    return quantile(x, 0.5, axis=axis, ignore_nan=ignore_nan)


def histogram(x, bins=10, range=None):
    """
    Count values in equal-width bins (same result as np.histogram).

    Parameters:
        x : array_like, numeric, any shape (all values are used); NaN is ignored
        bins : int >= 1, number of bins
        range : (low, high), optional; default is (min(x), max(x)).
            Values outside the range are not counted.

    Returns:
        counts : ndarray of int, shape (bins,)
        edges : ndarray of float, shape (bins + 1,)
        Every bin is [left, right) except the last one, which includes the right edge.

    Raises:
        TypeError if x is not numeric.
        ValueError if bins < 1, x has infinite values, or range has low > high.

    Notes:
        Each value's bin is found with one vectorised binary search
        (np.searchsorted) and np.bincount counts them, so there is no Python loop.

    Complexity:
        O(n log bins) time, O(n + bins) space.
    """
    x = np.asarray(x)
    if x.dtype.kind not in "biuf":
        raise TypeError(f"x must be numeric, got dtype {x.dtype}")
    x = x.astype(float).ravel()
    x = x[~np.isnan(x)]
    if np.isinf(x).any():
        raise ValueError("x contains infinite values")
    if not isinstance(bins, (int, np.integer)) or bins < 1:
        raise ValueError(f"bins must be a positive int, got {bins!r}")

    if range is not None:
        low = float(range[0])
        high = float(range[1])
        if low > high:
            raise ValueError(f"range must have low <= high, got {range}")
    elif x.size > 0:
        low = float(x.min())
        high = float(x.max())
    else:
        low, high = 0.0, 1.0
    if low == high:  # all values equal: make a range of width 1 around them
        low -= 0.5
        high += 0.5

    edges = np.linspace(low, high, bins + 1)
    x = x[(x >= low) & (x <= high)]
    bin_index = np.searchsorted(edges, x, side="right") - 1
    bin_index[bin_index == bins] = bins - 1  # x == high goes in the last bin
    counts = np.bincount(bin_index, minlength=bins)
    return counts, edges


class RunningStats:
    """
    Streaming mean, variance and std, updated one batch at a time (Welford).

    Useful for files read with io.iter_csv_chunks: only the running count,
    mean and sum of squared deviations (M2) are kept, never the data.
    NaN values are ignored column by column.

    Example:
        rs = RunningStats()
        for chunk in io.iter_csv_chunks(path, chunksize=100):
            rs.update(chunk)
        rs.mean, rs.std()

    Complexity:
        O(batch size * d) time per update, O(d) memory for d columns.
    """

    def __init__(self):
        self.count = None
        self._mean = None
        self._m2 = None

    def update(self, X):
        """
        Add a batch of rows.

        Parameters:
            X : array_like, shape (n,) for one column or (n, d) for d columns

        Returns:
            self

        Raises:
            TypeError if X is not numeric.
            ValueError if X has a different number of columns than earlier batches.

        The batch is summarised on its own (count, mean, M2) and then merged
        with the running values using Chan et al.'s formula, the batch version
        of Welford's update:
            delta = mean_b - mean_a
            mean  = mean_a + delta * n_b / n
            M2    = M2_a + M2_b + delta^2 * n_a * n_b / n
        """
        X = np.asarray(X)
        if X.dtype.kind not in "biuf":
            raise TypeError(f"X must be numeric, got dtype {X.dtype}")
        X = X.astype(float)
        if X.ndim == 1:
            X = X[:, None]
        if X.ndim != 2:
            raise ValueError(f"X must be 1-D or 2-D, got shape {X.shape}")
        n_columns = X.shape[1]
        if self.count is None:
            self.count = np.zeros(n_columns, int)
            self._mean = np.zeros(n_columns)
            self._m2 = np.zeros(n_columns)
        elif n_columns != self.count.size:
            raise ValueError(f"expected {self.count.size} columns, got {n_columns}")

        # Summary of this batch on its own (NaN values left out).
        valid = ~np.isnan(X)
        n_b = valid.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_b = np.where(valid, X, 0).sum(axis=0) / n_b
            m2_b = (np.where(valid, X - mean_b, 0) ** 2).sum(axis=0)

            # Merge it with the running summary (Chan et al.).
            n = self.count + n_b
            delta = mean_b - self._mean
            new_mean = self._mean + delta * n_b / n
            new_m2 = self._m2 + m2_b + delta**2 * self.count * n_b / n

        has_new = n_b > 0  # columns where this batch had no values stay the same
        self._mean = np.where(has_new, new_mean, self._mean)
        self._m2 = np.where(has_new, new_m2, self._m2)
        self.count = n
        return self

    @property
    def mean(self):
        """Running mean per column, shape (d,); NaN for columns with no values."""
        if self.count is None:
            raise ValueError("no data has been added yet")
        return np.where(self.count > 0, self._mean, np.nan)

    def var(self, ddof=0):
        """Running variance per column, shape (d,); NaN where count <= ddof."""
        if self.count is None:
            raise ValueError("no data has been added yet")
        divisor = self.count - ddof
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(divisor > 0, self._m2 / divisor, np.nan)

    def std(self, ddof=0):
        """Running standard deviation per column, shape (d,)."""
        return np.sqrt(self.var(ddof))
