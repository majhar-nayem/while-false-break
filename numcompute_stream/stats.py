"""Descriptive statistics, histogram, quantiles and their streaming versions.

Batch functions (mean, var, quantile, histogram, ...) work on one array.
Axis rules follow NumPy: axis=None uses every value and returns a float,
axis=0 gives one value per column, axis=1 one value per row.
With ignore_nan=False (default) any NaN makes the result NaN; with
ignore_nan=True NaN values are left out.

Streaming classes are updated one chunk at a time and never keep the data:
    RunningStats        mean / variance / std (Welford, Chan et al. merge)
    StreamingHistogram  fixed-bin histogram with an optional sliding window,
                        and approximate quantiles read from it
    StreamingStats      all of the above behind one update_stats(X_chunk) call
They always work per column: a chunk has shape (n, d), or (n,) for one column,
and results have one entry per column.
"""

from collections import deque

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


def _as_columns(X, name="X"):
    """Return X as a float array of shape (n, d); a 1-D X becomes one column."""
    X = np.asarray(X)
    if X.dtype.kind not in "biuf":
        raise TypeError(f"{name} must be numeric, got dtype {X.dtype}")
    X = X.astype(float)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2:
        raise ValueError(f"{name} must be 1-D or 2-D, got shape {X.shape}")
    return X


class StreamingHistogram:
    """
    Equal-width histogram of each column, updated chunk by chunk.

    Parameters:
        bins : int >= 1, number of bins per column (even when range is None).
            The default is generous because, after the range doubles, up to
            half of the bins can lie outside the data.
        range : (low, high) or None.
            None (default): each column's range starts at the min and max of
            the first chunk and doubles whenever a later finite value falls
            outside it. Doubling merges neighbouring pairs of bins, so the bins
            stay equal width, every finite value is binned, and no value is
            ever lost. This needs an even number of bins.
            (low, high): a fixed range for every column; values outside it are
            not binned but counted in n_below / n_above.
            Infinite values are always counted in n_below / n_above.
        window : int >= 1 or None. If set, only the last `window` chunks are
            counted (a sliding window); older chunks are dropped.

    Attributes (after the first update):
        counts : ndarray of int, shape (d, bins)
        edges : ndarray, shape (d, bins + 1)
        n_below, n_above : ndarray of int, shape (d,), values outside the range
            (only infinite values when the range adapts)
        count : ndarray of int, shape (d,), non-NaN values counted

    Raises:
        ValueError for a bad bins, range or window, or a chunk whose number of
            columns differs from the first chunk.
        TypeError if a chunk is not numeric.
        RuntimeError if results are read before the first update.

    Notes:
        Because the bins are equal width, a value's bin is found with one
        division, floor((x - low) / width), for every value of every column at
        once; np.bincount on column * bins + bin counts them all in one call.
        NaN is ignored. The window keeps one small (d, bins) count array per
        chunk and subtracts the oldest one when it falls out. When the range
        doubles, the stored chunk counts are merged in the same way, so the
        window stays consistent.

    Complexity:
        O(n * d + d * bins) time per update, plus O(bins * window) for each
        doubling (at most log2(new spread / old spread) per column);
        O(window * d * bins) memory (O(d * bins) without a window).
    """

    def __init__(self, bins=256, range=None, window=None):
        if not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError(f"bins must be a positive int, got {bins!r}")
        if range is not None and not float(range[0]) < float(range[1]):
            raise ValueError(f"range must have low < high, got {range}")
        if range is None and bins % 2:
            raise ValueError(f"bins must be even when the range adapts (range=None), got {bins}")
        if window is not None and (not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError(f"window must be a positive int or None, got {window!r}")
        self.bins = int(bins)
        self.range = range
        self.window = window
        self.reset()

    def reset(self):
        """Forget everything, including the bin edges. Returns self."""
        self._low = None
        self._high = None
        self._chunks = deque()  # per-chunk (counts, below, above), only with a window
        self._counts = None
        self._below = None
        self._above = None
        return self

    def _set_range(self, X):
        """Fix the bin range of each column from the first chunk (or from self.range)."""
        d = X.shape[1]
        if self.range is not None:
            self._low = np.full(d, float(self.range[0]))
            self._high = np.full(d, float(self.range[1]))
        else:
            finite = np.isfinite(X)
            any_value = finite.any(axis=0)
            low = np.where(finite, X, np.inf).min(axis=0)
            high = np.where(finite, X, -np.inf).max(axis=0)
            low = np.where(any_value, low, 0.0)  # empty column: arbitrary range [0, 1]
            high = np.where(any_value, high, 1.0)
            constant = low == high  # all values equal: a range of width 1 around them
            self._low = np.where(constant, low - 0.5, low)
            # Start just above the max, so the max sits inside a [left, right) bin
            # and stays in the right bin when the range later doubles.
            self._high = np.nextafter(np.where(constant, high + 0.5, high), np.inf)
        self._counts = np.zeros((d, self.bins), dtype=int)
        self._below = np.zeros(d, dtype=int)
        self._above = np.zeros(d, dtype=int)

    def update(self, X_chunk):
        """
        Add a chunk of values.

        Parameters:
            X_chunk : array_like, shape (n,) or (n, d)

        Returns:
            self

        Notes:
            A chunk with no rows is ignored (it does not set the range or
            count towards the window).
        """
        X = _as_columns(X_chunk, "X_chunk")
        if X.shape[0] == 0:
            return self
        if self._low is None:
            self._set_range(X)
        d = self._low.size
        if X.shape[1] != d:
            raise ValueError(f"expected {d} columns, got {X.shape[1]}")
        if self.range is None:
            self._widen(X)

        valid = ~np.isnan(X)
        below = (X < self._low) & valid
        above = (X > self._high) & valid
        inside = valid & ~below & ~above

        width = (self._high - self._low) / self.bins
        with np.errstate(invalid="ignore"):
            bin_index = np.floor((X - self._low) / width)
        # x == high lands one past the end: clip it into the last bin.
        bin_index = np.clip(np.nan_to_num(bin_index), 0, self.bins - 1).astype(int)
        column = np.broadcast_to(np.arange(d), X.shape)
        flat = column[inside] * self.bins + bin_index[inside]
        counts = np.bincount(flat, minlength=d * self.bins).reshape(d, self.bins)
        below = below.sum(axis=0)
        above = above.sum(axis=0)

        self._counts += counts
        self._below += below
        self._above += above
        if self.window is not None:
            self._chunks.append((counts, below, above))
            if len(self._chunks) > self.window:
                old_counts, old_below, old_above = self._chunks.popleft()
                self._counts -= old_counts
                self._below -= old_below
                self._above -= old_above
        return self

    def _widen(self, X):
        """Double each column's range until it covers this chunk's finite values."""
        finite = np.isfinite(X)
        chunk_low = np.where(finite, X, np.inf).min(axis=0)
        chunk_high = np.where(finite, X, -np.inf).max(axis=0)
        outside = (chunk_low < self._low) | (chunk_high > self._high)
        for j in np.flatnonzero(outside):  # only the columns that need it, usually none
            while chunk_low[j] < self._low[j] or chunk_high[j] > self._high[j]:
                width = self._high[j] - self._low[j]
                grow_left = chunk_low[j] < self._low[j]
                if grow_left:
                    self._low[j] -= width
                else:
                    self._high[j] += width
                self._merge_bins(j, grow_left)

    def _merge_bins(self, j, grow_left):
        """
        Re-bin column j after its range doubled.

        Each new bin covers two old ones, so neighbouring pairs of old counts
        are summed into one half of the bins and the other half (the new
        part of the range) starts empty. Stored window chunks are re-binned too.
        """
        half = self.bins // 2

        def merged(row):
            pairs = row.reshape(half, 2).sum(axis=1)
            empty = np.zeros(half, dtype=row.dtype)
            return np.concatenate([empty, pairs] if grow_left else [pairs, empty])

        self._counts[j] = merged(self._counts[j])
        for counts, _, _ in self._chunks:
            counts[j] = merged(counts[j])

    def _check_updated(self):
        """Raise RuntimeError if no chunk has been added yet."""
        if self._low is None:
            raise RuntimeError("no data has been added yet; call update first")

    @property
    def counts(self):
        """Counts per bin, shape (d, bins)."""
        self._check_updated()
        return self._counts.copy()

    @property
    def edges(self):
        """Bin edges per column, shape (d, bins + 1)."""
        self._check_updated()
        steps = np.linspace(0.0, 1.0, self.bins + 1)
        return self._low[:, None] + np.multiply.outer(self._high - self._low, steps)

    @property
    def n_below(self):
        """Values below the range per column, shape (d,)."""
        self._check_updated()
        return self._below.copy()

    @property
    def n_above(self):
        """Values above the range per column, shape (d,)."""
        self._check_updated()
        return self._above.copy()

    @property
    def count(self):
        """Non-NaN values counted per column (inside and outside the range), shape (d,)."""
        self._check_updated()
        return self._counts.sum(axis=1) + self._below + self._above

    def quantile(self, q):
        """
        Approximate quantiles of each column, read from the histogram.

        Parameters:
            q : float or array_like of floats in [0, 1]

        Returns:
            ndarray of shape (d,) for a scalar q, or q.shape + (d,);
            NaN for a column with no values

        Raises:
            ValueError if q is outside [0, 1].

        Notes:
            The q * count-th value is found in the cumulative counts, and
            values inside its bin are assumed to be spread evenly. The
            estimate is always in the same bin as the ceil(q * count)-th
            smallest value, so it is within one bin width of that value.
            np.quantile interpolates between neighbouring values instead, so
            the two agree to about a bin width when the data is dense, but
            can differ by more when there are only a few values with gaps
            wider than a bin. q = 0 gives the left edge of the first bin that
            holds a value. A quantile that falls among the values below (or
            above) the range is reported as the range's low (or high) edge.

        Complexity:
            O(len(q) * d * bins) time and space.
        """
        self._check_updated()
        q = np.asarray(q, dtype=float)
        if np.isnan(q).any() or np.any((q < 0) | (q > 1)):
            raise ValueError("q must be in the range [0, 1]")

        total = self.count  # (d,)
        target = np.multiply.outer(q, total)  # (..., d): how many values lie below the answer
        cum_end = self._below[:, None] + np.cumsum(self._counts, axis=1)  # (d, bins)
        cum_start = cum_end - self._counts

        # Bin holding the target: the number of bins that end before it. Empty
        # bins before the first value are always skipped, so q = 0 (target 0)
        # lands in the first bin that holds a value, not at the range's edge.
        ends_before = (cum_end < target[..., None]) | (cum_end == 0)
        j = ends_before.sum(axis=-1)  # (..., d), bins means "above the range"
        j_safe = np.minimum(j, self.bins - 1)
        columns = np.arange(total.size)
        in_bin = self._counts[columns, j_safe]
        start = cum_start[columns, j_safe]
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = np.where(in_bin > 0, (target - start) / in_bin, 0.0)
        fraction = np.clip(fraction, 0.0, 1.0)
        width = (self._high - self._low) / self.bins
        value = self._low + (j_safe + fraction) * width

        among_below = (self._below > 0) & (target <= self._below)
        value = np.where(among_below, self._low, value)
        value = np.where(j >= self.bins, self._high, value)
        return np.where(total > 0, value, np.nan)


class StreamingStats:
    """
    Streaming summary of each column: count, mean, variance, min, max,
    histogram and quantiles, all updated with one call per chunk.

    Example:
        stats = StreamingStats()
        for chunk in io.iter_csv_chunks(path, chunksize=200, usecols=[5, 9]):
            stats.update_stats(chunk)
        stats.mean, stats.std(), stats.quantile([0.25, 0.5, 0.75])

    Parameters:
        bins, range, window : passed to StreamingHistogram. The window only
            applies to the histogram and quantiles; count, mean, variance,
            min and max always cover the whole stream.
        axis : 0 (default) or None, with the batch functions' meaning. Rows
            always arrive in chunks, so axis=0 summarises each column and
            returns arrays of shape (d,) (quantiles: q.shape + (d,)); axis=None
            pools every value of every column and returns plain floats
            (quantiles: q.shape).

    Raises:
        TypeError if a chunk is not numeric.
        ValueError if axis is not 0 or None, or a chunk has a different
            number of columns.
        RuntimeError if results are read before the first update.

    Notes:
        Mean and variance are exact (RunningStats merges each chunk's own
        mean and M2 with Chan et al.'s formula, which stays accurate where
        sum(x^2) - n * mean^2 would cancel). Quantiles are approximate, because
        exact ones would need every value: each is within one bin width of
        the data value at that rank (see StreamingHistogram.quantile).
        NaN is ignored column by column; a column with no values gives NaN.

    Complexity:
        O(n * d + d * bins) time per chunk, O(d * bins) memory (times window).
    """

    def __init__(self, bins=256, range=None, window=None, axis=0):
        if axis not in (0, None):
            raise ValueError(f"axis must be 0 (per column) or None (all values), got {axis!r}")
        self.axis = axis
        self._histogram = StreamingHistogram(bins=bins, range=range, window=window)
        self.reset()

    def _out(self, values):
        """Per-column results as they are for axis=0; the single pooled value for axis=None."""
        if self.axis is None:
            return values[0].item() if np.ndim(values) == 1 else values[..., 0]
        return values

    def reset(self):
        """Forget all data. Returns self."""
        self._moments = RunningStats()
        self._histogram.reset()
        self._min = None
        self._max = None
        self.n_chunks = 0
        return self

    def update_stats(self, X_chunk):
        """
        Add one chunk of rows.

        Parameters:
            X_chunk : array_like, shape (n, d), or (n,) for one column

        Returns:
            self

        Notes:
            A chunk with no rows is ignored.
        """
        X = _as_columns(X_chunk, "X_chunk")
        if X.shape[0] == 0:
            return self
        if self.axis is None:
            X = X.reshape(-1, 1)  # pool every value into one column
        if self._min is not None and X.shape[1] != self._min.size:
            raise ValueError(f"expected {self._min.size} columns, got {X.shape[1]}")
        self._moments.update(X)
        self._histogram.update(X)
        valid = ~np.isnan(X)
        chunk_min = np.where(valid, X, np.inf).min(axis=0)
        chunk_max = np.where(valid, X, -np.inf).max(axis=0)
        if self._min is None:
            self._min, self._max = chunk_min, chunk_max
        else:
            self._min = np.minimum(self._min, chunk_min)
            self._max = np.maximum(self._max, chunk_max)
        self.n_chunks += 1
        return self

    def _check_updated(self):
        """Raise RuntimeError if no chunk has been added yet."""
        if self._min is None:
            raise RuntimeError("no data has been added yet; call update_stats first")

    @property
    def count(self):
        """Non-NaN values seen per column, shape (d,)."""
        self._check_updated()
        return self._out(self._moments.count.copy())

    @property
    def mean(self):
        """Mean per column, shape (d,)."""
        self._check_updated()
        return self._out(self._moments.mean)

    def var(self, ddof=0):
        """Variance per column, shape (d,); NaN where count <= ddof."""
        self._check_updated()
        return self._out(self._moments.var(ddof))

    def std(self, ddof=0):
        """Standard deviation per column, shape (d,) (a float for axis=None)."""
        value = np.sqrt(self.var(ddof))
        return float(value) if self.axis is None else value

    @property
    def min(self):
        """Minimum per column, shape (d,); NaN for a column with no values."""
        self._check_updated()
        return self._out(np.where(np.isinf(self._min) & (self._min > 0), np.nan, self._min))

    @property
    def max(self):
        """Maximum per column, shape (d,); NaN for a column with no values."""
        self._check_updated()
        return self._out(np.where(np.isinf(self._max) & (self._max < 0), np.nan, self._max))

    def quantile(self, q):
        """Approximate quantiles per column (see StreamingHistogram.quantile)."""
        self._check_updated()
        return self._out(self._histogram.quantile(q))

    def median(self):
        """Approximate median per column, shape (d,)."""
        return self.quantile(0.5)

    def histogram(self):
        """
        Return (counts, edges), shapes (d, bins) and (d, bins + 1).

        For axis=None the shapes are (bins,) and (bins + 1,).
        """
        self._check_updated()
        counts, edges = self._histogram.counts, self._histogram.edges
        return (counts[0], edges[0]) if self.axis is None else (counts, edges)
