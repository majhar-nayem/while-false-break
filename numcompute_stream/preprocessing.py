"""Preprocessing transformers with a fit / partial_fit / transform API.

All transformers take X of shape (n_samples, n_features). The scalers ignore
NaN when fitting and keep NaN in the output; SimpleImputer fills it.

Streaming: partial_fit(X_chunk) updates what was learned so far with one more
chunk, keeping only small running summaries (counts, sums, min/max, category
lists), never the data. fit(X) is the same as reset + partial_fit(X), so
fitting chunk by chunk gives the same result as fitting on all the data at once
(except the approximate streaming median, see SimpleImputer). An empty chunk
is ignored.
"""

import numpy as np

from numcompute_stream.stats import RunningStats, StreamingHistogram


def _check_2d(X, n_features=None, fitting=False):
    """Convert X to a 2-D array and check its shape."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(
            f"X must be 2-D, got shape {X.shape}; use X.reshape(-1, 1) for one feature"
        )
    if fitting and X.shape[0] == 0:
        raise ValueError("cannot fit on X with 0 samples")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features, but was fitted with {n_features}")
    return X


def _replace_zero_or_nan(values):
    """Replace 0 and NaN with 1, so the values are safe to divide by."""
    is_bad = (values == 0) | np.isnan(values)
    return np.where(is_bad, 1.0, values)


class Transformer:
    """Base class: subclasses implement fit and transform (and partial_fit to stream)."""

    def fit_transform(self, X, y=None):
        """Fit to X, then transform X."""
        return self.fit(X, y).transform(X)

    def _check_fitted(self):
        """Raise RuntimeError if fit has not been called yet."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(
                f"{type(self).__name__} is not fitted yet; call fit or partial_fit first"
            )

    def _check_chunk(self, X):
        """Check a partial_fit chunk: 2-D, and the same width as earlier chunks."""
        return _check_2d(X, getattr(self, "n_features_in_", None))

    def _forget(self):
        """Drop everything learned so far, so the next partial_fit starts fresh."""
        for name in [name for name in vars(self) if name.endswith("_")]:
            delattr(self, name)


def _min_max_ignoring_nan(X):
    """Column min and max of a float array, NaN for a column with no values (no warning)."""
    valid = ~np.isnan(X)
    low = np.where(valid, X, np.inf).min(axis=0)
    high = np.where(valid, X, -np.inf).max(axis=0)
    has_value = valid.any(axis=0)
    return np.where(has_value, low, np.nan), np.where(has_value, high, np.nan)


class StandardScaler(Transformer):
    """
    Standardise each column: z = (x - mean) / std.

    Attributes:
        mean_ : ndarray, shape (n_features,)
        scale_ : ndarray, shape (n_features,), the std (1 for constant columns)
        n_samples_seen_ : ndarray of int, shape (n_features,), non-NaN values per column

    Raises:
        ValueError if X is not 2-D, is empty when fitting, or has a different
            number of features from the data it was fitted on.

    Notes:
        partial_fit merges each chunk's own mean and sum of squared deviations
        into the running ones (Chan et al.'s parallel form of Welford's
        update, in stats.RunningStats). Unlike sum(x^2) - n * mean^2 this
        does not lose precision when the mean is large. A column that is
        constant so far (std 0) is scaled by 1, so it never divides by zero.

    Complexity:
        O(n * d) time per chunk, O(d) memory.
    """

    def fit(self, X, y=None):
        """Learn the mean and std of each column. Returns self."""
        _check_2d(X, fitting=True)
        self._forget()
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Update the running mean and std with one more chunk. Returns self."""
        X = self._check_chunk(X).astype(float)
        if X.shape[0] == 0:
            return self
        if not hasattr(self, "running_"):
            self.running_ = RunningStats()
        self.running_.update(X)
        self.mean_ = self.running_.mean
        self.scale_ = _replace_zero_or_nan(self.running_.std())
        self.n_samples_seen_ = self.running_.count.copy()
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return (X - mean_) / scale_, shape (m, d)."""
        self._check_fitted()
        X = _check_2d(X, self.n_features_in_).astype(float)
        return (X - self.mean_) / self.scale_


class MinMaxScaler(Transformer):
    """
    Scale each column linearly to feature_range = (low, high).

    x' = low + (x - min) * (high - low) / (max - min). Constant columns map to low.

    Parameters:
        feature_range : tuple (low, high) with low < high

    Raises:
        ValueError if low >= high, plus the shape errors of StandardScaler.

    Notes:
        partial_fit keeps the running min and max of each column (np.fmin /
        np.fmax ignore NaN). A later chunk with a wider spread changes the
        scaling, so values transformed earlier are not comparable with later
        ones; transformed values can also fall outside feature_range when a
        new chunk goes beyond the min/max seen so far.

    Complexity:
        O(n * d) time per chunk, O(d) memory.
    """

    def __init__(self, feature_range=(0, 1)):
        if feature_range[0] >= feature_range[1]:
            raise ValueError(f"feature_range min must be < max, got {feature_range}")
        self.feature_range = feature_range

    def fit(self, X, y=None):
        """Learn the min and max of each column. Returns self."""
        _check_2d(X, fitting=True)
        self._forget()
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Update the running min and max with one more chunk. Returns self."""
        X = self._check_chunk(X).astype(float)
        if X.shape[0] == 0:
            return self
        chunk_min, chunk_max = _min_max_ignoring_nan(X)
        if hasattr(self, "data_min_"):
            chunk_min = np.fmin(self.data_min_, chunk_min)
            chunk_max = np.fmax(self.data_max_, chunk_max)
        self.data_min_ = chunk_min
        self.data_max_ = chunk_max

        low, high = self.feature_range
        data_range = _replace_zero_or_nan(self.data_max_ - self.data_min_)
        self.scale_ = (high - low) / data_range
        self.min_ = low - self.data_min_ * self.scale_
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return X * scale_ + min_, shape (m, d)."""
        self._check_fitted()
        X = _check_2d(X, self.n_features_in_).astype(float)
        return X * self.scale_ + self.min_


class SimpleImputer(Transformer):
    """
    Replace missing values in each column.

    Parameters:
        strategy : "mean", "median", "most_frequent" or "constant";
            mean/median need numeric data, most_frequent/constant also work on text
        fill_value : value for strategy="constant" (default 0, or "missing" for text)
        missing_values : marker of a missing cell (np.nan, or e.g. "" for text)

        median_bins : int, bins of the histogram behind the streaming median

    Attributes:
        statistics_ : ndarray, shape (n_features,), the value used for each column

    Raises:
        ValueError if strategy is unknown, or mean/median is used on text.

    Notes:
        Ties in most_frequent go to the smallest value. A column that is
        entirely missing gets NaN for mean/median.

        partial_fit updates the fill values with each chunk:
        - mean: running sum and count of the present values per column (exact);
        - most_frequent: running count of every value seen in each column
          (exact), merged with np.unique;
        - median: an exact median needs every value, so a stats.StreamingHistogram
          per column (its range widens as later chunks need it) gives an
          approximate one, within one bin width of the middle data value (see
          StreamingHistogram.quantile). fit on a single array is still exact.

    Complexity:
        O(n * d) per chunk for mean/constant, O(n * d * log n) for
        median/most_frequent; memory O(d), O(d * median_bins) for median, and
        O(number of distinct values) for most_frequent.
    """

    STRATEGIES = ("mean", "median", "most_frequent", "constant")

    def __init__(self, strategy="mean", fill_value=None, missing_values=np.nan, median_bins=256):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values
        self.median_bins = median_bins

    def _missing_mask(self, X):
        """Return a boolean array that is True where X holds a missing value."""
        marker = self.missing_values
        marker_is_nan = isinstance(marker, float) and np.isnan(marker)
        if not marker_is_nan:
            return X == marker
        if X.dtype.kind == "f":
            return np.isnan(X)
        return np.zeros(X.shape, dtype=bool)  # text columns cannot hold NaN

    def _prepare(self, X, n_features=None, fitting=False):
        """Check X, turn integer columns into floats and find the missing cells."""
        X = _check_2d(X, n_features, fitting)
        if X.dtype.kind in "biu":
            X = X.astype(float)
        return X, self._missing_mask(X)

    def _update_value_counts(self, X, mask):
        """Add this chunk's value counts to value_counts_ (one (values, counts) pair per column)."""
        if not hasattr(self, "value_counts_"):
            self.value_counts_ = [(X[:0, j], np.zeros(0, dtype=int)) for j in range(X.shape[1])]
        for j, (old_values, old_counts) in enumerate(self.value_counts_):  # loop over columns
            present = X[~mask[:, j], j]
            values, inverse = np.unique(np.concatenate([old_values, present]), return_inverse=True)
            weights = np.concatenate([old_counts, np.ones(present.size, dtype=int)])
            counts = np.bincount(inverse, weights=weights, minlength=values.size).astype(int)
            self.value_counts_[j] = (values, counts)

    def _most_frequent(self):
        """Most common value of each column so far (fill_value if there is none)."""
        results = []
        for values, counts in self.value_counts_:
            if values.size > 0:
                results.append(values[np.argmax(counts)])  # values are sorted: ties -> smallest
            else:
                results.append(self.fill_value)
        return np.array(results)

    def fit(self, X, y=None):
        """Work out the fill value of each column. Returns self."""
        X, mask = self._prepare(X, fitting=True)
        self._forget()
        self.partial_fit(X)
        if self.strategy == "median":  # one array: use the exact median
            self.statistics_ = np.nanmedian(np.where(mask, np.nan, X), axis=0)
        return self

    def partial_fit(self, X, y=None):
        """Update the fill value of each column with one more chunk. Returns self."""
        X, mask = self._prepare(X, getattr(self, "n_features_in_", None))
        if X.shape[0] == 0:
            return self
        is_numeric = X.dtype.kind == "f"
        n_features = X.shape[1]

        if self.strategy in ("mean", "median") and not is_numeric:
            raise ValueError(f"strategy={self.strategy!r} needs numeric data")
        if self.strategy == "mean":
            if not hasattr(self, "sum_"):
                self.sum_ = np.zeros(n_features)
                self.count_ = np.zeros(n_features, dtype=int)
            self.sum_ += np.where(mask, 0.0, X).sum(axis=0)
            self.count_ += (~mask).sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                self.statistics_ = np.where(self.count_ > 0, self.sum_ / self.count_, np.nan)
        elif self.strategy == "median":
            if not hasattr(self, "histogram_"):
                self.histogram_ = StreamingHistogram(bins=self.median_bins)
            self.histogram_.update(np.where(mask, np.nan, X))
            self.statistics_ = self.histogram_.quantile(0.5)
        elif self.strategy == "constant":
            fill = self.fill_value
            if fill is None:
                fill = 0.0 if is_numeric else "missing"
            self.statistics_ = np.full(n_features, fill)
        else:
            self._update_value_counts(X, mask)
            self.statistics_ = self._most_frequent()

        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        """Return X with missing cells replaced by statistics_, shape (m, d)."""
        self._check_fitted()
        X, mask = self._prepare(X, self.n_features_in_)
        return np.where(mask, self.statistics_, X)


# The Assignment 2.2 specification calls this class Imputer; both names work.
Imputer = SimpleImputer


class OneHotEncoder(Transformer):
    """
    Encode categorical columns as 0/1 indicator columns.

    Parameters:
        handle_unknown : "error" or "ignore"; for a category not seen in fit,
            raise ValueError or output all zeros
        categories : None or list with one array of categories per column.
            If given, the categories are fixed: fitting only checks the
            number of columns, and the output width never changes.

    Attributes:
        categories_ : list with the sorted categories of each column
        n_output_features_ : int, number of output columns

    Returns (transform):
        ndarray, shape (m, total number of categories)

    Raises:
        ValueError if handle_unknown is invalid, X contains NaN, or an unknown
            category is found with handle_unknown="error".

    Notes:
        partial_fit adds categories first seen in a new chunk (np.union1d keeps
        each list sorted), so the output gets wider as the stream goes on. A
        model trained on the narrower output then no longer fits the new
        columns. For a pipeline that streams into a model, pass the categories
        up front (categories=...) or fit on a chunk that has them all, and use
        handle_unknown="ignore" so a rare new value becomes all zeros.

    Complexity:
        O(n * d * log k) time for k categories per column.
    """

    def __init__(self, handle_unknown="error", categories=None):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got {handle_unknown!r}")
        self.handle_unknown = handle_unknown
        self.categories = categories

    def _prepare(self, X, n_features=None, fitting=False):
        """Check X and reject NaN, which has no sensible category."""
        X = _check_2d(X, n_features, fitting)
        if X.dtype.kind == "f" and np.isnan(X).any():
            raise ValueError("X contains NaN; impute missing values before one-hot encoding")
        return X

    def fit(self, X, y=None):
        """Store the sorted categories of each column. Returns self."""
        self._prepare(X, fitting=True)
        self._forget()
        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Add the categories first seen in this chunk. Returns self."""
        X = self._prepare(X, getattr(self, "n_features_in_", None))
        n_features = X.shape[1]
        if self.categories is not None:
            if len(self.categories) != n_features:
                raise ValueError(
                    f"categories has {len(self.categories)} columns but X has {n_features}"
                )
            self.categories_ = [np.unique(np.asarray(c)) for c in self.categories]
        elif hasattr(self, "categories_"):
            self.categories_ = [
                np.union1d(categories, X[:, j]) for j, categories in enumerate(self.categories_)
            ]
        elif X.shape[0] > 0:
            self.categories_ = [np.unique(X[:, j]) for j in range(n_features)]
        else:
            return self
        self.n_features_in_ = n_features
        self.n_output_features_ = sum(categories.size for categories in self.categories_)
        return self

    def transform(self, X):
        """Return the one-hot matrix, shape (m, sum of len(categories_))."""
        self._check_fitted()
        X = self._prepare(X, self.n_features_in_)
        n_rows = X.shape[0]
        rows = np.arange(n_rows)
        total_categories = sum(categories.size for categories in self.categories_)
        out = np.zeros((n_rows, total_categories))

        offset = 0  # first output column of the current input column
        for j, categories in enumerate(self.categories_):  # loop over columns only
            column = X[:, j]
            position = np.searchsorted(categories, column)
            position = np.clip(position, 0, categories.size - 1)
            known = categories[position] == column
            if not known.all() and self.handle_unknown == "error":
                unknown = np.unique(column[~known]).tolist()
                raise ValueError(f"unknown categories {unknown} in column {j}")
            out[rows[known], offset + position[known]] = 1
            offset += categories.size
        return out
