"""Preprocessing transformers with a fit / transform / fit_transform API.

All transformers take X of shape (n_samples, n_features). The scalers ignore
NaN when fitting and keep NaN in the output; SimpleImputer fills it.
"""

import numpy as np


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
    """Base class: subclasses implement fit and transform."""

    def fit_transform(self, X, y=None):
        """Fit to X, then transform X."""
        return self.fit(X, y).transform(X)

    def _check_fitted(self):
        """Raise RuntimeError if fit has not been called yet."""
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet; call fit first")


class StandardScaler(Transformer):
    """
    Standardise each column: z = (x - mean) / std.

    Attributes:
        mean_ : ndarray, shape (n_features,)
        scale_ : ndarray, shape (n_features,), the std (1 for constant columns)

    Raises:
        ValueError if X is not 2-D, is empty when fitting, or has a different
            number of features from the data it was fitted on.

    Complexity:
        O(n * d) time, O(d) memory.
    """

    def fit(self, X, y=None):
        """Learn the mean and std of each column. Returns self."""
        X = _check_2d(X, fitting=True).astype(float)
        self.mean_ = np.nanmean(X, axis=0)
        self.scale_ = _replace_zero_or_nan(np.nanstd(X, axis=0))
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

    Complexity:
        O(n * d) time, O(d) memory.
    """

    def __init__(self, feature_range=(0, 1)):
        if feature_range[0] >= feature_range[1]:
            raise ValueError(f"feature_range min must be < max, got {feature_range}")
        self.feature_range = feature_range

    def fit(self, X, y=None):
        """Learn the min and max of each column. Returns self."""
        X = _check_2d(X, fitting=True).astype(float)
        low, high = self.feature_range
        self.data_min_ = np.nanmin(X, axis=0)
        data_range = np.nanmax(X, axis=0) - self.data_min_
        data_range = _replace_zero_or_nan(data_range)
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

    Attributes:
        statistics_ : ndarray, shape (n_features,), the value used for each column

    Raises:
        ValueError if strategy is unknown, or mean/median is used on text.

    Notes:
        Ties in most_frequent go to the smallest value. A column that is
        entirely missing gets NaN for mean/median.

    Complexity:
        O(n * d) for mean/constant, O(n * d * log n) for median/most_frequent.
    """

    STRATEGIES = ("mean", "median", "most_frequent", "constant")

    def __init__(self, strategy="mean", fill_value=None, missing_values=np.nan):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values

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

    def _most_frequent(self, X, mask):
        """Most common non-missing value of each column (fill_value if there is none)."""
        results = []
        for j in range(X.shape[1]):  # each column has its own set of values
            present = X[~mask[:, j], j]
            values, counts = np.unique(present, return_counts=True)
            if values.size > 0:
                results.append(values[np.argmax(counts)])
            else:
                results.append(self.fill_value)
        return np.array(results)

    def fit(self, X, y=None):
        """Work out the fill value of each column. Returns self."""
        X, mask = self._prepare(X, fitting=True)
        is_numeric = X.dtype.kind == "f"
        n_features = X.shape[1]

        if self.strategy in ("mean", "median"):
            if not is_numeric:
                raise ValueError(f"strategy={self.strategy!r} needs numeric data")
            X_with_nan = np.where(mask, np.nan, X)
            if self.strategy == "mean":
                self.statistics_ = np.nanmean(X_with_nan, axis=0)
            else:
                self.statistics_ = np.nanmedian(X_with_nan, axis=0)
        elif self.strategy == "constant":
            fill = self.fill_value
            if fill is None:
                fill = 0.0 if is_numeric else "missing"
            self.statistics_ = np.full(n_features, fill)
        else:
            self.statistics_ = self._most_frequent(X, mask)

        self.n_features_in_ = n_features
        return self

    def transform(self, X):
        """Return X with missing cells replaced by statistics_, shape (m, d)."""
        self._check_fitted()
        X, mask = self._prepare(X, self.n_features_in_)
        return np.where(mask, self.statistics_, X)


class OneHotEncoder(Transformer):
    """
    Encode categorical columns as 0/1 indicator columns.

    Parameters:
        handle_unknown : "error" or "ignore"; for a category not seen in fit,
            raise ValueError or output all zeros

    Attributes:
        categories_ : list with the sorted categories of each column

    Returns (transform):
        ndarray, shape (m, total number of categories)

    Raises:
        ValueError if handle_unknown is invalid, X contains NaN, or an unknown
            category is found with handle_unknown="error".

    Complexity:
        O(n * d * log k) time for k categories per column.
    """

    def __init__(self, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got {handle_unknown!r}")
        self.handle_unknown = handle_unknown

    def _prepare(self, X, n_features=None, fitting=False):
        """Check X and reject NaN, which has no sensible category."""
        X = _check_2d(X, n_features, fitting)
        if X.dtype.kind == "f" and np.isnan(X).any():
            raise ValueError("X contains NaN; impute missing values before one-hot encoding")
        return X

    def fit(self, X, y=None):
        """Store the sorted categories of each column. Returns self."""
        X = self._prepare(X, fitting=True)
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
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
