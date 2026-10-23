"""Preprocessing transformers with a fit / transform / fit_transform API.

All transformers take X of shape (n_samples, n_features). The scalers ignore
NaN when fitting and keep NaN in the output; SimpleImputer fills it.
"""

import numpy as np


def _check_2d(X, n_features=None, fitting=False):
    """Convert X to a 2-D array and check its shape."""
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got shape {X.shape}; use X.reshape(-1, 1) for one feature")
    if fitting and X.shape[0] == 0:
        raise ValueError("cannot fit on X with 0 samples")
    if n_features is not None and X.shape[1] != n_features:
        raise ValueError(f"X has {X.shape[1]} features, but was fitted with {n_features}")
    return X


class Transformer:
    """Base class: subclasses implement fit and transform."""

    def fit_transform(self, X, y=None):
        """Fit to X, then transform X."""
        return self.fit(X, y).transform(X)

    def _check_fitted(self):
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError(f"{type(self).__name__} is not fitted yet; call fit first")


class StandardScaler(Transformer):
    """Standardise each column: z = (x - mean) / std.

    Attributes: mean_, scale_ (std; 1 for constant columns), shape (n_features,).
    Complexity: O(n * d) time, O(d) memory.
    """

    def fit(self, X, y=None):
        X = _check_2d(X, fitting=True).astype(float)
        self.mean_ = np.nanmean(X, axis=0)
        std = np.nanstd(X, axis=0)
        self.scale_ = np.where((std == 0) | np.isnan(std), 1.0, std)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return (X - mean_) / scale_, shape (m, d)."""
        self._check_fitted()
        X = _check_2d(X, self.n_features_in_).astype(float)
        return (X - self.mean_) / self.scale_


class MinMaxScaler(Transformer):
    """Scale each column linearly to feature_range = (lo, hi).

    x' = lo + (x - min) * (hi - lo) / (max - min). Constant columns map to lo.
    Complexity: O(n * d) time, O(d) memory.
    """

    def __init__(self, feature_range=(0, 1)):
        if feature_range[0] >= feature_range[1]:
            raise ValueError(f"feature_range min must be < max, got {feature_range}")
        self.feature_range = feature_range

    def fit(self, X, y=None):
        X = _check_2d(X, fitting=True).astype(float)
        lo, hi = self.feature_range
        self.data_min_ = np.nanmin(X, axis=0)
        data_range = np.nanmax(X, axis=0) - self.data_min_
        data_range = np.where((data_range == 0) | np.isnan(data_range), 1.0, data_range)
        self.scale_ = (hi - lo) / data_range
        self.min_ = lo - self.data_min_ * self.scale_
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return X * scale_ + min_, shape (m, d)."""
        self._check_fitted()
        X = _check_2d(X, self.n_features_in_).astype(float)
        return X * self.scale_ + self.min_


class SimpleImputer(Transformer):
    """Replace missing values in each column.

    Parameters
    ----------
    strategy : "mean", "median", "most_frequent" or "constant"
        mean/median need numeric data; most_frequent/constant also work on text.
    fill_value : value for strategy="constant" (default 0, or "missing" for text)
    missing_values : marker of a missing cell (np.nan, or e.g. "" for text)

    Ties in most_frequent go to the smallest value. A column that is entirely
    missing gets NaN for mean/median.
    Complexity: O(n * d) for mean/constant, O(n * d * log n) for median/most_frequent.
    """

    STRATEGIES = ("mean", "median", "most_frequent", "constant")

    def __init__(self, strategy="mean", fill_value=None, missing_values=np.nan):
        if strategy not in self.STRATEGIES:
            raise ValueError(f"strategy must be one of {self.STRATEGIES}, got {strategy!r}")
        self.strategy = strategy
        self.fill_value = fill_value
        self.missing_values = missing_values

    def _prepare(self, X, n_features=None, fitting=False):
        X = _check_2d(X, n_features, fitting)
        if X.dtype.kind in "biu":
            X = X.astype(float)
        mv = self.missing_values
        if isinstance(mv, float) and np.isnan(mv):
            mask = np.isnan(X) if X.dtype.kind == "f" else np.zeros(X.shape, dtype=bool)
        else:
            mask = X == mv
        return X, mask

    def fit(self, X, y=None):
        X, mask = self._prepare(X, fitting=True)
        numeric = X.dtype.kind == "f"
        d = X.shape[1]

        if self.strategy in ("mean", "median"):
            if not numeric:
                raise ValueError(f"strategy={self.strategy!r} needs numeric data")
            X_nan = np.where(mask, np.nan, X)
            stat = np.nanmean if self.strategy == "mean" else np.nanmedian
            self.statistics_ = stat(X_nan, axis=0)
        elif self.strategy == "constant":
            fill = self.fill_value
            if fill is None:
                fill = 0.0 if numeric else "missing"
            self.statistics_ = np.full(d, fill)
        else:
            stats = []
            for j in range(d):  # each column has its own set of values
                values, counts = np.unique(X[~mask[:, j], j], return_counts=True)
                stats.append(values[np.argmax(counts)] if values.size else self.fill_value)
            self.statistics_ = np.array(stats)

        self.n_features_in_ = d
        return self

    def transform(self, X):
        """Return X with missing cells replaced by statistics_, shape (m, d)."""
        self._check_fitted()
        X, mask = self._prepare(X, self.n_features_in_)
        return np.where(mask, self.statistics_, X)


class OneHotEncoder(Transformer):
    """Encode categorical columns as 0/1 indicator columns.

    Parameters
    ----------
    handle_unknown : "error" or "ignore"
        For a category not seen in fit: raise ValueError, or output all zeros.

    Attributes: categories_, a list with the sorted categories of each column.
    Output shape: (m, total number of categories). NaN input is rejected.
    Complexity: O(n * d * log k) time for k categories per column.
    """

    def __init__(self, handle_unknown="error"):
        if handle_unknown not in ("error", "ignore"):
            raise ValueError(f"handle_unknown must be 'error' or 'ignore', got {handle_unknown!r}")
        self.handle_unknown = handle_unknown

    def _prepare(self, X, n_features=None, fitting=False):
        X = _check_2d(X, n_features, fitting)
        if X.dtype.kind == "f" and np.isnan(X).any():
            raise ValueError("X contains NaN; impute missing values before one-hot encoding")
        return X

    def fit(self, X, y=None):
        X = self._prepare(X, fitting=True)
        self.categories_ = [np.unique(X[:, j]) for j in range(X.shape[1])]
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return the one-hot matrix, shape (m, sum of len(categories_))."""
        self._check_fitted()
        X = self._prepare(X, self.n_features_in_)
        rows = np.arange(X.shape[0])
        out = np.zeros((X.shape[0], sum(c.size for c in self.categories_)))
        offset = 0
        for j, cats in enumerate(self.categories_):  # loop over columns only
            col = X[:, j]
            pos = np.clip(np.searchsorted(cats, col), 0, cats.size - 1)
            known = cats[pos] == col
            if not known.all() and self.handle_unknown == "error":
                raise ValueError(f"unknown categories {np.unique(col[~known]).tolist()} in column {j}")
            out[rows[known], offset + pos[known]] = 1
            offset += cats.size
        return out
