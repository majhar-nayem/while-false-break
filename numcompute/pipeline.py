"""Transformer / Estimator protocol, Pipeline, Compose and FeatureUnion.

The protocol is just method names:
    Transformer (preprocessing step): fit(X, y=None) -> self, transform(X) -> X_new
    Estimator   (model):              fit(X, y) -> self,      predict(X) -> y_pred
Any object with these methods can be used as a step (duck typing); the base
classes below only add fit_transform / fit_predict for free.
"""

import numpy as np

from numcompute.preprocessing import Transformer

__all__ = ["ColumnSelector", "Compose", "Estimator", "FeatureUnion", "Pipeline", "Transformer"]


class Estimator:
    """Base class for models: subclasses implement fit(X, y) and predict(X)."""

    def fit(self, X, y):
        """Learn from X and y and return self (implemented by subclasses)."""
        raise NotImplementedError

    def predict(self, X):
        """Return predictions for X (implemented by subclasses)."""
        raise NotImplementedError

    def fit_predict(self, X, y):
        """Fit to (X, y), then predict X."""
        return self.fit(X, y).predict(X)


def _check_steps(steps):
    """Check that steps is a non-empty list of (name, step) pairs with unique names."""
    if not isinstance(steps, (list, tuple)) or len(steps) == 0:
        raise ValueError("steps must be a non-empty list of (name, step) pairs")
    names = []
    for item in steps:
        if not (isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)):
            raise ValueError(f"each step must be a (name, step) pair, got {item!r}")
        names.append(item[0])
    if len(set(names)) != len(names):
        raise ValueError(f"step names must be unique, got {names}")
    return list(steps)


class Pipeline:
    """
    Chain steps so that each one is fitted on the output of the one before.

    Example (from the specification):
        pipe = Pipeline([("scale", StandardScaler()), ("encode", OneHotEncoder())])
        X_tr = pipe.fit_transform(X)

    Parameters:
        steps : list of (name, step) pairs. Every step except the last must
            be a transformer (fit + transform). The last step can be a
            transformer or an estimator (fit + predict).

    Raises:
        ValueError if steps is empty, a name is repeated, or a middle step
            has no transform method.
        TypeError if a step has no fit method.

    Use pipe["name"] to get a step back, e.g. to read a fitted scaler's mean_.

    Complexity:
        the sum of the steps' own costs.
    """

    def __init__(self, steps):
        self.steps = _check_steps(steps)
        for name, step in self.steps:
            if not hasattr(step, "fit"):
                raise TypeError(f"step {name!r} has no fit method")
        for name, step in self.steps[:-1]:
            if not hasattr(step, "transform"):
                raise ValueError(
                    f"step {name!r} must have a transform method "
                    "(only the last step can be a model)"
                )

    def __getitem__(self, name):
        """Return the step with this name."""
        return dict(self.steps)[name]

    def __repr__(self):
        names = [name for name, _ in self.steps]
        return "Pipeline(" + " -> ".join(names) + ")"

    def _last(self):
        """Return the last step (a transformer or a model)."""
        return self.steps[-1][1]

    def _transform_head(self, X, y=None, fit=False):
        """Run X through every step except the last (fitting each one if fit=True)."""
        for _, step in self.steps[:-1]:
            if fit:
                step.fit(X, y)
            X = step.transform(X)
        return X

    def fit(self, X, y=None):
        """
        Fit every step in order.

        Parameters:
            X : array_like, input data
            y : array_like, optional, targets (passed to every step's fit)

        Returns:
            self
        """
        X = self._transform_head(X, y, fit=True)
        self._last().fit(X, y)
        return self

    def transform(self, X):
        """
        Apply every fitted step to X. The last step must be a transformer.

        Returns:
            the output of the last step

        Raises:
            TypeError if the last step has no transform method.
        """
        if not hasattr(self._last(), "transform"):
            raise TypeError("the last step has no transform method; use predict()")
        X = self._transform_head(X)
        return self._last().transform(X)

    def fit_transform(self, X, y=None):
        """Fit the pipeline, then transform X."""
        return self.fit(X, y).transform(X)

    def predict(self, X):
        """
        Transform X with every step except the last, then call the last step's predict.

        Raises:
            TypeError if the last step has no predict method.
        """
        if not hasattr(self._last(), "predict"):
            raise TypeError("the last step has no predict method; use transform()")
        X = self._transform_head(X)
        return self._last().predict(X)


class Compose:
    """
    Chain plain functions with no fitting: Compose([f, g])(X) == g(f(X)).

    Useful for fixed transforms such as np.log1p, or the .transform of an
    already fitted transformer. Use Pipeline when the steps need fitting.

    Parameters:
        functions : non-empty list of callables

    Raises:
        TypeError if an item is not callable.
        ValueError if the list is empty.

    Complexity:
        the sum of the functions' costs.
    """

    def __init__(self, functions):
        functions = list(functions)
        if not functions:
            raise ValueError("functions must not be empty")
        for function in functions:
            if not callable(function):
                raise TypeError(f"{function!r} is not callable")
        self.functions = functions

    def __call__(self, X):
        """Apply each function in turn to X."""
        for function in self.functions:
            X = function(X)
        return X


class FeatureUnion(Transformer):
    """
    Run several transformers on the same input and put their outputs side by side.

    Example: scaled numeric columns next to one-hot encoded text columns,
    with each branch starting with a ColumnSelector.

    Parameters:
        transformers : list of (name, transformer) pairs; each output must be 2-D
            with the same number of rows

    Returns (transform):
        ndarray, shape (n, total number of output columns)

    Raises:
        ValueError if the list is empty or names are repeated.

    Complexity:
        the sum of the transformers' costs, plus O(n * total columns) to join.
    """

    def __init__(self, transformers):
        self.transformers = _check_steps(transformers)

    def __getitem__(self, name):
        """Return the transformer with this name."""
        return dict(self.transformers)[name]

    def fit(self, X, y=None):
        """Fit every transformer on the same X. Returns self."""
        for _, transformer in self.transformers:
            transformer.fit(X, y)
        self.n_features_in_ = np.asarray(X).shape[1]
        return self

    def transform(self, X):
        """Transform X with every transformer and join the outputs column-wise."""
        self._check_fitted()
        outputs = [transformer.transform(X) for _, transformer in self.transformers]
        return np.hstack(outputs)


class ColumnSelector(Transformer):
    """
    Keep only some columns of a 2-D array, optionally converting their type.

    Parameters:
        columns : list of int, column indices to keep
        dtype : optional, e.g. float or str, type to convert the selected columns to

    Returns (transform):
        ndarray, shape (n, len(columns))

    Raises:
        ValueError if X is not 2-D or a column index is out of range.

    Handy when one array mixes numbers and text (dtype=object): one branch
    selects the number columns with dtype=float, another the text with dtype=str.

    Complexity:
        O(n * len(columns)) time and space.
    """

    def __init__(self, columns, dtype=None):
        self.columns = list(columns)
        self.dtype = dtype

    def fit(self, X, y=None):
        """Check that X is 2-D and every column index exists. Returns self."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        for column in self.columns:
            if not 0 <= column < X.shape[1]:
                raise ValueError(f"column {column} out of range for {X.shape[1]} columns")
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return the selected columns of X, converted to dtype if one was given."""
        self._check_fitted()
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X must be 2-D with {self.n_features_in_} columns, got shape {X.shape}"
            )
        selected = X[:, self.columns]
        if self.dtype is not None:
            selected = selected.astype(self.dtype)
        return selected
