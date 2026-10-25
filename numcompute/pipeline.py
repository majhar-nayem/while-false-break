"""Transformer/Estimator protocol and composition (Pipeline, FeatureUnion).

The protocol mirrors scikit-learn's so that components are interchangeable:

* **Transformer** - ``fit(X, y=None) -> self``, ``transform(X) -> X_out``,
  and ``fit_transform`` provided by :class:`BaseTransformer`.
* **Estimator**   - ``fit(X, y) -> self``, ``predict(X) -> y_pred``.

Fitted state lives in attributes ending with ``_`` (e.g. ``mean_``). They
are created only by ``fit``, which is how :func:`check_is_fitted` tells
whether an object is fitted.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from numcompute.utils import NotFittedError

__all__ = [
    "Transformer",
    "Estimator",
    "BaseTransformer",
    "BaseEstimator",
    "check_is_fitted",
    "Pipeline",
    "Compose",
    "FeatureUnion",
    "ColumnSelector",
    "FunctionTransformer",
]


@runtime_checkable
class Transformer(Protocol):
    """Structural type of a transformer: anything with ``fit`` and ``transform``.

    Used for ``isinstance`` checks (``isinstance(obj, Transformer)``); no
    inheritance is needed. Subclass :class:`BaseTransformer` to also get
    ``fit_transform``, ``get_params`` and ``set_params``.

    Notes
    -----
    Complexity: defined by the implementing class.
    """

    def fit(self, X, y=None):
        """Learn parameters from ``X`` and return ``self``."""
        ...

    def transform(self, X):
        """Return the transformed ``X``."""
        ...


@runtime_checkable
class Estimator(Protocol):
    """Structural type of a predictive model: anything with ``fit`` and ``predict``.

    Notes
    -----
    Complexity: defined by the implementing class.
    """

    def fit(self, X, y):
        """Learn from ``X`` and targets ``y`` and return ``self``."""
        ...

    def predict(self, X):
        """Return predictions for ``X``."""
        ...


def check_is_fitted(obj, attributes: str | Sequence[str] | None = None) -> None:
    """Raise :class:`NotFittedError` if ``obj`` has not been fitted.

    Parameters
    ----------
    obj : object
        Transformer or estimator.
    attributes : str or sequence of str, optional
        Attributes that ``fit`` sets. By default, any public attribute
        ending in ``_`` counts as evidence of fitting.

    Raises
    ------
    NotFittedError
        If the attributes are missing.

    Returns
    -------
    None

    Notes
    -----
    Complexity: O(a) for a attributes (O(#attributes of obj) by default).
    """
    if attributes is None:
        fitted = any(k.endswith("_") and not k.startswith("_") for k in vars(obj))
    else:
        attrs = [attributes] if isinstance(attributes, str) else attributes
        fitted = all(hasattr(obj, a) for a in attrs)
    if not fitted:
        raise NotFittedError(
            f"This {type(obj).__name__} instance is not fitted yet; call fit() first"
        )


class _ParamsMixin:
    """``get_params``/``set_params`` and a readable ``repr`` from ``__init__``."""

    @classmethod
    def _param_names(cls) -> list[str]:
        sig = inspect.signature(cls.__init__)
        return [
            p.name
            for p in sig.parameters.values()
            if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
        ]

    def get_params(self) -> dict:
        """Return constructor parameters as a dict.

        Returns
        -------
        dict
            Mapping of parameter name to current value.
        """
        return {k: getattr(self, k) for k in self._param_names() if hasattr(self, k)}

    def set_params(self, **params):
        """Set constructor parameters.

        Parameters
        ----------
        **params
            New parameter values.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            For unknown parameter names.
        """
        valid = set(self._param_names())
        for k, v in params.items():
            if k not in valid:
                raise ValueError(f"invalid parameter {k!r} for {type(self).__name__}")
            setattr(self, k, v)
        return self

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v!r}" for k, v in self.get_params().items())
        return f"{type(self).__name__}({args})"


class BaseTransformer(_ParamsMixin):
    """Base class for transformers: implement ``fit`` and ``transform``.

    Notes
    -----
    Complexity: Defined by each subclass; ``fit_transform`` costs one ``fit`` plus one
    ``transform``.
    """

    def fit(self, X, y=None):  # pragma: no cover - abstract
        """Learn parameters from ``X``; return ``self``.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Training data.
        y : array_like, optional
            Ignored by unsupervised transformers.

        Returns
        -------
        self

        Notes
        -----
        Complexity: Defined by the subclass.
        """
        raise NotImplementedError

    def transform(self, X):  # pragma: no cover - abstract
        """Apply the learned transformation to ``X``.

        Parameters
        ----------
        X : array_like, shape (m, d)
            Data to transform.

        Returns
        -------
        X_out : ndarray

        Notes
        -----
        Complexity: Defined by the subclass.
        """
        raise NotImplementedError

    def fit_transform(self, X, y=None):
        """Fit to ``X`` then transform it.

        Parameters
        ----------
        X : array_like
            Training data.
        y : array_like, optional
            Ignored by unsupervised transformers.

        Returns
        -------
        X_out : ndarray

        Notes
        -----
        Complexity: Cost of ``fit`` plus ``transform``.
        """
        return self.fit(X, y).transform(X)


class BaseEstimator(_ParamsMixin):
    """Base class for predictive models: implement ``fit`` and ``predict``.

    Notes
    -----
    Complexity: Defined by each subclass.
    """

    def fit(self, X, y):  # pragma: no cover - abstract
        """Learn from ``X`` and targets ``y``; return ``self``.

        Parameters
        ----------
        X : array_like, shape (n, d)
            Training data.
        y : array_like, shape (n,)
            Targets.

        Returns
        -------
        self

        Notes
        -----
        Complexity: Defined by the subclass.
        """
        raise NotImplementedError

    def predict(self, X):  # pragma: no cover - abstract
        """Predict targets for ``X``.

        Parameters
        ----------
        X : array_like, shape (m, d)
            Data to predict for.

        Returns
        -------
        y_pred : ndarray, shape (m,)

        Notes
        -----
        Complexity: Defined by the subclass.
        """
        raise NotImplementedError

    def fit_predict(self, X, y):
        """Fit then predict on the same data.

        Parameters
        ----------
        X : array_like
        y : array_like

        Returns
        -------
        y_pred : ndarray

        Notes
        -----
        Complexity: Cost of ``fit`` plus ``predict``.
        """
        return self.fit(X, y).predict(X)


def _is_transformer(step) -> bool:
    return hasattr(step, "fit") and hasattr(step, "transform")


def _validate_steps(steps, kind: str):
    if not isinstance(steps, (list, tuple)) or not steps:
        raise ValueError(f"{kind} needs a non-empty list of (name, object) tuples")
    names = []
    for item in steps:
        if not (
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        ):
            raise TypeError(
                f"each {kind} step must be a (str, object) tuple, got {item!r}"
            )
        names.append(item[0])
    dupes = {n for n in names if names.count(n) > 1}
    if dupes:
        raise ValueError(
            f"{kind} step names must be unique; duplicated: {sorted(dupes)}"
        )
    bad = [n for n in names if "__" in n]
    if bad:
        raise ValueError(f"step names must not contain '__': {bad}")
    return list(steps)


class Pipeline(_ParamsMixin):
    """Chain transformers, optionally ending with an estimator.

    Parameters
    ----------
    steps : list of (str, object)
        Every step but the last must be a transformer. The last may be a
        transformer, an estimator, or ``"passthrough"``/``None`` (no-op).

    Attributes
    ----------
    named_steps : dict
        Step objects by name.

    Raises
    ------
    TypeError
        If a step is not a ``(name, object)`` tuple or an intermediate step
        lacks ``fit``/``transform``.
    ValueError
        If names are duplicated or the list is empty.

    Examples
    --------
    >>> from numcompute.preprocessing import StandardScaler, MinMaxScaler
    >>> pipe = Pipeline([("std", StandardScaler()), ("mm", MinMaxScaler())])
    >>> pipe.fit_transform([[1.0], [2.0], [3.0]]).ravel().tolist()
    [0.0, 0.5, 1.0]

    Notes
    -----
    Complexity: Each method costs the sum of the corresponding step costs; the pipeline
    adds O(s) overhead for s steps.
    """

    def __init__(self, steps):
        self.steps = _validate_steps(steps, "Pipeline")
        for name, step in self.steps[:-1]:
            if not _is_transformer(step):
                raise TypeError(
                    f"intermediate step {name!r} ({type(step).__name__}) must "
                    "implement fit and transform"
                )

    @property
    def named_steps(self) -> dict:
        """Steps as a ``{name: object}`` dict.

        Returns
        -------
        dict

        Notes
        -----
        Complexity: O(s) for s steps.
        """
        return dict(self.steps)

    @property
    def _final(self):
        step = self.steps[-1][1]
        return None if step in (None, "passthrough") else step

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, key):
        """Return a step by index or name, or a sub-pipeline for a slice."""
        if isinstance(key, slice):
            return Pipeline(self.steps[key])
        if isinstance(key, str):
            return self.named_steps[key]
        return self.steps[key][1]

    def _transform_head(self, X, fit: bool, y=None):
        for _, step in self.steps[:-1]:
            if not fit:
                X = step.transform(X)
            elif hasattr(step, "fit_transform"):
                X = step.fit_transform(X, y)
            else:  # the protocol only requires fit + transform
                X = step.fit(X, y).transform(X)
        return X

    def fit(self, X, y=None):
        """Fit every step in order on the output of the previous one.

        Parameters
        ----------
        X : array_like
            Training data.
        y : array_like, optional
            Targets, passed to every step's ``fit``.

        Returns
        -------
        self : Pipeline

        Notes
        -----
        Complexity: Sum of the steps' ``fit_transform``/``fit`` costs.
        """
        Xt = self._transform_head(X, fit=True, y=y)
        if self._final is not None:
            self._final.fit(Xt, y)
        return self

    def transform(self, X):
        """Apply all transformers (the last step must be a transformer).

        Parameters
        ----------
        X : array_like

        Returns
        -------
        X_out : ndarray

        Raises
        ------
        AttributeError
            If the final step has no ``transform``.

        Notes
        -----
        Complexity: Sum of the steps' ``transform`` costs.
        """
        Xt = self._transform_head(X, fit=False)
        if self._final is None:
            return Xt
        if not hasattr(self._final, "transform"):
            raise AttributeError(f"final step {self.steps[-1][0]!r} has no transform()")
        return self._final.transform(Xt)

    def fit_transform(self, X, y=None):
        """Fit all steps and return the transformed training data.

        Parameters
        ----------
        X : array_like
        y : array_like, optional

        Returns
        -------
        X_out : ndarray

        Notes
        -----
        Complexity: Sum of the steps' ``fit_transform`` costs.
        """
        Xt = self._transform_head(X, fit=True, y=y)
        if self._final is None:
            return Xt
        if hasattr(self._final, "fit_transform"):
            return self._final.fit_transform(Xt, y)
        return self._final.fit(Xt, y).transform(Xt)

    def predict(self, X):
        """Transform ``X`` and call the final estimator's ``predict``.

        Parameters
        ----------
        X : array_like

        Returns
        -------
        y_pred : ndarray

        Raises
        ------
        AttributeError
            If the final step has no ``predict``.

        Notes
        -----
        Complexity: Sum of the transformers' ``transform`` costs plus the estimator's
        ``predict``.
        """
        if not hasattr(self._final, "predict"):
            raise AttributeError(f"final step {self.steps[-1][0]!r} has no predict()")
        return self._final.predict(self._transform_head(X, fit=False))

    def fit_predict(self, X, y):
        """Fit the pipeline and predict on the training data.

        Parameters
        ----------
        X : array_like
        y : array_like

        Returns
        -------
        y_pred : ndarray

        Notes
        -----
        Complexity: Cost of ``fit`` plus ``predict``.
        """
        return self.fit(X, y).predict(X)

    def get_params(self) -> dict:
        """Return ``{"steps": steps}``.

        Returns
        -------
        dict

        Notes
        -----
        Complexity: O(1).
        """
        return {"steps": self.steps}


class Compose(Pipeline):
    """Chain unnamed transformations: ``Compose([f, g, h])(X) == h(g(f(X)))``.

    A lighter-weight :class:`Pipeline` for when step names are not needed.
    Steps are named ``"step0"``, ``"step1"``, ... and plain callables (e.g.
    ``np.log1p``) are wrapped in :class:`FunctionTransformer`, so stateless
    functions and fitted transformers can be mixed. A Compose can be called
    directly (``compose(X)``) once fitted.

    Parameters
    ----------
    transforms : sequence of transformer or callable
        Applied left to right. The last element may also be an estimator.

    Raises
    ------
    TypeError
        If an element is neither a transformer/estimator nor callable.
    ValueError
        If ``transforms`` is empty.

    Notes
    -----
    Complexity: the sum of the steps' costs.

    Examples
    --------
    >>> from numcompute.preprocessing import MinMaxScaler
    >>> c = Compose([np.log10, MinMaxScaler()])
    >>> c.fit_transform([[1.0], [10.0], [100.0]]).ravel().tolist()
    [0.0, 0.5, 1.0]
    """

    def __init__(self, transforms):
        transforms = list(transforms) if transforms is not None else []
        if not transforms:
            raise ValueError("Compose needs at least one transform")
        steps = []
        for i, t in enumerate(transforms):
            if not hasattr(t, "fit"):
                if not callable(t):
                    raise TypeError(
                        f"transform {i} ({type(t).__name__}) is neither a "
                        "transformer nor callable"
                    )
                t = FunctionTransformer(t)
            steps.append((f"step{i}", t))
        self.transforms = transforms
        super().__init__(steps)

    def __call__(self, X):
        """Apply the fitted chain to ``X`` (same as :meth:`transform`).

        Parameters
        ----------
        X : array_like

        Returns
        -------
        X_out : ndarray

        Notes
        -----
        Complexity: the sum of the steps' ``transform`` costs.
        """
        return self.transform(X)

    def get_params(self) -> dict:
        """Return ``{"transforms": transforms}``.

        Returns
        -------
        dict

        Notes
        -----
        Complexity: O(1).
        """
        return {"transforms": self.transforms}


class FeatureUnion(BaseTransformer):
    """Apply several transformers to the same input and concatenate columns.

    Parameters
    ----------
    transformer_list : list of (str, transformer)
        Each transformer must output a 2-D array with ``n`` rows.

    Raises
    ------
    TypeError
        If an entry is not a ``(name, transformer)`` tuple.
    ValueError
        If names are duplicated or the list is empty.

    Notes
    -----
    Complexity: the sum of the parts plus O(n * total_features) to stack.

    Examples
    --------
    >>> from numcompute.preprocessing import StandardScaler, MinMaxScaler
    >>> fu = FeatureUnion([("std", StandardScaler()), ("mm", MinMaxScaler())])
    >>> fu.fit_transform([[1.0], [3.0]]).tolist()
    [[-1.0, 0.0], [1.0, 1.0]]
    """

    def __init__(self, transformer_list):
        self.transformer_list = _validate_steps(transformer_list, "FeatureUnion")
        for name, t in self.transformer_list:
            if not _is_transformer(t):
                raise TypeError(f"{name!r} must implement fit and transform")

    def fit(self, X, y=None):
        """Fit every transformer on ``X``.

        Parameters
        ----------
        X : array_like
        y : array_like, optional

        Returns
        -------
        self : FeatureUnion

        Notes
        -----
        Complexity: Sum of the transformers' ``fit`` costs.
        """
        for _, t in self.transformer_list:
            t.fit(X, y)
        self.n_outputs_ = len(self.transformer_list)
        return self

    def transform(self, X):
        """Transform ``X`` with every transformer and hstack the results.

        Parameters
        ----------
        X : array_like

        Returns
        -------
        X_out : ndarray, shape (n, sum of output widths)

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        ValueError
            If the parts have different numbers of rows.

        Notes
        -----
        Complexity: Sum of the transformers' ``transform`` costs plus O(n w) to stack w
        output columns.
        """
        check_is_fitted(self, "n_outputs_")
        parts = [np.asarray(t.transform(X)) for _, t in self.transformer_list]
        parts = [p.reshape(-1, 1) if p.ndim == 1 else p for p in parts]
        rows = {p.shape[0] for p in parts}
        if len(rows) > 1:
            raise ValueError(
                f"transformers returned different row counts: {sorted(rows)}"
            )
        if len({p.dtype.kind for p in parts}) > 1:
            parts = [p.astype(object) for p in parts]
        return np.hstack(parts)


class ColumnSelector(BaseTransformer):
    """Select columns of a 2-D array (use inside a FeatureUnion branch).

    Parameters
    ----------
    columns : int or sequence of int
        Column indices to keep.
    dtype : data-type, optional
        Cast the selection (e.g. ``float`` for numeric columns read from a
        mixed object table).

    Examples
    --------
    >>> ColumnSelector([0, 2]).fit_transform([[1, 2, 3], [4, 5, 6]]).tolist()
    [[1, 3], [4, 6]]

    Notes
    -----
    Complexity: O(n c) time and memory to copy c selected columns.
    """

    def __init__(self, columns, dtype=None):
        self.columns = columns
        self.dtype = dtype

    def fit(self, X, y=None):
        """Record the input width and validate the column indices.

        Parameters
        ----------
        X : array_like, shape (n, d)
        y : ignored

        Returns
        -------
        self : ColumnSelector

        Raises
        ------
        ValueError
            If ``X`` is not 2-D or an index is out of range.

        Notes
        -----
        Complexity: O(c) to validate c column indices.
        """
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        cols = np.atleast_1d(np.asarray(self.columns))
        if cols.size and (cols.min() < -X.shape[1] or cols.max() >= X.shape[1]):
            raise ValueError(
                f"columns {cols.tolist()} out of range for {X.shape[1]} columns"
            )
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        """Return the selected columns.

        Parameters
        ----------
        X : array_like, shape (n, d)

        Returns
        -------
        X_out : ndarray, shape (n, len(columns))

        Raises
        ------
        NotFittedError
            If called before ``fit``.
        ValueError
            If ``X`` has a different number of columns than at fit time.

        Notes
        -----
        Complexity: O(n c) time and memory.
        """
        check_is_fitted(self, "n_features_in_")
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has shape {X.shape}, expected (n, {self.n_features_in_})"
            )
        out = X[:, np.atleast_1d(np.asarray(self.columns))]
        return out.astype(self.dtype) if self.dtype is not None else out


class FunctionTransformer(BaseTransformer):
    """Wrap a stateless function as a transformer.

    Parameters
    ----------
    func : callable, optional
        ``func(X) -> X_out``; ``None`` is the identity.

    Examples
    --------
    >>> FunctionTransformer(np.log1p).fit_transform([[0.0]]).tolist()
    [[0.0]]

    Notes
    -----
    Complexity: The cost of ``func``.
    """

    def __init__(self, func: Callable | None = None):
        self.func = func

    def fit(self, X, y=None):
        """No-op; returns ``self``.

        Parameters
        ----------
        X : array_like
        y : ignored

        Returns
        -------
        self : FunctionTransformer

        Notes
        -----
        Complexity: O(1).
        """
        self.fitted_ = True
        return self

    def transform(self, X):
        """Apply ``func`` to ``X``.

        Parameters
        ----------
        X : array_like

        Returns
        -------
        X_out : ndarray

        Notes
        -----
        Complexity: The cost of ``func``.
        """
        return np.asarray(X) if self.func is None else self.func(np.asarray(X))
