"""Tests for numcompute.pipeline."""

import numpy as np
import pytest

from numcompute.pipeline import (
    BaseEstimator,
    ColumnSelector,
    FeatureUnion,
    FunctionTransformer,
    Pipeline,
)
from numcompute.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)
from numcompute.utils import NotFittedError


class MeanThresholdClassifier(BaseEstimator):
    """Toy estimator: predicts 1 when the row mean exceeds a learned threshold."""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        self.threshold_ = X.mean(axis=1)[np.asarray(y) == 1].min()
        return self

    def predict(self, X):
        return (np.asarray(X, dtype=float).mean(axis=1) >= self.threshold_).astype(int)


def test_spec_example_scale_then_encode():
    X = np.array([[1.0], [2.0], [1.0]])
    pipe = Pipeline([("scale", StandardScaler()), ("encode", OneHotEncoder())])
    out = pipe.fit_transform(X)
    np.testing.assert_array_equal(out, [[1, 0], [0, 1], [1, 0]])


def test_pipeline_equals_manual_chain(rng):
    X = rng.normal(size=(20, 3))
    X[3, 1] = np.nan
    pipe = Pipeline([("impute", SimpleImputer()), ("scale", StandardScaler())])
    manual = StandardScaler().fit_transform(SimpleImputer().fit_transform(X))
    np.testing.assert_allclose(pipe.fit_transform(X), manual)
    np.testing.assert_allclose(pipe.transform(X), manual)  # fit state reused


def test_pipeline_with_estimator_predict():
    X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0], [6.0, 6.0]])
    y = np.array([0, 0, 1, 1])
    pipe = Pipeline([("scale", MinMaxScaler()), ("clf", MeanThresholdClassifier())])
    np.testing.assert_array_equal(pipe.fit(X, y).predict(X), y)
    np.testing.assert_array_equal(pipe.fit_predict(X, y), y)


def test_access_and_slicing():
    pipe = Pipeline(
        [("a", StandardScaler()), ("b", MinMaxScaler()), ("c", "passthrough")]
    )
    assert len(pipe) == 3
    assert isinstance(pipe["a"], StandardScaler)
    assert isinstance(pipe[1], MinMaxScaler)
    assert isinstance(pipe[:2], Pipeline)
    X = np.array([[1.0], [3.0]])
    np.testing.assert_allclose(pipe.fit_transform(X), [[0.0], [1.0]])


def test_pipeline_validation():
    with pytest.raises(ValueError, match="unique"):
        Pipeline([("a", StandardScaler()), ("a", MinMaxScaler())])
    with pytest.raises(TypeError, match="fit and transform"):
        Pipeline([("clf", MeanThresholdClassifier()), ("scale", StandardScaler())])
    with pytest.raises(ValueError):
        Pipeline([])
    with pytest.raises(NotFittedError):
        Pipeline([("s", StandardScaler())]).transform([[1.0]])
    with pytest.raises(AttributeError, match="predict"):
        Pipeline([("s", StandardScaler())]).fit([[1.0]]).predict([[1.0]])


def test_feature_union_mixed_columns():
    raw = np.array(
        [[25, 50000, "adl"], [32, 61000, "syd"], [41, 72000, "adl"]], dtype=object
    )
    num = Pipeline(
        [("sel", ColumnSelector([0, 1], dtype=float)), ("scale", StandardScaler())]
    )
    cat = Pipeline([("sel", ColumnSelector([2])), ("ohe", OneHotEncoder())])
    out = FeatureUnion([("num", num), ("cat", cat)]).fit_transform(raw)
    assert out.shape == (3, 4)
    np.testing.assert_array_equal(out[:, 2:], [[1, 0], [0, 1], [1, 0]])
    np.testing.assert_allclose(out[:, :2].astype(float).mean(axis=0), 0, atol=1e-12)


def test_function_transformer_and_params():
    ft = FunctionTransformer(np.log1p)
    np.testing.assert_allclose(ft.fit_transform([[0.0, np.e - 1]]), [[0.0, 1.0]])
    sc = StandardScaler(with_mean=False)
    assert sc.get_params() == {"with_mean": False, "with_std": True}
    assert "with_mean=False" in repr(sc)
    sc.set_params(with_mean=True)
    assert sc.with_mean is True
    with pytest.raises(ValueError, match="invalid parameter"):
        sc.set_params(bogus=1)


def test_column_selector_errors():
    with pytest.raises(ValueError, match="out of range"):
        ColumnSelector([5]).fit(np.ones((2, 2)))
    sel = ColumnSelector([0]).fit(np.ones((2, 2)))
    with pytest.raises(ValueError, match="expected"):
        sel.transform(np.ones((2, 3)))


def test_compose_mixes_functions_and_transformers():
    from numcompute.pipeline import Compose

    c = Compose([np.log10, MinMaxScaler()])
    np.testing.assert_allclose(
        c.fit_transform([[1.0], [10.0], [100.0]]), [[0], [0.5], [1]]
    )
    np.testing.assert_allclose(c([[1000.0]]), [[1.5]])  # callable once fitted
    assert [name for name, _ in c.steps] == ["step0", "step1"]
    with pytest.raises(ValueError):
        Compose([])
    with pytest.raises(TypeError, match="neither"):
        Compose([42])


def test_protocols_are_structural():
    from numcompute.pipeline import Estimator, Transformer

    assert isinstance(StandardScaler(), Transformer)
    assert isinstance(Pipeline([("s", StandardScaler())]), Transformer)
    assert isinstance(MeanThresholdClassifier(), Estimator)
    assert not isinstance(MeanThresholdClassifier(), Transformer)

    class Duck:  # no inheritance needed
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    assert isinstance(Duck(), Transformer)
    Pipeline([("duck", Duck()), ("s", StandardScaler())]).fit_transform([[1.0], [2.0]])


class Doubler:
    """Duck-typed transformer with only fit/transform (no fit_transform)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float) * 2


def test_passthrough_and_duck_typed_final_steps():
    X = np.array([[1.0], [3.0]])
    for final in (None, "passthrough"):
        pipe = Pipeline([("scale", StandardScaler()), ("end", final)])
        np.testing.assert_allclose(pipe.fit_transform(X), [[-1], [1]])
        np.testing.assert_allclose(pipe.transform(X), [[-1], [1]])
    pipe = Pipeline([("a", Doubler()), ("b", Doubler())])
    np.testing.assert_allclose(pipe.fit_transform(X), 4 * X)


def test_estimator_helpers_and_params():
    from numcompute.pipeline import Compose, check_is_fitted

    X, y = np.array([[0.0], [1.0], [5.0], [6.0]]), np.array([0, 0, 1, 1])
    clf = MeanThresholdClassifier()
    with pytest.raises(NotFittedError):
        check_is_fitted(clf)  # no attribute ending in "_" yet
    np.testing.assert_array_equal(clf.fit_predict(X, y), y)
    check_is_fitted(clf)
    pipe = Pipeline([("s", StandardScaler()), ("clf", MeanThresholdClassifier())])
    with pytest.raises(AttributeError, match="no transform"):
        pipe.fit(X, y).transform(X)
    assert pipe.get_params()["steps"] is pipe.steps
    comp = Compose([np.log1p])
    assert comp.get_params()["transforms"] == [np.log1p]


def test_step_and_union_validation():
    with pytest.raises(TypeError, match=r"\(str, object\) tuple"):
        Pipeline([StandardScaler()])
    with pytest.raises(ValueError, match="'__'"):
        Pipeline([("a__b", StandardScaler())])
    with pytest.raises(TypeError, match="fit and transform"):
        FeatureUnion([("bad", object())])
    shrink = FunctionTransformer(lambda X: X[:1])
    union = FeatureUnion([("all", FunctionTransformer()), ("one", shrink)])
    with pytest.raises(ValueError, match="different row counts"):
        union.fit_transform(np.ones((3, 2)))
    with pytest.raises(ValueError, match="2-D"):
        ColumnSelector([0]).fit(np.ones(3))


def test_feature_union_mixed_dtypes_become_object():
    X = np.array([[1, "a"], [2, "b"]], dtype=object)
    union = FeatureUnion(
        [("num", ColumnSelector([0], dtype=float)), ("cat", ColumnSelector([1]))]
    )
    out = union.fit_transform(X)
    assert out.dtype == object and out[1, 0] == 2.0 and out[1, 1] == "b"
