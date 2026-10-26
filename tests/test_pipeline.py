"""Tests for numcompute.pipeline. Run with:  python -m pytest tests/test_pipeline.py"""

import numpy as np
import pytest

from numcompute.pipeline import ColumnSelector, Compose, Estimator, FeatureUnion, Pipeline
from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


class ThresholdModel(Estimator):
    """Tiny model for testing: predicts 1 when the first column is above its training mean."""

    def fit(self, X, y):
        self.cut_ = np.mean(X[:, 0])
        return self

    def predict(self, X):
        return (X[:, 0] > self.cut_).astype(int)


def test_spec_example_scale_then_encode():
    X = np.array([[1.0], [2.0], [1.0], [3.0]])
    pipe = Pipeline([("scale", StandardScaler()), ("encode", OneHotEncoder())])
    out = pipe.fit_transform(X)
    assert out.shape == (4, 3) and np.all(out.sum(axis=1) == 1)


def test_pipeline_equals_manual_chain():
    X = np.random.default_rng(0).normal(size=(20, 3))
    X[[2, 7], [0, 1]] = np.nan
    pipe = Pipeline([("impute", SimpleImputer("median")), ("scale", StandardScaler())])
    manual = StandardScaler().fit_transform(SimpleImputer("median").fit_transform(X))
    assert np.allclose(pipe.fit_transform(X), manual)
    X_new = np.ones((2, 3))  # reuses what was learned in fit
    assert np.allclose(
        pipe.transform(X_new), pipe["scale"].transform(pipe["impute"].transform(X_new))
    )


def test_pipeline_with_estimator():
    X = np.array([[1.0], [2.0], [3.0], [4.0]])
    pipe = Pipeline([("scale", MinMaxScaler()), ("model", ThresholdModel())])
    assert pipe.fit(X, [0, 0, 1, 1]).predict(X).tolist() == [0, 0, 1, 1]
    assert pipe["model"].cut_ == 0.5  # fitted on the scaled data
    with pytest.raises(TypeError):  # last step is a model
        pipe.transform(X)
    assert repr(pipe) == "Pipeline(scale -> model)"


def test_pipeline_validation():
    with pytest.raises(ValueError):
        Pipeline([])
    with pytest.raises(ValueError):
        Pipeline([("a", StandardScaler()), ("a", MinMaxScaler())])
    with pytest.raises(ValueError):
        Pipeline([("model", ThresholdModel()), ("scale", StandardScaler())])
    with pytest.raises(ValueError):
        Pipeline([StandardScaler()])
    with pytest.raises(TypeError):
        Pipeline([("x", 42)])
    pipe = Pipeline([("scale", StandardScaler())])
    with pytest.raises(RuntimeError):  # not fitted yet
        pipe.transform(np.ones((2, 2)))


def test_feature_union_mixed_columns():
    raw = np.array([[22.0, "male"], [np.nan, "female"], [30.0, "male"]], dtype=object)
    numeric = Pipeline(
        [("pick", ColumnSelector([0], dtype=float)), ("impute", SimpleImputer("mean"))]
    )
    text = Pipeline([("pick", ColumnSelector([1], dtype=str)), ("onehot", OneHotEncoder())])
    union = FeatureUnion([("num", numeric), ("cat", text)])
    out = union.fit_transform(raw)
    assert out.shape == (3, 3)
    assert out.tolist() == [[22.0, 0, 1], [26.0, 1, 0], [30.0, 0, 1]]
    assert union["cat"]["onehot"].categories_[0].tolist() == ["female", "male"]
    with pytest.raises(ValueError):
        FeatureUnion([])


def test_compose():
    f = Compose([np.log1p, lambda x: x * 2])
    assert np.allclose(f(np.array([0.0, np.e - 1])), [0.0, 2.0])
    scaler = MinMaxScaler().fit([[0.0], [10.0]])
    assert Compose([scaler.transform, np.ravel])([[5.0]]).tolist() == [0.5]
    with pytest.raises(TypeError):
        Compose([np.log1p, "not a function"])
    with pytest.raises(ValueError):
        Compose([])


def test_column_selector_errors():
    sel = ColumnSelector([3])
    with pytest.raises(ValueError):
        sel.fit(np.ones((2, 2)))
    with pytest.raises(ValueError):
        ColumnSelector([0]).fit(np.ones(3))
    fitted = ColumnSelector([0]).fit(np.ones((2, 2)))
    with pytest.raises(ValueError):
        fitted.transform(np.ones((2, 3)))


def test_estimator_base():
    assert ThresholdModel().fit_predict(np.array([[0.0], [2.0]]), [0, 1]).tolist() == [0, 1]
    with pytest.raises(NotImplementedError):
        Estimator().fit([[1]], [1])
