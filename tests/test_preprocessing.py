"""Tests for numcompute.preprocessing. Run with:  python -m pytest tests/test_preprocessing.py"""

import numpy as np
import pytest

from numcompute import io
from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


def test_standard_scaler():
    X = np.array([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
    Z = StandardScaler().fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0)
    assert np.allclose(Z[:, 0].std(), 1)
    assert np.all(Z[:, 1] == 0)  # constant column: no division by zero


def test_scaler_keeps_nan():
    Z = StandardScaler().fit_transform([[1.0], [np.nan], [3.0]])
    assert np.isnan(Z[1, 0]) and Z[0, 0] == -1 and Z[2, 0] == 1


def test_minmax_scaler():
    Z = MinMaxScaler(feature_range=(-1, 1)).fit_transform([[0.0], [5.0], [10.0]])
    assert Z.ravel().tolist() == [-1.0, 0.0, 1.0]
    with pytest.raises(ValueError):
        MinMaxScaler((1, 0))


def test_imputer_numeric():
    X = [[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan], [100.0, 20.0]]
    assert SimpleImputer("mean").fit_transform(X)[1, 0] == 104 / 3
    assert SimpleImputer("median").fit_transform(X)[1, 0] == 3.0
    assert SimpleImputer("most_frequent").fit_transform(X)[2, 1] == 20.0
    assert SimpleImputer("constant", fill_value=-1).fit_transform(X)[1, 0] == -1


def test_imputer_text():
    X = np.array([["adl"], [""], ["syd"], ["adl"]])
    out = SimpleImputer("most_frequent", missing_values="").fit_transform(X)
    assert out[:, 0].tolist() == ["adl", "adl", "syd", "adl"]
    with pytest.raises(ValueError):
        SimpleImputer("mean", missing_values="").fit(X)


def test_one_hot():
    enc = OneHotEncoder().fit([["b"], ["a"], ["b"]])
    assert enc.transform([["a"], ["b"]]).tolist() == [[1, 0], [0, 1]]
    with pytest.raises(ValueError):
        enc.transform([["c"]])
    ignore = OneHotEncoder(handle_unknown="ignore").fit([["a"], ["b"]])
    assert ignore.transform([["c"]]).tolist() == [[0, 0]]


def test_errors():
    with pytest.raises(RuntimeError):  # not fitted
        StandardScaler().transform([[1.0]])
    fitted = StandardScaler().fit(np.ones((3, 2)))
    with pytest.raises(ValueError):  # wrong number of features
        fitted.transform(np.ones((3, 3)))
    with pytest.raises(ValueError):  # 1-D input
        StandardScaler().fit(np.ones(3))
    with pytest.raises(ValueError):  # no rows
        StandardScaler().fit(np.empty((0, 2)))
    with pytest.raises(ValueError):
        OneHotEncoder().fit([[1.0], [np.nan]])


def test_titanic(titanic_path):
    X_numeric = io.load_csv(titanic_path, usecols=["Age", "Fare"])
    X_imputed = SimpleImputer("median").fit_transform(X_numeric)
    assert np.isnan(X_numeric).any()
    assert not np.isnan(X_imputed).any()

    Z = StandardScaler().fit_transform(X_imputed)
    assert np.allclose(Z.mean(axis=0), 0)
    assert np.allclose(Z.std(axis=0), 1)

    X_text = io.load_csv(titanic_path, usecols=["Sex", "Embarked"], dtype=str)
    X_text = SimpleImputer("most_frequent", missing_values="").fit_transform(X_text)
    encoder = OneHotEncoder().fit(X_text)
    assert [c.tolist() for c in encoder.categories_] == [["female", "male"], ["C", "Q", "S"]]
    one_hot = encoder.transform(X_text)
    assert one_hot.shape == (len(X_text), 5)
    assert np.all(one_hot.sum(axis=1) == 2)  # exactly one 1 per original column
