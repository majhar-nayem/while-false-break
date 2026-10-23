"""Tests for numcompute.preprocessing. Run with:  python test_preprocessing.py"""

import os

import numpy as np

from numcompute import io
from numcompute.preprocessing import MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler


def raises(error, fn, *args):
    try:
        fn(*args)
    except error:
        return True
    return False


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
    assert raises(ValueError, MinMaxScaler, (1, 0))


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
    assert raises(ValueError, SimpleImputer("mean", missing_values="").fit, X)


def test_one_hot():
    enc = OneHotEncoder().fit([["b"], ["a"], ["b"]])
    assert enc.transform([["a"], ["b"]]).tolist() == [[1, 0], [0, 1]]
    assert raises(ValueError, enc.transform, [["c"]])
    ignore = OneHotEncoder(handle_unknown="ignore").fit([["a"], ["b"]])
    assert ignore.transform([["c"]]).tolist() == [[0, 0]]


def test_errors():
    assert raises(RuntimeError, StandardScaler().transform, [[1.0]])          # not fitted
    assert raises(ValueError, StandardScaler().fit(np.ones((3, 2))).transform, np.ones((3, 3)))
    assert raises(ValueError, StandardScaler().fit, np.ones(3))               # 1-D input
    assert raises(ValueError, StandardScaler().fit, np.empty((0, 2)))         # no rows
    assert raises(ValueError, OneHotEncoder().fit, [[1.0], [np.nan]])


def test_titanic():
    path = "data/titanic.csv"
    if not os.path.exists(path):
        print("  (skipped: put titanic.csv in the data folder)")
        return
    Xn = io.load_csv(path, usecols=["Age", "Fare"])
    Xi = SimpleImputer("median").fit_transform(Xn)
    Z = StandardScaler().fit_transform(Xi)
    print("  missing before/after impute:", int(np.isnan(Xn).sum()), int(np.isnan(Xi).sum()))
    print("  scaled mean:", Z.mean(axis=0).round(6), "std:", Z.std(axis=0).round(6))

    Xc = io.load_csv(path, usecols=["Sex", "Embarked"], dtype=str)
    Xc = SimpleImputer("most_frequent", missing_values="").fit_transform(Xc)
    enc = OneHotEncoder().fit(Xc)
    print("  categories:", [c.tolist() for c in enc.categories_])
    print("  one-hot shape:", enc.transform(Xc).shape)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("All tests passed.")
