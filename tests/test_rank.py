"""Tests for numcompute.rank. Run with:  python test_rank.py"""

import os

import numpy as np

from numcompute import io
from numcompute.rank import percentile, rank


def raises(error, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except error:
        return True
    return False


def test_rank_ties():
    a = [10, 20, 20, 30]
    assert rank(a).tolist() == [1, 2.5, 2.5, 4]
    assert rank(a, method="dense").tolist() == [1, 2, 2, 3]
    assert rank(a, method="ordinal").tolist() == [1, 2, 3, 4]
    assert rank([3, 1, 3, 3]).tolist() == [3, 1, 3, 3]          # unsorted input


def test_rank_edge_cases():
    assert rank([5, 5, 5]).tolist() == [2, 2, 2]                # all equal
    assert rank([5, 5, 5], method="dense").tolist() == [1, 1, 1]
    r = rank([2.0, np.nan, 1.0])
    assert np.isnan(r[1]) and r[0] == 2 and r[2] == 1           # NaN stays NaN
    assert rank([]).size == 0
    assert np.all(np.isnan(rank([np.nan, np.nan])))
    view = np.array([4, 0, 3, 0, 1, 0])[::2]                    # non-contiguous: [4, 3, 1]
    assert rank(view).tolist() == [3, 2, 1]


def test_rank_errors():
    assert raises(ValueError, rank, [1, 2], method="min")
    assert raises(ValueError, rank, [[1, 2]])
    assert raises(TypeError, rank, ["a", "b"])


def test_percentile_matches_numpy():
    rng = np.random.default_rng(0)
    x = rng.normal(size=101)
    q = [0, 10, 25, 50, 75, 99.9, 100]
    for method in ("linear", "lower", "higher", "midpoint"):
        assert np.allclose(percentile(x, q, method), np.percentile(x, q, method=method))


def test_percentile_edge_cases():
    assert percentile([1, 2, 3, 4], 50) == 2.5
    assert percentile([1, 2, 3, 4], 50, "lower") == 2.0
    assert percentile([1, 2, 3, 4], 50, "higher") == 3.0
    assert percentile([1, 2, 3, 4], 50, "midpoint") == 2.5
    assert percentile([7], 30) == 7.0                           # single value
    assert percentile([1.0, np.nan, 3.0], 50) == 2.0            # NaN ignored
    assert percentile([[1, 2], [3, 4]], 50) == 2.5              # 2-D uses all values
    assert percentile([1, 2, 3], [0, 100]).shape == (2,)


def test_percentile_errors():
    assert raises(ValueError, percentile, [1, 2], 101)
    assert raises(ValueError, percentile, [1, 2], -1)
    assert raises(ValueError, percentile, [], 50)
    assert raises(ValueError, percentile, [np.nan], 50)
    assert raises(ValueError, percentile, [1, 2], 50, "nearest")
    assert raises(TypeError, percentile, ["a"], 50)


def test_titanic():
    path = "data/titanic.csv"
    if not os.path.exists(path):
        print("  (skipped: put titanic.csv in the data folder)")
        return
    X = io.load_csv(path, usecols=["Pclass", "Age"])
    sample = X[:6, 0]
    print("  Pclass:", sample)
    for method in ("average", "dense", "ordinal"):
        print(f"  {method:>8}:", rank(sample, method))
    print("  age ranks (NaN stays NaN):", rank(X[:8, 1]))
    for method in ("linear", "lower", "higher", "midpoint"):
        print(f"  age 25/50/75th ({method:>8}):", percentile(X[:, 1], [25, 50, 75], method))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("All tests passed.")
