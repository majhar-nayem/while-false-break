"""Tests for numcompute.rank. Run with:  python -m pytest tests/test_rank.py"""

import numpy as np
import pytest

from numcompute import io
from numcompute.rank import percentile, rank


def test_rank_ties():
    a = [10, 20, 20, 30]
    assert rank(a).tolist() == [1, 2.5, 2.5, 4]
    assert rank(a, method="dense").tolist() == [1, 2, 2, 3]
    assert rank(a, method="ordinal").tolist() == [1, 2, 3, 4]
    assert rank([3, 1, 3, 3]).tolist() == [3, 1, 3, 3]  # unsorted input


def test_rank_edge_cases():
    assert rank([5, 5, 5]).tolist() == [2, 2, 2]  # all equal
    assert rank([5, 5, 5], method="dense").tolist() == [1, 1, 1]
    r = rank([2.0, np.nan, 1.0])
    assert np.isnan(r[1]) and r[0] == 2 and r[2] == 1  # NaN stays NaN
    assert rank([]).size == 0
    assert np.all(np.isnan(rank([np.nan, np.nan])))
    view = np.array([4, 0, 3, 0, 1, 0])[::2]  # non-contiguous: [4, 3, 1]
    assert rank(view).tolist() == [3, 2, 1]


def test_rank_errors():
    with pytest.raises(ValueError):
        rank([1, 2], method="min")
    with pytest.raises(ValueError):
        rank([[1, 2]])
    with pytest.raises(TypeError):
        rank(["a", "b"])


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
    assert percentile([7], 30) == 7.0  # single value
    assert percentile([1.0, np.nan, 3.0], 50) == 2.0  # NaN ignored
    assert percentile([[1, 2], [3, 4]], 50) == 2.5  # 2-D uses all values
    assert percentile([1, 2, 3], [0, 100]).shape == (2,)


def test_percentile_errors():
    with pytest.raises(ValueError):
        percentile([1, 2], 101)
    with pytest.raises(ValueError):
        percentile([1, 2], -1)
    with pytest.raises(ValueError):
        percentile([], 50)
    with pytest.raises(ValueError):
        percentile([np.nan], 50)
    with pytest.raises(ValueError):
        percentile([1, 2], 50, "nearest")
    with pytest.raises(TypeError):
        percentile(["a"], 50)


def test_titanic(titanic_path):
    X = io.load_csv(titanic_path, usecols=["Pclass", "Age"])
    pclass = X[:6, 0]  # [3, 1, 3, 1, 3, 3]
    assert rank(pclass, "dense").tolist() == [2, 1, 2, 1, 2, 2]
    assert rank(pclass, "average").tolist() == [4.5, 1.5, 4.5, 1.5, 4.5, 4.5]

    age_ranks = rank(X[:8, 1])
    assert np.isnan(age_ranks[5])  # passenger 6 has no age

    ages = X[:, 1]
    for method in ("linear", "lower", "higher", "midpoint"):
        expected = np.nanpercentile(ages, [25, 50, 75], method=method)
        assert np.allclose(percentile(ages, [25, 50, 75], method), expected)
