"""Tests for numcompute.sort_search. Run with:  python -m pytest tests/test_sort_search.py"""

import numpy as np
import pytest

from numcompute import io
from numcompute.sort_search import binary_search, quickselect, stable_sort, topk


def test_stable_sort():
    assert stable_sort([3, 1, 3, 2]).tolist() == [1, 2, 3, 3]
    X = np.array([[3.0, np.nan, 1.0], [2.0, 5.0, 4.0]])
    assert stable_sort(X, axis=0)[0].tolist() == [2.0, 5.0, 1.0]
    assert np.isnan(stable_sort(X[0])[-1])  # NaN goes last
    view = np.arange(20)[::-1][::3]  # non-contiguous
    assert np.array_equal(stable_sort(view), np.sort(view))
    with pytest.raises(ValueError):
        stable_sort(5)


def test_topk():
    vals, idx = topk([3, 1, 4, 1, 5], 2)
    assert vals.tolist() == [5, 4] and idx.tolist() == [4, 2]
    vals, idx = topk([5, 1, 5, 5, 2], 2)  # ties: lower index wins
    assert idx.tolist() == [0, 2]
    vals, idx = topk([3.0, np.nan, 1.0, 2.0], 2, largest=False)
    assert vals.tolist() == [1.0, 2.0] and idx.tolist() == [2, 3]


def test_topk_extreme_k():
    a = np.array([2.0, np.nan, 7.0])
    assert topk(a, 0, return_indices=False).size == 0
    assert topk(a, 2, return_indices=False).tolist() == [7.0, 2.0]  # k = all non-NaN
    with pytest.raises(ValueError):  # NaN is not counted
        topk(a, 3)
    with pytest.raises(ValueError):
        topk(a, -1)
    with pytest.raises(TypeError):
        topk(a, 1.5)
    with pytest.raises(ValueError):
        topk([], 1)


def test_quickselect():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 10, size=101)  # many duplicates
    for k in [0, 50, 100]:
        assert quickselect(a, k) == np.sort(a)[k]
    assert quickselect(np.full(1000, 7.0), 500) == 7.0  # all equal
    with pytest.raises(ValueError):
        quickselect([], 0)
    with pytest.raises(ValueError):
        quickselect([1.0, np.nan], 0)
    with pytest.raises(ValueError):
        quickselect([1, 2], 2)


def test_binary_search():
    a = np.array([1, 3, 3, 7])
    assert binary_search(a, 3) == (1, True)  # first of duplicates
    assert binary_search(a, 4) == (3, False)
    assert binary_search(a, 0) == (0, False) and binary_search(a, 9) == (4, False)
    q = np.array([0, 1, 3, 5, 7, 8])
    idx, found = binary_search(a, q)
    assert np.array_equal(idx, np.searchsorted(a, q))
    assert found.tolist() == [False, True, True, False, True, False]


def test_binary_search_edge_cases():
    assert binary_search([], 5) == (0, False)
    a = np.array([1.0, 2.0, np.nan])
    assert binary_search(a, np.nan) == (2, False)
    assert binary_search(a, 5.0) == (2, False)
    with pytest.raises(ValueError):
        binary_search([[1, 2]], 1)
    with pytest.raises(TypeError):
        binary_search(["a", "b"], "a")


def test_titanic(titanic_path):
    fare = io.load_csv(titanic_path, usecols="Fare")[:, 0]

    top_values, top_rows = topk(fare, 5)
    assert np.array_equal(top_values, np.sort(fare)[::-1][:5])
    assert np.array_equal(fare[top_rows], top_values)

    k = len(fare) // 2
    assert quickselect(fare, k) == np.sort(fare)[k]

    sorted_fare = np.sort(fare)
    queries = np.array([7.25, 100.0, 1000.0])
    positions, found = binary_search(sorted_fare, queries)
    assert np.array_equal(positions, np.searchsorted(sorted_fare, queries))
    assert found.tolist() == [True, False, False]
