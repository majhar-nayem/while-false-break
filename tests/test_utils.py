"""Tests for numcompute.utils. Run with:  python -m pytest tests/test_utils.py"""

import numpy as np
import pytest

from numcompute.utils import (
    argtopk,
    batch_iterator,
    euclidean_distances,
    logsumexp,
    manhattan_distances,
    relu,
    sigmoid,
    softmax,
)


def test_distances_match_loops():
    rng = np.random.default_rng(0)
    A, B = rng.normal(size=(5, 3)), rng.normal(size=(4, 3))
    loop = np.array([[np.sqrt(((a - b) ** 2).sum()) for b in B] for a in A])
    assert np.allclose(euclidean_distances(A, B), loop)
    loop = np.array([[np.abs(a - b).sum() for b in B] for a in A])
    assert np.allclose(manhattan_distances(A, B), loop)
    D = euclidean_distances(A)
    assert D.shape == (5, 5) and np.all(D >= 0) and np.allclose(D, D.T)
    P = np.random.default_rng(42).normal(size=(200, 8))
    assert np.all(np.diag(euclidean_distances(P)) == 0)  # exactly 0, not ~1e-8


def test_distances_errors_and_strides():
    with pytest.raises(ValueError):
        euclidean_distances(np.ones((2, 3)), np.ones((2, 4)))
    with pytest.raises(ValueError):
        euclidean_distances(np.ones(3))
    with pytest.raises(TypeError):
        manhattan_distances([["a"]])
    view = np.arange(24.0).reshape(4, 6)[:, ::2]  # non-contiguous
    assert np.allclose(euclidean_distances(view), euclidean_distances(view.copy()))


def test_sigmoid_and_relu():
    s = sigmoid([-1000.0, 0.0, 1000.0])  # no overflow warning
    assert s.tolist() == [0.0, 0.5, 1.0]
    assert np.allclose(sigmoid([2.0]), 1 / (1 + np.exp(-2.0)))
    assert relu([-2, 0, 3]).tolist() == [0, 0, 3]


def test_softmax_stable():
    p = softmax([1000.0, 1000.0, 1000.0])  # naive exp overflows
    assert np.allclose(p, 1 / 3)
    P = softmax(np.random.default_rng(1).normal(size=(4, 5)), axis=1)
    assert np.allclose(P.sum(axis=1), 1)
    assert np.allclose(softmax([1.0, 2.0]), softmax([101.0, 102.0]))  # shift invariant
    with pytest.raises(ValueError):
        softmax(3.0)


def test_softmax_infinite_and_empty():
    with np.errstate(all="raise"):  # fails if any overflow / invalid warning is hit
        assert softmax([np.inf, 1.0]).tolist() == [1.0, 0.0]
        assert softmax([np.inf, np.inf, 1.0]).tolist() == [0.5, 0.5, 0.0]
        assert softmax([-np.inf, 0.0]).tolist() == [0.0, 1.0]
        assert softmax([1e308, -1e308]).tolist() == [1.0, 0.0]
        assert np.all(np.isnan(softmax([-np.inf, -np.inf])))  # undefined -> NaN
    assert np.all(np.isnan(softmax([np.nan, 1.0])))  # NaN propagates
    X = np.array([[np.inf, 1.0], [1.0, 1.0]])
    assert softmax(X, axis=1).tolist() == [[1.0, 0.0], [0.5, 0.5]]  # per-slice handling
    assert softmax(X, axis=0).tolist() == [[1.0, 0.5], [0.0, 0.5]]
    with pytest.raises(ValueError, match="empty"):
        softmax([])


def test_logsumexp():
    x = np.array([1.0, 2.0, 3.0])
    assert np.isclose(logsumexp(x), np.log(np.exp(x).sum()))
    assert np.isclose(logsumexp([1000.0, 1000.0]), 1000 + np.log(2))  # no overflow
    assert logsumexp([-np.inf, -np.inf]) == -np.inf
    assert logsumexp(np.ones((2, 3)), axis=1).shape == (2,)
    with pytest.raises(ValueError):
        logsumexp([])


def test_argtopk():
    X = np.array([[1, 9, 3, 7], [8, 2, 6, 4]])
    assert argtopk(X, 2).tolist() == [[1, 3], [0, 2]]
    assert argtopk(X, 1, largest=False).tolist() == [[0], [1]]
    assert argtopk(X, 1, axis=0).tolist() == [[1, 0, 1, 0]]
    with pytest.raises(ValueError):
        argtopk(X, 5)
    with pytest.raises(ValueError):
        argtopk(X, 0)


def test_batch_iterator():
    X = np.arange(10)
    batches = list(batch_iterator(X, 4))
    assert [b.tolist() for b in batches] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    shuffled = np.concatenate(list(batch_iterator(X, 3, shuffle=True, seed=0)))
    assert sorted(shuffled.tolist()) == X.tolist()
    assert list(batch_iterator(np.empty((0, 2)), 3)) == []
    with pytest.raises(ValueError):
        list(batch_iterator(X, 0))
