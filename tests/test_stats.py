"""Tests for numcompute.stats. Run with:  python -m pytest tests/test_stats.py"""

import numpy as np
import pytest

from numcompute import io
from numcompute.stats import RunningStats, amax, amin, histogram, mean, median, quantile, std, var


def test_basic_stats_match_numpy():
    X = np.random.default_rng(0).normal(size=(30, 4))
    for axis in (None, 0, 1):
        assert np.allclose(mean(X, axis), np.mean(X, axis=axis))
        assert np.allclose(var(X, axis, ddof=1), np.var(X, axis=axis, ddof=1))
        assert np.allclose(std(X, axis), np.std(X, axis=axis))
        assert np.allclose(median(X, axis), np.median(X, axis=axis))
        assert np.allclose(amin(X, axis), np.min(X, axis=axis))
        assert np.allclose(amax(X, axis), np.max(X, axis=axis))
    assert isinstance(mean(X), float)  # axis=None -> plain float
    assert mean(X, axis=0).shape == (4,) and mean(X, axis=1).shape == (30,)


def test_nan_handling():
    X = np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, np.nan]])
    assert np.isnan(mean(X[:, 0]))  # NaN propagates by default
    assert mean(X, axis=0, ignore_nan=True).tolist() == [2.0, 4.0]
    assert np.allclose(std(X, axis=0, ignore_nan=True), [1.0, 0.0])
    assert amin(X, axis=1, ignore_nan=True)[:2].tolist() == [1.0, 3.0]
    assert np.isnan(amax(X, axis=1, ignore_nan=True)[2])  # all-NaN row -> NaN
    assert np.isnan(mean(X, axis=1, ignore_nan=True)[2])


def test_quantile():
    x = np.random.default_rng(1).normal(size=(11, 3))
    q = [0, 0.1, 0.5, 0.9, 1]
    assert np.allclose(quantile(x, q, axis=0), np.quantile(x, q, axis=0))
    assert np.allclose(quantile(x, 0.3), np.quantile(x, 0.3))
    assert quantile([1, 2, 3, 4], 0.5) == 2.5
    y = np.array([1.0, np.nan, 3.0])
    assert np.isnan(quantile(y, 0.5))
    assert quantile(y, 0.5, ignore_nan=True) == 2.0
    with pytest.raises(ValueError):
        quantile([1, 2], 1.5)


def test_edge_cases():
    assert var([5.0], ddof=0) == 0.0
    assert np.isnan(var([5.0], ddof=1))  # n - ddof = 0
    assert std([7, 7, 7]) == 0.0  # all equal
    view = np.arange(20.0).reshape(4, 5)[:, ::2]  # non-contiguous
    assert np.allclose(mean(view, 0), np.mean(view, axis=0))
    big = 1e9 + np.array([1.0, 2.0, 3.0])  # large mean, small spread
    assert np.isclose(var(big), 2 / 3)
    with pytest.raises(ValueError):
        mean([])
    with pytest.raises(ValueError):
        mean(np.ones((2, 2)), axis=2)
    with pytest.raises(ValueError):
        var([1, 2], ddof=-1)
    with pytest.raises(TypeError):
        mean(["a", "b"])


def test_histogram():
    x = np.random.default_rng(2).normal(size=500)
    counts, edges = histogram(x, bins=12)
    np_counts, np_edges = np.histogram(x, bins=12)
    assert np.array_equal(counts, np_counts) and np.allclose(edges, np_edges)
    counts, _ = histogram([0, 1, 2, 3, 4], bins=2, range=(0, 4))
    assert counts.tolist() == [2, 3]  # right edge is in the last bin
    counts, _ = histogram([5, 5, 5], bins=3)  # all equal
    assert counts.sum() == 3
    assert histogram([1.0, np.nan, 2.0], bins=1)[0].tolist() == [2]
    with pytest.raises(ValueError):
        histogram([1, 2], bins=0)
    with pytest.raises(ValueError):
        histogram([1, np.inf])


def test_running_stats_match_batch():
    X = np.random.default_rng(3).normal(5, 2, size=(103, 3))
    X[[4, 50, 77], [0, 1, 2]] = np.nan
    rs = RunningStats()
    for start in range(0, len(X), 10):
        rs.update(X[start : start + 10])
    assert np.allclose(rs.mean, np.nanmean(X, axis=0))
    assert np.allclose(rs.var(), np.nanvar(X, axis=0))
    assert np.allclose(rs.std(ddof=1), np.nanstd(X, axis=0, ddof=1))
    assert rs.count.tolist() == [102, 102, 102]


def test_running_stats_edge_cases():
    rs = RunningStats().update([1.0, 2.0]).update([np.nan]).update([3.0])
    assert rs.mean.tolist() == [2.0]  # 1-D input = one column
    running = RunningStats().update(np.ones((2, 2)))
    with pytest.raises(ValueError):  # different number of columns
        running.update(np.ones((2, 3)))
    with pytest.raises(ValueError):  # no data yet
        _ = RunningStats().mean


def test_titanic(titanic_path):
    X = io.load_csv(titanic_path, usecols=["Age", "Fare"])
    assert np.isnan(mean(X, axis=0)[0])  # NaN propagates by default
    assert np.allclose(mean(X, axis=0, ignore_nan=True), np.nanmean(X, axis=0))

    ages = X[:, 0]
    quartiles = quantile(ages, [0.25, 0.5, 0.75], ignore_nan=True)
    assert np.allclose(quartiles, np.nanquantile(ages, [0.25, 0.5, 0.75]))

    counts, _ = histogram(ages, bins=8)
    assert counts.sum() == np.count_nonzero(~np.isnan(ages))

    running = RunningStats()
    for chunk in io.iter_csv_chunks(titanic_path, chunksize=100, usecols=["Age", "Fare"]):
        running.update(chunk)
    assert np.allclose(running.mean, mean(X, axis=0, ignore_nan=True))
    assert np.allclose(running.std(), np.nanstd(X, axis=0))
