"""Tests for the streaming classes in numcompute_stream.stats.

Run with:  python -m pytest tests/test_stream_stats.py
"""

import numpy as np
import pytest

from numcompute_stream import io
from numcompute_stream.stats import StreamingHistogram, StreamingStats


def _chunks(X, sizes):
    """Split X into consecutive chunks with the given row counts."""
    return np.split(X, np.cumsum(sizes)[:-1])


def test_chunked_moments_match_numpy_on_joined_data():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=1e6, scale=3.0, size=(500, 3))  # large mean: naive formulas lose precision
    X[rng.random(X.shape) < 0.1] = np.nan
    stats = StreamingStats()
    for chunk in _chunks(X, [1, 7, 150, 2, 340]):  # uneven sizes, including one row
        stats.update_stats(chunk)
    assert np.allclose(stats.mean, np.nanmean(X, axis=0))
    assert np.allclose(stats.var(ddof=1), np.nanvar(X, axis=0, ddof=1))
    assert np.allclose(stats.min, np.nanmin(X, axis=0))
    assert np.allclose(stats.max, np.nanmax(X, axis=0))
    assert stats.count.tolist() == (~np.isnan(X)).sum(axis=0).tolist()
    assert stats.n_chunks == 5


def test_all_nan_and_zero_variance_chunks():
    stats = StreamingStats()
    stats.update_stats(np.array([[np.nan, 2.0], [np.nan, 2.0]]))  # column 0 all NaN
    assert np.isnan(stats.mean[0]) and np.isnan(stats.min[0]) and np.isnan(stats.quantile(0.5)[0])
    assert stats.var()[1] == 0.0  # zero-variance chunk: no NaN, no warning
    stats.update_stats(np.array([[4.0, 2.0], [6.0, 2.0]]))
    assert stats.mean.tolist() == [5.0, 2.0]
    assert stats.var().tolist() == [1.0, 0.0]


def test_histogram_counts_add_up_and_match_batch():
    x = np.random.default_rng(1).normal(size=1000)
    hist = StreamingHistogram(bins=20, range=(-4, 4))
    for chunk in np.array_split(x, 9):
        hist.update(chunk)
    expected, edges = np.histogram(x, bins=20, range=(-4, 4))
    assert hist.counts[0].tolist() == expected.tolist()
    assert np.allclose(hist.edges[0], edges)
    assert hist.count[0] == 1000
    assert hist.counts.sum() + hist.n_below.sum() + hist.n_above.sum() == 1000


def test_values_outside_the_first_range_are_still_counted():
    hist = StreamingHistogram(bins=4)  # range taken from the first chunk: [0, 4]
    hist.update([0.0, 1.0, 4.0])
    hist.update([-10.0, 9.0, 2.0, np.nan])
    assert hist.n_below.tolist() == [1] and hist.n_above.tolist() == [1]
    assert hist.count.tolist() == [6]  # NaN is ignored
    assert hist.quantile(0.0)[0] == 0.0 and hist.quantile(1.0)[0] == 4.0  # clamped to the range


def test_quantiles_within_one_bin_width():
    x = np.random.default_rng(2).uniform(0, 10, size=(5000, 2))
    stats = StreamingStats(bins=100, range=(0, 10))
    for chunk in np.array_split(x, 13):
        stats.update_stats(chunk)
    q = [0.05, 0.25, 0.5, 0.75, 0.95]
    estimate = stats.quantile(q)
    assert estimate.shape == (5, 2)
    assert np.all(np.abs(estimate - np.quantile(x, q, axis=0)) <= 10 / 100)
    assert np.all(np.abs(stats.median() - np.median(x, axis=0)) <= 10 / 100)


def test_sliding_window_forgets_old_chunks():
    hist = StreamingHistogram(bins=2, range=(0, 2), window=2)
    hist.update([0.5, 0.5, 0.5])  # bin 0
    hist.update([1.5])  # bin 1
    hist.update([1.5, 1.5])  # bin 1: the first chunk drops out
    assert hist.counts[0].tolist() == [0, 3]
    assert hist.quantile(0.5)[0] > 1.0


def test_errors_and_reset():
    stats = StreamingStats()
    with pytest.raises(RuntimeError):
        _ = stats.mean
    with pytest.raises(TypeError):
        stats.update_stats(np.array(["a", "b"]))
    stats.update_stats(np.ones((3, 2)))
    with pytest.raises(ValueError):
        stats.update_stats(np.ones((3, 5)))
    with pytest.raises(ValueError):
        stats.quantile(1.5)
    with pytest.raises(ValueError):
        StreamingHistogram(bins=0)
    with pytest.raises(ValueError):
        StreamingHistogram(window=0)
    stats.reset()
    with pytest.raises(RuntimeError):
        stats.var()


def test_titanic_chunks_match_whole_file(titanic_path):
    # Columns 5 and 9 are Age (177 missing) and Fare.
    whole = io.load_csv(titanic_path, usecols=[5, 9])
    stats = StreamingStats()
    for chunk in io.iter_csv_chunks(titanic_path, chunksize=100, usecols=[5, 9]):
        stats.update_stats(chunk)
    assert stats.count.tolist() == [714, 891]
    assert np.allclose(stats.mean, np.nanmean(whole, axis=0))
    assert np.allclose(stats.std(), np.nanstd(whole, axis=0))
