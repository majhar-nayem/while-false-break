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
    stats = StreamingStats().update_stats(np.empty((0, 2)))  # an empty first chunk is ignored
    with pytest.raises(RuntimeError):
        _ = stats.mean
    stats.update_stats(np.array([[np.nan, 2.0], [np.nan, 2.0]]))  # column 0 all NaN
    assert np.isnan(stats.mean[0]) and np.isnan(stats.min[0]) and np.isnan(stats.quantile(0.5)[0])
    assert stats.var()[1] == 0.0  # zero-variance chunk: no NaN, no warning
    stats.update_stats(np.array([[4.0, 2.0], [6.0, 2.0]]))
    stats.update_stats(np.empty((0, 2)))  # and so is a later one
    assert stats.mean.tolist() == [5.0, 2.0]
    assert stats.var().tolist() == [1.0, 0.0]
    assert stats.n_chunks == 2 and stats.count.tolist() == [2, 4]


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


def test_range_widens_to_cover_later_values():
    hist = StreamingHistogram(bins=4)  # range taken from the first chunk: [0, 4]
    hist.update([])  # an empty chunk does not count as the first chunk
    hist.update([0.0, 1.0, 4.0])
    hist.update([-10.0, 9.0, 2.0, np.nan, np.inf])
    edges = hist.edges[0]
    assert edges[0] <= -10 and edges[-1] >= 9  # doubled until both new values fit
    assert np.allclose(np.diff(edges), edges[1] - edges[0])  # bins are still equal width
    assert hist.counts.sum() == 6 and hist.n_above.tolist() == [1]  # only inf is outside
    assert hist.count.tolist() == [7]  # NaN is ignored

    # Widening downwards leaves empty bins below the smallest value; the 0 and
    # 1 quantiles must still be the data's ends, not the edges of the range.
    hist = StreamingHistogram(bins=64).update(np.linspace(0.0, 1.0, 50)).update([-0.1])
    edges = hist.edges[0]
    width = edges[1] - edges[0]
    assert edges[0] < -0.9  # the range doubled to about [-1, 1]
    assert abs(hist.quantile(0.0)[0] + 0.1) <= width
    assert abs(hist.quantile(1.0)[0] - 1.0) <= width


def test_widening_matches_batch_histogram_and_quantiles():
    rng = np.random.default_rng(4)
    chunks = [rng.uniform(0, 1, 100)] + [rng.normal(50, 30, 300) for _ in range(5)]
    hist = StreamingHistogram(bins=64)  # the first chunk covers only [0, 1]
    for chunk in chunks:
        hist.update(chunk)
    everything = np.concatenate(chunks)
    expected, _ = np.histogram(everything, bins=hist.edges[0])
    # The first chunk's max becomes a bin edge after widening; recomputing the
    # edges can round it to either side, so one value may sit in a neighbour bin.
    difference = hist.counts[0] - expected
    assert difference.sum() == 0 and np.abs(difference).sum() <= 2
    width = hist.edges[0][1] - hist.edges[0][0]
    q = [0.1, 0.5, 0.9]
    assert np.all(np.abs(hist.quantile(q)[:, 0] - np.quantile(everything, q)) <= width)
    windowed = StreamingHistogram(bins=64, window=2)  # stored chunks are re-binned too
    for chunk in chunks:
        windowed.update(chunk)
    recent, _ = np.histogram(np.concatenate(chunks[-2:]), bins=windowed.edges[0])
    assert windowed.counts[0].tolist() == recent.tolist()  # the narrow chunk dropped out


def test_fixed_range_counts_values_outside():
    hist = StreamingHistogram(bins=4, range=(0, 4))
    hist.update([-1.0, 0.0, 4.0, 5.0, 2.0])
    assert hist.n_below.tolist() == [1] and hist.n_above.tolist() == [1]
    assert hist.quantile(0.0)[0] == 0.0 and hist.quantile(1.0)[0] == 4.0  # clamped to the range
    with pytest.raises(ValueError):
        hist.update(np.ones((2, 3)))  # different number of columns
    with pytest.raises(ValueError):
        StreamingHistogram(bins=5)  # an adapting range needs an even number of bins
    with pytest.raises(ValueError):
        StreamingHistogram(range=(2, 2))


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

    # The exact bound: an estimate shares a bin with the ceil(q * n)-th smallest
    # value. It holds even for a few scattered values, where np.quantile
    # interpolates across gaps much wider than a bin.
    few = np.array([5.02, 0.03, 9.96, 0.14, 9.87])
    hist = StreamingHistogram(bins=100, range=(0, 10)).update(few)
    q = np.array([0.0, 0.3, 0.5, 0.7, 1.0])
    rank = np.maximum(np.ceil(q * few.size).astype(int), 1)
    assert np.all(np.abs(hist.quantile(q)[:, 0] - np.sort(few)[rank - 1]) <= 10 / 100)
    assert abs(hist.quantile(0.3)[0] - np.quantile(few, 0.3)) > 10 / 100  # numpy interpolates


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


def test_axis_none_pools_every_value():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(300, 3))
    X[0, 1] = np.nan
    pooled = StreamingStats(bins=200, axis=None)
    for chunk in np.array_split(X, 4):
        pooled.update_stats(chunk)
    assert pooled.count == 899 and isinstance(pooled.count, int)
    assert isinstance(pooled.mean, float)
    assert pooled.mean == pytest.approx(np.nanmean(X))
    assert pooled.std() == pytest.approx(np.nanstd(X))
    assert pooled.min == np.nanmin(X) and pooled.max == np.nanmax(X)
    width = np.diff(pooled.histogram()[1][:2])[0]
    assert abs(pooled.median() - np.nanmedian(X)) <= width
    assert pooled.quantile([0.25, 0.75]).shape == (2,)
    counts, edges = pooled.histogram()
    assert counts.shape == (200,) and edges.shape == (201,) and counts.sum() == 899
    with pytest.raises(ValueError):
        StreamingStats(axis=1)
