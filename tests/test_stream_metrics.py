"""Tests for the streaming classes in numcompute_stream.metrics.

Run with:  python -m pytest tests/test_stream_metrics.py
"""

import numpy as np
import pytest

from numcompute_stream.metrics import (
    StreamingAccuracy,
    StreamingAUC,
    StreamingConfusionMatrix,
    StreamingF1,
    StreamingMSE,
    StreamingPrecision,
    StreamingRecall,
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_curve,
)


def _feed(metric, a, b, n_chunks=6):
    """Update the metric with a and b split into n_chunks pieces."""
    for a_chunk, b_chunk in zip(np.array_split(a, n_chunks), np.array_split(b, n_chunks)):
        metric.update(a_chunk, b_chunk)
    return metric


def test_accumulated_metrics_equal_batch_metrics():
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 3, size=400)
    y_pred = np.where(rng.random(400) < 0.7, y_true, rng.integers(0, 3, size=400))
    assert np.array_equal(
        _feed(StreamingConfusionMatrix(), y_true, y_pred).result(),
        confusion_matrix(y_true, y_pred),
    )
    assert _feed(StreamingAccuracy(), y_true, y_pred).result() == pytest.approx(
        accuracy(y_true, y_pred)
    )
    for stream_cls, batch in (
        (StreamingPrecision, precision),
        (StreamingRecall, recall),
        (StreamingF1, f1),
    ):
        streamed = _feed(stream_cls(pos_label=2), y_true, y_pred).result()
        assert streamed == pytest.approx(batch(y_true, y_pred, pos_label=2))
    y = rng.normal(size=100)
    assert _feed(StreamingMSE(), y, y + 0.5).result() == pytest.approx(mse(y, y + 0.5))


def test_labels_grow_and_fixed_labels_reject_unknowns():
    matrix = StreamingConfusionMatrix()
    matrix.update([0, 1], [0, 0])
    matrix.update([2], [1])  # class 2 first seen in the second chunk
    assert matrix.labels_.tolist() == [0, 1, 2]
    assert matrix.result().tolist() == [[1, 0, 0], [1, 0, 0], [0, 1, 0]]

    fixed = StreamingConfusionMatrix(labels=["no", "yes"])
    fixed.update(["yes"], ["no"])
    with pytest.raises(ValueError):
        fixed.update(["maybe"], ["no"])
    with pytest.raises(ValueError):
        fixed.update(["yes", "no"], ["no"])  # different lengths


def test_rolling_window_forgets_old_chunks():
    metric = StreamingAccuracy(window=2)
    metric.update([1, 1], [0, 0])  # 0 % correct
    metric.update([1, 1], [1, 1])  # 100 %
    assert metric.result() == 0.5
    metric.update([0, 0], [0, 0])  # the first chunk drops out
    assert metric.result() == 1.0
    windowed = StreamingConfusionMatrix(window=1)
    windowed.update([0], [0])
    windowed.update([5], [5])  # new label while the old chunk is still stored
    assert windowed.result().tolist() == [[0, 0], [0, 1]]


def test_zero_denominators_and_reset():
    metric = StreamingPrecision(pos_label=1)
    with pytest.raises(RuntimeError):
        metric.result()  # nothing added yet
    metric.update([1, 0], [0, 0])  # nothing predicted positive
    assert metric.result() == 0.0
    assert StreamingRecall(pos_label=7).update([0], [0]).result() == 0.0  # class never seen
    metric.reset()
    with pytest.raises(RuntimeError):
        metric.result()
    accuracy_metric = StreamingAccuracy().update([], [])  # an empty chunk is ignored
    with pytest.raises(RuntimeError):
        accuracy_metric.result()
    with pytest.raises(ValueError):
        StreamingAccuracy(window=0)


def test_streaming_auc_is_close_to_exact_auc():
    rng = np.random.default_rng(1)
    y_true = rng.integers(0, 2, size=2000)
    scores = np.clip(rng.normal(0.4 + 0.2 * y_true, 0.2), 0, 1)
    streamed = _feed(StreamingAUC(), y_true, scores, n_chunks=10)
    exact = auc(*roc_curve(y_true, scores)[:2])
    assert streamed.result() == pytest.approx(exact, abs=0.01)
    fpr, tpr = streamed.roc_curve()
    assert fpr[0] == 0 and tpr[-1] == 1 and np.all(np.diff(fpr) >= 0)


def test_streaming_auc_edge_cases():
    perfect = StreamingAUC().update([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert perfect.result() == 1.0
    tied = StreamingAUC().update([0, 1], [0.5, 0.5])
    assert tied.result() == 0.5  # a tie counts as half, like roc_curve
    one_class = StreamingAUC().update([1, 1], [0.3, 0.9])
    assert np.isnan(one_class.result())  # undefined until both classes appear
    one_class.update([0], [0.1])
    assert one_class.result() == 1.0
    with pytest.raises(ValueError):
        StreamingAUC().update([0, 1], [0.5, np.nan])
    with pytest.raises(TypeError):
        StreamingAUC().update([0, 1], ["a", "b"])


def test_mse_and_auc_reset_window_and_parameters():
    mse_metric = StreamingMSE(window=1)
    with pytest.raises(RuntimeError):
        mse_metric.result()
    mse_metric.update([0.0, 0.0], [1.0, 1.0]).update([0.0], [3.0])
    assert mse_metric.result() == 9.0  # only the last chunk counts
    with pytest.raises(TypeError):
        mse_metric.update(["a"], ["b"])
    mse_metric.reset()
    with pytest.raises(RuntimeError):
        mse_metric.result()

    auc_metric = StreamingAUC(window=1)
    auc_metric.update([0, 1], [0.9, 0.1])  # perfectly wrong
    auc_metric.update([0, 1], [0.1, 0.9])  # perfectly right; the first chunk drops out
    assert auc_metric.result() == 1.0
    assert StreamingAUC(range=(0, 10)).update([0, 1], [2.0, 8.0]).result() == 1.0
    assert StreamingAUC().update([0, 1], [-5.0, 7.0]).result() == 1.0  # clipped into end bins
    auc_metric.reset()
    with pytest.raises(RuntimeError):
        auc_metric.result()
    with pytest.raises(RuntimeError):
        auc_metric.roc_curve()
    for bad in ({"bins": 0}, {"range": (1, 1)}, {"window": 0}):
        with pytest.raises(ValueError):
            StreamingAUC(**bad)
