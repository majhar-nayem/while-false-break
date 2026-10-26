"""Tests for numcompute.metrics. Run with:  python -m pytest tests/test_metrics.py"""

import numpy as np
import pytest

from numcompute.metrics import (
    accuracy,
    auc,
    confusion_matrix,
    f1,
    mse,
    precision,
    recall,
    roc_curve,
)

Y_TRUE = [1, 0, 1, 1, 0, 0, 1, 0]
Y_PRED = [1, 0, 0, 1, 1, 0, 1, 0]  # TP=3, FP=1, FN=1, TN=3


def test_classification_metrics():
    assert accuracy(Y_TRUE, Y_PRED) == 6 / 8
    assert precision(Y_TRUE, Y_PRED) == 3 / 4
    assert recall(Y_TRUE, Y_PRED) == 3 / 4
    assert f1(Y_TRUE, Y_PRED) == 0.75
    assert precision(Y_TRUE, Y_PRED, pos_label=0) == 3 / 4


def test_zero_division_gives_zero():
    assert precision([1, 1], [0, 0]) == 0.0  # nothing predicted positive
    assert recall([0, 0], [1, 0]) == 0.0  # no real positives
    assert f1([0, 0], [0, 0]) == 0.0


def test_confusion_matrix():
    C = confusion_matrix(Y_TRUE, Y_PRED)
    assert C.tolist() == [[3, 1], [1, 3]]  # rows = true, cols = predicted
    C = confusion_matrix(["cat", "dog", "cat"], ["dog", "dog", "cat"])
    assert C.tolist() == [[1, 1], [0, 1]]
    C = confusion_matrix([0, 1, 2], [0, 1, 2], labels=[2, 1, 0])
    assert C.tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert confusion_matrix([0, 0], [0, 0], labels=[0, 1]).tolist() == [[2, 0], [0, 0]]


def test_mse():
    assert mse([1, 2, 3], [1, 2, 3]) == 0.0
    assert mse([0, 0], [1, 3]) == 5.0
    with pytest.raises(TypeError):
        mse(["a"], ["b"])


def test_roc_and_auc():
    y = [0, 0, 1, 1]
    s = [0.1, 0.4, 0.35, 0.8]
    fpr, tpr, thr = roc_curve(y, s)
    assert fpr.tolist() == [0, 0, 0.5, 0.5, 1] and tpr.tolist() == [0, 0.5, 0.5, 1, 1]
    assert thr[0] == np.inf
    assert auc(fpr, tpr) == 0.75
    assert auc(*roc_curve([0, 1], [0.0, 1.0])[:2]) == 1.0  # perfect scores
    fpr, tpr, _ = roc_curve([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5])  # all tied
    assert fpr.tolist() == [0, 1] and auc(fpr, tpr) == 0.5


def test_errors():
    with pytest.raises(ValueError):
        accuracy([1, 0], [1])
    with pytest.raises(ValueError):
        accuracy([], [])
    with pytest.raises(ValueError):
        accuracy([[1]], [[1]])
    with pytest.raises(ValueError):  # only one class
        roc_curve([1, 1], [0.2, 0.3])
    with pytest.raises(ValueError):
        roc_curve([0, 1], [0.2, np.nan])
    with pytest.raises(ValueError):  # x not increasing
        auc([0, 1, 0.5], [0, 1, 1])
    with pytest.raises(ValueError):
        auc([0], [0])
