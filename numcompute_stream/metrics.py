"""Evaluation metrics: classification, regression (MSE), and ROC / AUC.

Batch functions (accuracy, precision, confusion_matrix, roc_curve, ...) score
one pair of arrays. Streaming classes accumulate over chunks instead:

    metric = StreamingAccuracy(window=None)
    for y_true_chunk, y_pred_chunk in stream:
        metric.update(y_true_chunk, y_pred_chunk)
        metric.result()   # over everything so far (or the last `window` chunks)
    metric.reset()

Every streaming class has update(...) -> self, result() and reset() -> self.
With window=w only the last w chunks count (a rolling-window metric).
"""

from collections import deque

import numpy as np

NUMERIC_KINDS = "biuf"  # NumPy dtype kinds for bool, int, unsigned int and float


def _check_pair(y_true, y_pred, numeric=False):
    """Convert both inputs to arrays and check they are 1-D, non-empty and the same length."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1:
        raise ValueError(f"y_true and y_pred must be 1-D, got {y_true.shape} and {y_pred.shape}")
    if y_true.size != y_pred.size:
        raise ValueError(f"y_true has {y_true.size} values but y_pred has {y_pred.size}")
    if y_true.size == 0:
        raise ValueError("y_true and y_pred are empty")
    both_numeric = y_true.dtype.kind in NUMERIC_KINDS and y_pred.dtype.kind in NUMERIC_KINDS
    if numeric and not both_numeric:
        raise TypeError("y_true and y_pred must be numeric")
    return y_true, y_pred


def _label_codes(y, labels, order, sorted_labels):
    """
    Find the position of each value of y in labels.

    Returns (codes, known): codes[i] is the index of y[i] in labels, and
    known[i] is False when y[i] is not in labels at all.
    """
    k = labels.size
    position = np.searchsorted(sorted_labels, y)
    position = np.clip(position, 0, k - 1)
    known = sorted_labels[position] == y
    return order[position], known


def confusion_matrix(y_true, y_pred, labels=None):
    """
    Count how often each true class was predicted as each class.

    Parameters:
        y_true : array_like, shape (n,), true labels
        y_pred : array_like, shape (n,), predicted labels
        labels : list, optional, class order (default: sorted unique labels)

    Returns:
        ndarray of int, shape (k, k); entry [i, j] = number of samples with
        true class labels[i] predicted as labels[j] (rows = true, columns = predicted)

    Raises:
        ValueError if the inputs are not 1-D, have different lengths, or are empty.

    Notes:
        Each label is turned into its position in labels (np.searchsorted), so
        the pair (true, pred) becomes one number true * k + pred and np.bincount
        counts all pairs at once. Samples whose label is not in labels are skipped.

    Complexity:
        O(n log k + k^2) time, O(n + k^2) space.
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred]))
    labels = np.asarray(labels)
    order = np.argsort(labels)
    sorted_labels = labels[order]
    k = labels.size

    true_codes, true_known = _label_codes(y_true, labels, order, sorted_labels)
    pred_codes, pred_known = _label_codes(y_pred, labels, order, sorted_labels)
    keep = true_known & pred_known

    pair_codes = true_codes[keep] * k + pred_codes[keep]
    counts = np.bincount(pair_codes, minlength=k * k)
    return counts.reshape(k, k)


def accuracy(y_true, y_pred):
    """
    Fraction of predictions that are correct.

    Parameters:
        y_true, y_pred : array_like, shape (n,)

    Returns:
        float in [0, 1]

    Raises:
        ValueError if the inputs are not 1-D, have different lengths, or are empty.

    Complexity:
        O(n) time and space.
    """
    y_true, y_pred = _check_pair(y_true, y_pred)
    return float(np.mean(y_true == y_pred))


def _counts(y_true, y_pred, pos_label):
    """Return the number of true positives, false positives and false negatives."""
    y_true, y_pred = _check_pair(y_true, y_pred)
    actual_pos = y_true == pos_label
    predicted_pos = y_pred == pos_label
    tp = np.sum(predicted_pos & actual_pos)
    fp = np.sum(predicted_pos & ~actual_pos)
    fn = np.sum(~predicted_pos & actual_pos)
    return tp, fp, fn


def _safe_ratio(numerator, denominator):
    """Return numerator / denominator as a float, or 0.0 if the denominator is 0."""
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def precision(y_true, y_pred, pos_label=1):
    """
    Of the samples predicted positive, the fraction that really are: TP / (TP + FP).

    Parameters:
        y_true, y_pred : array_like, shape (n,)
        pos_label : the label that counts as positive (default 1)

    Returns:
        float in [0, 1]; 0.0 if nothing was predicted positive

    Raises:
        ValueError if the inputs are not 1-D, have different lengths, or are empty.

    Complexity:
        O(n) time and space.
    """
    tp, fp, _ = _counts(y_true, y_pred, pos_label)
    return _safe_ratio(tp, tp + fp)


def recall(y_true, y_pred, pos_label=1):
    """
    Of the real positives, the fraction that were found: TP / (TP + FN).

    Parameters, Returns and Raises are the same as precision
    (0.0 if there are no real positives).

    Complexity:
        O(n) time and space.
    """
    tp, _, fn = _counts(y_true, y_pred, pos_label)
    return _safe_ratio(tp, tp + fn)


def f1(y_true, y_pred, pos_label=1):
    """
    Harmonic mean of precision and recall: 2TP / (2TP + FP + FN).

    Parameters, Returns and Raises are the same as precision
    (0.0 if there are no positives at all).

    Notes:
        Using the count form avoids dividing by precision + recall = 0.

    Complexity:
        O(n) time and space.
    """
    tp, fp, fn = _counts(y_true, y_pred, pos_label)
    return _safe_ratio(2 * tp, 2 * tp + fp + fn)


def mse(y_true, y_pred):
    """
    Mean squared error: mean((y_true - y_pred)^2).

    Parameters:
        y_true, y_pred : array_like, shape (n,), numeric

    Returns:
        float >= 0

    Raises:
        TypeError if the inputs are not numeric.
        ValueError if the inputs are not 1-D, have different lengths, or are empty.

    Complexity:
        O(n) time and space.
    """
    y_true, y_pred = _check_pair(y_true, y_pred, numeric=True)
    errors = y_true.astype(float) - y_pred
    return float(np.mean(errors**2))


def roc_curve(y_true, y_score, pos_label=1):
    """
    ROC curve: false and true positive rates for every score threshold.

    Parameters:
        y_true : array_like, shape (n,), true labels (must contain both classes)
        y_score : array_like, shape (n,), numeric; higher = more likely positive
        pos_label : the label that counts as positive (default 1)

    Returns:
        fpr : ndarray, false positive rate at each threshold, starts at 0
        tpr : ndarray, true positive rate at each threshold, starts at 0
        thresholds : ndarray, the score cut-offs, starts at +inf

    Raises:
        TypeError if y_score is not numeric.
        ValueError if the inputs are not 1-D, have different lengths, are
            empty, y_score has NaN, or y_true has only one class.

    Notes:
        Scores are sorted from high to low and cumulative sums count the true
        and false positives above each threshold. Equal scores form one
        threshold, so ties move the curve diagonally instead of in steps.

    Complexity:
        O(n log n) time for the sort, O(n) space.
    """
    y_true, y_score = _check_pair(y_true, y_score)
    if y_score.dtype.kind not in NUMERIC_KINDS:
        raise TypeError("y_score must be numeric")
    y_score = y_score.astype(float)
    if np.isnan(y_score).any():
        raise ValueError("y_score contains NaN")

    is_pos = y_true == pos_label
    n_pos = is_pos.sum()
    n_neg = (~is_pos).sum()
    if n_pos == 0 or n_neg == 0:
        raise ValueError("y_true must contain both positive and negative samples")

    # Sort from the highest score to the lowest.
    order = np.argsort(-y_score, kind="stable")
    sorted_scores = y_score[order]
    sorted_is_pos = is_pos[order]

    # Index of the last sample in each block of equal scores.
    score_changes = np.flatnonzero(np.diff(sorted_scores))
    last_of_tie = np.append(score_changes, sorted_scores.size - 1)

    tp = np.cumsum(sorted_is_pos)[last_of_tie]
    fp = (last_of_tie + 1) - tp

    fpr = np.concatenate([[0.0], fp / n_neg])
    tpr = np.concatenate([[0.0], tp / n_pos])
    thresholds = np.concatenate([[np.inf], sorted_scores[last_of_tie]])
    return fpr, tpr, thresholds


def auc(x, y):
    """
    Area under a curve with the trapezoidal rule (e.g. auc(fpr, tpr)).

    Parameters:
        x : array_like, shape (n,), increasing
        y : array_like, shape (n,)

    Returns:
        float

    Raises:
        ValueError if x and y have different lengths, fewer than 2 points,
            or x is not increasing.

    Complexity:
        O(n) time and space.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError(f"x and y must be 1-D with the same length, got {x.shape} and {y.shape}")
    if x.size < 2:
        raise ValueError("need at least 2 points to compute an area")

    widths = np.diff(x)
    if np.any(widths < 0):
        raise ValueError("x must be increasing")
    average_heights = (y[1:] + y[:-1]) / 2
    return float(np.sum(widths * average_heights))


class _ChunkWindow:
    """
    Running total of per-chunk arrays, optionally over the last `window` chunks only.

    add() adds a chunk's array and, once there are more than `window` chunks,
    subtracts the oldest one, so the total never has to be recomputed.
    """

    def __init__(self, window):
        if window is not None and (not isinstance(window, (int, np.integer)) or window < 1):
            raise ValueError(f"window must be a positive int or None, got {window!r}")
        self.window = window
        self.reset()

    def reset(self):
        self.total = None
        self.parts = deque()
        self.n_chunks = 0

    def add(self, part):
        self.total = part.copy() if self.total is None else self.total + part
        self.n_chunks += 1
        if self.window is not None:
            self.parts.append(part)
            if len(self.parts) > self.window:
                self.total = self.total - self.parts.popleft()

    def apply(self, func):
        """Replace the total and every stored part with func(part) (used to add new labels)."""
        if self.total is not None:
            self.total = func(self.total)
            self.parts = deque(func(part) for part in self.parts)


class StreamingConfusionMatrix:
    """
    Confusion matrix accumulated over chunks (rows = true, columns = predicted).

    Parameters:
        labels : list, optional, class order. If given, a label outside it
            raises ValueError. If None, the labels grow as new ones appear
            (kept sorted) and the matrix grows with them.
        window : int >= 1 or None, count only the last `window` chunks

    Attributes:
        labels_ : ndarray, the current class order
        n_samples_ : int, samples counted (inside the window)

    Raises:
        ValueError for chunks that are not 1-D, differ in length, or (with
            fixed labels) contain an unknown label.
        RuntimeError from result() before any update.

    Notes:
        Each chunk is counted with the same bincount trick as confusion_matrix:
        the pair (true, pred) becomes the single code true * k + pred.
        An empty chunk is ignored.

    Complexity:
        O(n log k + k^2) time per chunk, O(k^2) memory (times window).
    """

    def __init__(self, labels=None, window=None):
        self.fixed_labels = labels is not None
        self.labels_ = np.asarray(labels) if labels is not None else None
        self._window = _ChunkWindow(window)

    def reset(self):
        """Forget all counts (inferred labels are forgotten too). Returns self."""
        self._window.reset()
        if not self.fixed_labels:
            self.labels_ = None
        return self

    def _add_labels(self, values):
        """Grow labels_ (and every stored matrix) to include the new values."""
        new_labels = np.unique(values) if self.labels_ is None else np.union1d(self.labels_, values)
        if self.labels_ is not None and new_labels.size == self.labels_.size:
            return
        if self.labels_ is not None:
            old_index = np.searchsorted(new_labels, self.labels_)
            k = new_labels.size

            def grow(matrix):
                bigger = np.zeros((k, k), dtype=matrix.dtype)
                bigger[np.ix_(old_index, old_index)] = matrix
                return bigger

            self._window.apply(grow)
        self.labels_ = new_labels

    def update(self, y_true_chunk, y_pred_chunk):
        """
        Add one chunk of true and predicted labels.

        Parameters:
            y_true_chunk, y_pred_chunk : array_like, shape (n,)

        Returns:
            self
        """
        y_true = np.asarray(y_true_chunk)
        y_pred = np.asarray(y_pred_chunk)
        if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.size != y_pred.size:
            raise ValueError(
                f"chunks must be 1-D with the same length, got {y_true.shape} and {y_pred.shape}"
            )
        if y_true.size == 0:
            return self
        if not self.fixed_labels:
            self._add_labels(np.concatenate([y_true, y_pred]))

        labels = self.labels_
        order = np.argsort(labels)
        sorted_labels = labels[order]
        true_codes, true_known = _label_codes(y_true, labels, order, sorted_labels)
        pred_codes, pred_known = _label_codes(y_pred, labels, order, sorted_labels)
        if not (true_known.all() and pred_known.all()):
            unknown = np.unique(np.concatenate([y_true[~true_known], y_pred[~pred_known]]))
            raise ValueError(f"labels {unknown.tolist()} are not in labels {labels.tolist()}")

        k = labels.size
        counts = np.bincount(true_codes * k + pred_codes, minlength=k * k).reshape(k, k)
        if self._window.total is None:
            self._window.total = np.zeros((k, k), dtype=int)
            self._window.n_chunks = 0
        self._window.add(counts)
        return self

    def result(self):
        """Return the accumulated matrix, shape (k, k), in the order of labels_."""
        if self._window.total is None:
            raise RuntimeError("no data has been added yet; call update first")
        return self._window.total.copy()

    @property
    def n_samples_(self):
        return 0 if self._window.total is None else int(self._window.total.sum())


class _StreamingLabelMetric:
    """Shared code for metrics computed from a StreamingConfusionMatrix."""

    def __init__(self, labels=None, window=None):
        self.confusion = StreamingConfusionMatrix(labels=labels, window=window)

    def update(self, y_true_chunk, y_pred_chunk):
        """Add one chunk of true and predicted labels. Returns self."""
        self.confusion.update(y_true_chunk, y_pred_chunk)
        return self

    def reset(self):
        """Forget all chunks. Returns self."""
        self.confusion.reset()
        return self

    def _pos_counts(self, pos_label):
        """True positives, false positives and false negatives for pos_label."""
        matrix = self.confusion.result()
        found = np.flatnonzero(self.confusion.labels_ == pos_label)
        if found.size == 0:  # positive class never seen: no TP, FP or FN
            return 0, 0, 0
        p = found[0]
        tp = matrix[p, p]
        return tp, matrix[:, p].sum() - tp, matrix[p, :].sum() - tp


class StreamingAccuracy(_StreamingLabelMetric):
    """
    Accuracy over every chunk so far (or the last `window` chunks).

    Parameters:
        labels : list, optional (see StreamingConfusionMatrix)
        window : int >= 1 or None

    result() returns a float in [0, 1]; RuntimeError before any update.

    Complexity:
        O(n log k + k^2) per update, O(1) for result.
    """

    def result(self):
        matrix = self.confusion.result()
        return _safe_ratio(np.trace(matrix), matrix.sum())


class StreamingPrecision(_StreamingLabelMetric):
    """
    Precision TP / (TP + FP) for pos_label, accumulated over chunks.

    Parameters:
        pos_label : the label that counts as positive (default 1)
        labels, window : see StreamingConfusionMatrix

    result() returns a float in [0, 1], 0.0 if nothing was predicted positive.
    """

    def __init__(self, pos_label=1, labels=None, window=None):
        super().__init__(labels=labels, window=window)
        self.pos_label = pos_label

    def result(self):
        tp, fp, _ = self._pos_counts(self.pos_label)
        return _safe_ratio(tp, tp + fp)


class StreamingRecall(StreamingPrecision):
    """Recall TP / (TP + FN) for pos_label over chunks; 0.0 if there are no real positives."""

    def result(self):
        tp, _, fn = self._pos_counts(self.pos_label)
        return _safe_ratio(tp, tp + fn)


class StreamingF1(StreamingPrecision):
    """F1 = 2TP / (2TP + FP + FN) for pos_label over chunks; 0.0 if there are no positives."""

    def result(self):
        tp, fp, fn = self._pos_counts(self.pos_label)
        return _safe_ratio(2 * tp, 2 * tp + fp + fn)


class StreamingMSE:
    """
    Mean squared error accumulated over chunks: a running sum of squared errors and a count.

    Parameters:
        window : int >= 1 or None

    Raises:
        TypeError / ValueError as mse; RuntimeError from result() before any update.

    Complexity:
        O(n) per update, O(1) memory (times window).
    """

    def __init__(self, window=None):
        self._window = _ChunkWindow(window)

    def update(self, y_true_chunk, y_pred_chunk):
        """Add one chunk. Returns self."""
        y_true, y_pred = _check_pair(y_true_chunk, y_pred_chunk, numeric=True)
        errors = y_true.astype(float) - y_pred
        self._window.add(np.array([np.sum(errors**2), errors.size]))
        return self

    def reset(self):
        """Forget all chunks. Returns self."""
        self._window.reset()
        return self

    def result(self):
        if self._window.total is None:
            raise RuntimeError("no data has been added yet; call update first")
        squared_sum, count = self._window.total
        return float(squared_sum / count) if count > 0 else 0.0


class StreamingAUC:
    """
    ROC AUC accumulated over chunks, from histograms of the scores.

    An exact AUC needs every score ever seen. Instead the scores of the
    positive and of the negative samples are counted in `bins` equal-width
    bins over `range`. The AUC is the chance that a random positive scores
    higher than a random negative:
        AUC = sum_b pos[b] * (neg below bin b + 0.5 * neg[b]) / (P * N)
    Samples in the same bin count as a tie (one half), exactly as tied
    scores do in roc_curve, so the result is exact whenever different scores
    never share a bin, and otherwise off by at most the share of pairs that do.

    Parameters:
        pos_label : the label that counts as positive (default 1)
        bins : int >= 1 (default 1000)
        range : (low, high), score range (default (0, 1) for probabilities);
            scores outside it are counted in the first or last bin
        window : int >= 1 or None

    Raises:
        TypeError if scores are not numeric.
        ValueError for bad chunks or scores containing NaN.
        RuntimeError from result() before any update.

    Returns (result):
        float in [0, 1], or NaN while only one class has been seen.

    Complexity:
        O(n + bins) per update, O(bins) for result, O(bins) memory (times window).
    """

    def __init__(self, pos_label=1, bins=1000, range=(0.0, 1.0), window=None):
        if not isinstance(bins, (int, np.integer)) or bins < 1:
            raise ValueError(f"bins must be a positive int, got {bins!r}")
        if not float(range[0]) < float(range[1]):
            raise ValueError(f"range must have low < high, got {range}")
        self.pos_label = pos_label
        self.bins = int(bins)
        self.range = (float(range[0]), float(range[1]))
        self._window = _ChunkWindow(window)

    def update(self, y_true_chunk, y_score_chunk):
        """Add one chunk of true labels and scores (higher = more likely positive). Returns self."""
        y_true, y_score = _check_pair(y_true_chunk, y_score_chunk)
        if y_score.dtype.kind not in NUMERIC_KINDS:
            raise TypeError("y_score must be numeric")
        y_score = y_score.astype(float)
        if np.isnan(y_score).any():
            raise ValueError("y_score contains NaN")
        low, high = self.range
        bin_index = np.floor((y_score - low) / (high - low) * self.bins)
        bin_index = np.clip(bin_index, 0, self.bins - 1).astype(int)
        is_pos = (y_true == self.pos_label).astype(int)
        # Row 0 counts negatives, row 1 positives: one bincount for both.
        counts = np.bincount(is_pos * self.bins + bin_index, minlength=2 * self.bins)
        self._window.add(counts.reshape(2, self.bins))
        return self

    def reset(self):
        """Forget all chunks. Returns self."""
        self._window.reset()
        return self

    def result(self):
        if self._window.total is None:
            raise RuntimeError("no data has been added yet; call update first")
        neg, pos = self._window.total
        n_neg, n_pos = neg.sum(), pos.sum()
        if n_neg == 0 or n_pos == 0:
            return float("nan")
        neg_below = np.cumsum(neg) - neg
        return float(np.sum(pos * (neg_below + 0.5 * neg)) / (n_pos * n_neg))

    def roc_curve(self):
        """
        Approximate ROC curve, one point per bin edge from high to low.

        Returns:
            fpr, tpr : ndarray, shape (bins + 1,), from (0, 0) to (1, 1)
        """
        if self._window.total is None:
            raise RuntimeError("no data has been added yet; call update first")
        neg, pos = self._window.total
        fp = np.concatenate([[0], np.cumsum(neg[::-1])])
        tp = np.concatenate([[0], np.cumsum(pos[::-1])])
        with np.errstate(invalid="ignore", divide="ignore"):
            return fp / neg.sum(), tp / pos.sum()
