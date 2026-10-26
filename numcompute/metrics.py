"""Evaluation metrics: classification, regression (MSE), and ROC / AUC."""

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
