

from __future__ import annotations

import numpy as np

from numcompute.utils import check_array, check_consistent_length

__all__ = [
    "confusion_matrix",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "precision_recall_f1",
    "mse",
    "rmse",
    "mae",
    "r2_score",
    "roc_curve",
    "auc",
    "roc_auc_score",
]

AVERAGES = ("binary", "macro", "micro", "weighted", None)


def _check_targets(y_true, y_pred, numeric: bool = False):
    yt = check_array(y_true, name="y_true", ndim=1, numeric=numeric, allow_empty=False)
    yp = check_array(y_pred, name="y_pred", ndim=1, numeric=numeric, allow_empty=False)
    check_consistent_length(yt, yp, names=("y_true", "y_pred"))
    return yt, yp


def confusion_matrix(y_true, y_pred, labels=None, normalize: str | None = None):
    
    yt, yp = _check_targets(y_true, y_pred)
    if normalize not in (None, "true", "pred", "all"):
        raise ValueError(
            f"normalize must be None, 'true', 'pred' or 'all', got {normalize!r}"
        )
    labels = (
        np.unique(np.concatenate([yt, yp])) if labels is None else np.asarray(labels)
    )
    if labels.size == 0:
        raise ValueError("labels must not be empty")
    k = labels.size
    sorter = np.argsort(labels, kind="stable")
    sl = labels[sorter]

    def codes(y):
        pos = np.clip(np.searchsorted(sl, y), 0, k - 1)
        ok = sl[pos] == y
        return sorter[pos], ok

    ct, okt = codes(yt)
    cp, okp = codes(yp)
    keep = okt & okp
    C = np.bincount(ct[keep] * k + cp[keep], minlength=k * k).reshape(k, k)
    if normalize is None:
        return C.astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        if normalize == "true":
            den = C.sum(axis=1, keepdims=True)
        elif normalize == "pred":
            den = C.sum(axis=0, keepdims=True)
        else:
            den = np.array(C.sum())
        return np.where(den > 0, C / np.where(den > 0, den, 1), 0.0)


def accuracy(y_true, y_pred) -> float:
    
    yt, yp = _check_targets(y_true, y_pred)
    return float(np.mean(yt == yp))


def precision_recall_f1(
    y_true,
    y_pred,
    average: str | None = "binary",
    pos_label=1,
    labels=None,
    zero_division: float = 0.0,
):
    
    if average not in AVERAGES:
        raise ValueError(f"average must be one of {AVERAGES}, got {average!r}")
    yt, yp = _check_targets(y_true, y_pred)
    if labels is None:
        labels = np.unique(np.concatenate([yt, yp]))
    labels = np.asarray(labels)
    if average == "binary":
        if labels.size > 2:
            raise ValueError(
                f"average='binary' needs at most 2 labels, found {labels.size} "
                f"({labels.tolist()}); use average='macro', 'micro' or 'weighted'"
            )
        if not np.any(labels == pos_label):
            labels = np.append(labels, pos_label)
    C = confusion_matrix(yt, yp, labels=labels)
    tp = np.diag(C).astype(np.float64)
    fp = C.sum(axis=0) - tp
    fn = C.sum(axis=1) - tp
    support = C.sum(axis=1)

    def div(a, b):
        return np.where(b > 0, a / np.where(b > 0, b, 1), zero_division)

    if average == "micro":
        tp, fp, fn = tp.sum(), fp.sum(), fn.sum()
    p = div(tp, tp + fp)
    r = div(tp, tp + fn)
    # F1 = 2PR/(P+R) = 2TP/(2TP+FP+FN); the count form is 0 (not
    # zero_division) when TP = 0 but FP or FN > 0.
    f = div(2 * tp, 2 * tp + fp + fn)

    if average == "binary":
        i = int(np.flatnonzero(labels == pos_label)[0])
        return float(p[i]), float(r[i]), float(f[i])
    if average == "micro":
        return float(p), float(r), float(f)
    if average == "macro":
        return float(p.mean()), float(r.mean()), float(f.mean())
    if average == "weighted":
        w = support / support.sum() if support.sum() else np.zeros_like(p)
        return float(p @ w), float(r @ w), float(f @ w)
    return p, r, f


def precision(
    y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0.0
):
    
    return precision_recall_f1(
        y_true, y_pred, average, pos_label, labels, zero_division
    )[0]


def recall(
    y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0.0
):
   
    return precision_recall_f1(
        y_true, y_pred, average, pos_label, labels, zero_division
    )[1]


def f1(y_true, y_pred, average="binary", pos_label=1, labels=None, zero_division=0.0):
    
    return precision_recall_f1(
        y_true, y_pred, average, pos_label, labels, zero_division
    )[2]


# --------------------------------------------------------------------------
# Regression
# --------------------------------------------------------------------------
def _check_regression(y_true, y_pred):
    yt = check_array(y_true, name="y_true", to_float=True, allow_empty=False)
    yp = check_array(y_pred, name="y_pred", to_float=True, allow_empty=False)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true has shape {yt.shape} but y_pred has shape {yp.shape}")
    return yt, yp


def mse(y_true, y_pred) -> float:
    
    yt, yp = _check_regression(y_true, y_pred)
    d = yt - yp
    return float(np.mean(d * d))


def rmse(y_true, y_pred) -> float:
    
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
   
    yt, yp = _check_regression(y_true, y_pred)
    return float(np.mean(np.abs(yt - yp)))


def r2_score(y_true, y_pred) -> float:
    
    yt, yp = _check_regression(y_true, y_pred)
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------
# ROC / AUC (bonus)
# --------------------------------------------------------------------------
def roc_curve(y_true, y_score, pos_label=1):
    
    yt = np.asarray(y_true)
    s = check_array(y_score, name="y_score", ndim=1, to_float=True, allow_nan=False)
    if yt.ndim != 1:
        raise ValueError(f"y_true must be 1-D, got shape {yt.shape}")
    check_consistent_length(yt, s, names=("y_true", "y_score"))
    pos = yt == pos_label
    n_pos = int(pos.sum())
    n_neg = pos.size - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("roc_curve needs both positive and negative samples in y_true")
    order = np.argsort(-s, kind="stable")
    s_sorted = s[order]
    p_sorted = pos[order]
    distinct = np.flatnonzero(np.diff(s_sorted))
    ends = np.r_[distinct, s_sorted.size - 1]  # last index of each tie block
    tps = np.cumsum(p_sorted)[ends]
    fps = (ends + 1) - tps
    tpr = np.r_[0.0, tps / n_pos]
    fpr = np.r_[0.0, fps / n_neg]
    thresholds = np.r_[np.inf, s_sorted[ends]]
    return fpr, tpr, thresholds


def auc(x, y) -> float:
    
    x = check_array(x, name="x", ndim=1, to_float=True)
    y = check_array(y, name="y", ndim=1, to_float=True)
    check_consistent_length(x, y, names=("x", "y"))
    if x.size < 2:
        raise ValueError(
            f"at least 2 points are needed to compute an area, got {x.size}"
        )
    dx = np.diff(x)
    if np.all(dx >= 0):
        sign = 1.0
    elif np.all(dx <= 0):
        sign = -1.0
    else:
        raise ValueError("x must be monotonic (increasing or decreasing)")
    return float(sign * np.sum(dx * (y[1:] + y[:-1]) / 2.0))


def roc_auc_score(y_true, y_score, pos_label=1) -> float:
    
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    return auc(fpr, tpr)
