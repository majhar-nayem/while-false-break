"""StreamTrainer: train a model chunk by chunk and log how it does over time.

Evaluation is test-then-train (also called prequential): every chunk is first
scored with the current model, while it is still unseen, and only then used
for training. Each chunk's score is therefore an honest estimate on new data,
without holding out a separate test set.

    trainer = StreamTrainer(pipe, window=5, pos_label=1)
    for X_chunk, y_chunk in stream:
        trainer.process_chunk(X_chunk, y_chunk)   # score_chunk, then fit_chunk
    trainer.history("cumulative_accuracy")       # one value per chunk
"""

import time

import numpy as np

from numcompute_stream.metrics import StreamingAccuracy, StreamingAUC

LOG_KEYS = (
    "chunk",
    "n_samples",
    "accuracy",
    "cumulative_accuracy",
    "rolling_accuracy",
    "cumulative_auc",
    "retrained",
    "n_retrains",
    "n_buffered",
    "fit_time",
    "predict_time",
    "memory_bytes",
)


def memory_footprint(obj, _seen=None):
    """
    Bytes held in NumPy arrays by obj and everything it refers to.

    Follows object attributes, lists, tuples, dicts and deques, counting
    each array once. This covers what a streaming model keeps between
    chunks: buffered rows, tree arrays and running statistics.

    Complexity:
        O(number of objects reachable from obj).
    """
    if _seen is None:
        _seen = set()
    if id(obj) in _seen:
        return 0
    _seen.add(id(obj))
    if isinstance(obj, np.ndarray):
        return obj.nbytes
    if isinstance(obj, dict):
        children = obj.values()
    elif isinstance(obj, (list, tuple, set, frozenset)) or type(obj).__name__ == "deque":
        children = obj
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        children = vars(obj).values()
    else:
        return 0
    return sum(memory_footprint(child, _seen) for child in children)


def _final_estimator(model):
    """The model itself, or the last step of a Pipeline."""
    steps = getattr(model, "steps", None)
    return steps[-1][1] if steps else model


class StreamTrainer:
    """
    Manage a streaming model: score each chunk, train on it, and log the results.

    Parameters:
        model : any object with partial_fit(X, y) and predict(X), e.g. a
            DecisionTreeClassifier or a Pipeline ending in one
        window : int >= 1 or None, number of chunks for rolling_accuracy
        pos_label : label of the positive class. If given and the model has
            predict_proba, the cumulative ROC AUC of that class is logged.

    Attributes:
        log_ : list of dicts, one per processed chunk, with the keys in LOG_KEYS:
            chunk, n_samples : chunk index and size
            accuracy : accuracy on this chunk before training on it
                (None while the model has not been trained yet)
            cumulative_accuracy, rolling_accuracy, cumulative_auc : over all
                scored chunks / the last `window` chunks (None if unavailable)
            retrained, n_retrains, n_buffered : did this chunk trigger a
                retrain, the total so far, and samples waiting in the buffer
            fit_time, predict_time : seconds spent in partial_fit and predict
            memory_bytes : memory_footprint(model) after training

    Raises:
        TypeError if the model has no partial_fit or predict method.

    Complexity:
        the model's own costs, plus O(n) per chunk for the metrics.
    """

    def __init__(self, model, window=None, pos_label=None):
        for method in ("partial_fit", "predict"):
            if not hasattr(model, method):
                raise TypeError(f"model has no {method} method")
        self.model = model
        self.window = window
        self.pos_label = pos_label
        self.reset()

    def reset(self):
        """Clear the log and the metrics (the model is not changed). Returns self."""
        self.log_ = []
        self._cumulative = StreamingAccuracy()
        self._rolling = StreamingAccuracy(window=self.window) if self.window else None
        self._auc = StreamingAUC(pos_label=self.pos_label) if self.pos_label is not None else None
        self._pending = None  # score_chunk results waiting for fit_chunk
        return self

    def score_chunk(self, X, y):
        """
        Score the current model on a chunk and update the running metrics.

        Parameters:
            X : array_like, shape (n, d)
            y : array_like, shape (n,)

        Returns:
            dict with accuracy, cumulative_accuracy, rolling_accuracy,
            cumulative_auc and predict_time; the metric values are None while
            the model has not been trained yet. The chunk is not trained on.
        """
        y = np.asarray(y)
        start = time.perf_counter()
        try:
            y_pred = self.model.predict(X)
        except RuntimeError:  # not trained yet: nothing to score
            y_pred = None
        predict_time = time.perf_counter() - start

        scores = {
            "accuracy": None,
            "cumulative_accuracy": None,
            "rolling_accuracy": None,
            "cumulative_auc": None,
            "predict_time": predict_time,
        }
        if y_pred is not None and y.size > 0:
            scores["accuracy"] = float(np.mean(y_pred == y))
            scores["cumulative_accuracy"] = self._cumulative.update(y, y_pred).result()
            if self._rolling is not None:
                scores["rolling_accuracy"] = self._rolling.update(y, y_pred).result()
            if self._auc is not None and hasattr(self.model, "predict_proba"):
                scores["cumulative_auc"] = self._auc_update(X, y)
        self._pending = scores
        return dict(scores)

    def _auc_update(self, X, y):
        """Add the positive-class probabilities of this chunk to the running AUC."""
        classes = getattr(_final_estimator(self.model), "classes_", None)
        if classes is None or self.pos_label not in classes:
            return None
        proba = self.model.predict_proba(X)[:, np.flatnonzero(classes == self.pos_label)[0]]
        value = self._auc.update(y, proba).result()
        return None if np.isnan(value) else value

    def fit_chunk(self, X, y):
        """
        Train the model on a chunk (model.partial_fit) and append a log record.

        If score_chunk was called just before, its results go in the same
        record; otherwise the metric fields are None.

        Returns:
            dict, the new log record
        """
        estimator = _final_estimator(self.model)
        retrains_before = getattr(estimator, "n_retrains_", None)
        start = time.perf_counter()
        self.model.partial_fit(X, y)
        fit_time = time.perf_counter() - start
        retrains_after = getattr(estimator, "n_retrains_", None)

        scores = self._pending or {
            "accuracy": None,
            "cumulative_accuracy": None,
            "rolling_accuracy": None,
            "cumulative_auc": None,
            "predict_time": 0.0,
        }
        self._pending = None
        record = {
            "chunk": len(self.log_),
            "n_samples": int(np.asarray(y).shape[0]),
            **scores,
            "retrained": retrains_before is not None and retrains_after > retrains_before,
            "n_retrains": retrains_after,
            "n_buffered": getattr(estimator, "n_buffered_", None),
            "fit_time": fit_time,
            "memory_bytes": memory_footprint(self.model),
        }
        self.log_.append({key: record[key] for key in LOG_KEYS})
        return self.log_[-1]

    def process_chunk(self, X, y):
        """Test-then-train on one chunk: score_chunk(X, y), then fit_chunk(X, y). Returns the record."""
        self.score_chunk(X, y)
        return self.fit_chunk(X, y)

    def run(self, chunks, flush=False):
        """
        Process every (X_chunk, y_chunk) pair of an iterable. Returns self.

        Parameters:
            chunks : iterable of (X_chunk, y_chunk)
            flush : bool, at the end call the model's flush() (if it has one)
                so the samples left in the buffer are trained on too
        """
        for X, y in chunks:
            self.process_chunk(X, y)
        if flush:
            estimator = _final_estimator(self.model)
            if hasattr(estimator, "flush"):
                estimator.flush()
        return self

    def history(self, key):
        """
        One logged value per chunk as a float array (None becomes NaN), ready to plot.

        Raises:
            KeyError if key is not one of LOG_KEYS.
        """
        if key not in LOG_KEYS:
            raise KeyError(f"unknown log key {key!r}; choose from {LOG_KEYS}")
        values = [record[key] for record in self.log_]
        return np.array([np.nan if v is None else v for v in values], dtype=float)
