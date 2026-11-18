"""Tests for streaming pipelines and numcompute_stream.stream.

Run with:  python -m pytest tests/test_stream.py
"""

import numpy as np
import pytest

from numcompute_stream.metrics import StreamingF1, StreamingMetric, StreamingRecall, f1
from numcompute_stream.pipeline import ColumnSelector, FeatureUnion, Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.stats import StreamingStats
from numcompute_stream.stream import LOG_KEYS, StreamTrainer, memory_footprint
from numcompute_stream.tree import DecisionTreeClassifier


def _stream(n=600, chunk=50, seed=0):
    """Two-class data with some missing values, cut into chunks."""
    rng = np.random.default_rng(seed)
    X = rng.normal(loc=10.0, scale=3.0, size=(n, 3))
    y = (X[:, 0] + X[:, 1] > 20).astype(int)
    X[rng.random(X.shape) < 0.05] = np.nan
    return [(X[i : i + chunk], y[i : i + chunk]) for i in range(0, n, chunk)]


def _pipeline(buffer_size=200):
    return Pipeline(
        [
            ("impute", SimpleImputer()),
            ("scale", StandardScaler()),
            ("model", DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size)),
        ]
    )


def test_pipeline_partial_fit_learns_across_chunks():
    chunks = _stream()
    pipe = _pipeline()
    for X, y in chunks[:-1]:
        pipe.partial_fit(X, y)
    X_last, y_last = chunks[-1]
    assert pipe["model"].n_retrains_ == 2  # 550 samples, buffer 200
    assert np.mean(pipe.predict(X_last) == y_last) > 0.8  # NaN rows imputed on the way
    assert pipe.predict_proba(X_last).shape == (50, 2)
    union = FeatureUnion(
        [
            ("numbers", Pipeline([("pick", ColumnSelector([0])), ("scale", StandardScaler())])),
            ("codes", Pipeline([("pick", ColumnSelector([1])), ("encode", OneHotEncoder())])),
        ]
    )
    union.partial_fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert union.transform(np.array([[1.0, 4.0]])).shape == (1, 3)


def test_pipeline_step_without_partial_fit_is_named():
    class BatchOnly:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    pipe = Pipeline([("batch_only", BatchOnly()), ("model", DecisionTreeClassifier())])
    with pytest.raises(TypeError, match="batch_only"):
        pipe.partial_fit(np.ones((2, 1)), [0, 1])
    with pytest.raises(TypeError):
        StreamTrainer(BatchOnly())


def test_trainer_scores_each_chunk_before_training_on_it():
    chunks = _stream()
    trainer = StreamTrainer(_pipeline(), window=3, pos_label=1).run(chunks)
    log = trainer.log_
    assert len(log) == len(chunks) and all(set(r) == set(LOG_KEYS) for r in log)
    # Nothing to score until the first retrain (after chunk 3, at 200 samples).
    assert [r["accuracy"] is None for r in log[:5]] == [True, True, True, True, False]
    assert [r["retrained"] for r in log].count(True) == 3
    assert np.isnan(trainer.history("accuracy")[0])
    # Test-then-train: chunk 3's score comes from the model before chunk 3 was added.
    fresh = _pipeline()
    for X, y in chunks[:4]:
        fresh.partial_fit(X, y)
    X4, y4 = chunks[4]
    assert log[4]["accuracy"] == np.mean(fresh.predict(X4) == y4)


def test_cumulative_and_rolling_accuracy_match_hand_calculation():
    chunks = _stream()
    trainer = StreamTrainer(_pipeline(), window=2).run(chunks)
    scored = [(r["accuracy"], r["n_samples"]) for r in trainer.log_ if r["accuracy"] is not None]
    correct = sum(acc * n for acc, n in scored)
    total = sum(n for _, n in scored)
    assert trainer.log_[-1]["cumulative_accuracy"] == pytest.approx(correct / total)
    last_two = scored[-2:]
    expected_rolling = sum(a * n for a, n in last_two) / sum(n for _, n in last_two)
    assert trainer.log_[-1]["rolling_accuracy"] == pytest.approx(expected_rolling)
    assert 0.5 < trainer.log_[-1]["cumulative_accuracy"] <= 1.0
    with pytest.raises(KeyError):
        trainer.history("loss")


def test_memory_and_auc_are_logged():
    trainer = StreamTrainer(_pipeline(buffer_size=100), pos_label=1).run(_stream(), flush=True)
    memory = trainer.history("memory_bytes")
    assert np.all(memory > 0)
    # The buffer empties at each retrain, so memory drops right after one.
    retrained = np.flatnonzero(trainer.history("retrained") == 1)
    assert np.all(memory[retrained] < memory[retrained - 1])
    assert 0.5 < trainer.log_[-1]["cumulative_auc"] <= 1.0
    assert memory_footprint({"a": np.zeros(10), "b": [np.zeros(5)]}) == 120


def test_trainer_does_not_hide_real_prediction_errors():
    class Broken:
        n_retrains_ = 1  # reports itself as trained

        def partial_fit(self, X, y):
            return self

        def predict(self, X):
            raise RuntimeError("something else went wrong")

    with pytest.raises(RuntimeError, match="something else"):
        StreamTrainer(Broken()).process_chunk(np.ones((2, 1)), [0, 1])


def test_non_contiguous_chunks():
    rng = np.random.default_rng(3)
    base = rng.normal(size=(400, 6))
    X = base[::2, ::2]  # strided view: every other row and column
    assert not X.flags["C_CONTIGUOUS"]
    y = (X[:, 0] > 0).astype(int)
    chunks = [(X[i : i + 50], y[i : i + 50]) for i in range(0, 200, 50)]
    copies = [(np.ascontiguousarray(a), b.copy()) for a, b in chunks]
    from_views = StreamTrainer(_pipeline(buffer_size=100)).run(chunks)
    from_copies = StreamTrainer(_pipeline(buffer_size=100)).run(copies)
    assert np.array_equal(
        from_views.history("accuracy"), from_copies.history("accuracy"), equal_nan=True
    )
    stats = StreamingStats()
    for chunk in np.array_split(base.T, 3, axis=1):  # transposed chunks are column-major
        stats.update_stats(chunk.T[:, ::3])
    assert np.allclose(stats.mean, base[:, ::3].mean(axis=0))


def test_trainer_logs_extra_metrics():
    chunks = _stream()
    metrics = {"f1": StreamingF1(), "recent_recall": StreamingRecall(window=2)}
    trainer = StreamTrainer(_pipeline(), metrics=metrics).run(chunks)
    assert "f1" in trainer.log_[-1] and "recent_recall" in trainer.log_[-1]
    assert np.isnan(trainer.history("f1")[0])  # not scored before the first retrain
    fresh = _pipeline()
    y_true, y_pred = [], []
    for X, y in chunks:  # the same test-then-train loop, by hand
        if fresh["model"].n_retrains_ > 0:
            y_true.append(y)
            y_pred.append(fresh.predict(X))
        fresh.partial_fit(X, y)
    expected = f1(np.concatenate(y_true), np.concatenate(y_pred))
    assert trainer.log_[-1]["f1"] == pytest.approx(expected)
    trainer.reset()
    with pytest.raises(RuntimeError):
        metrics["f1"].result()  # reset() also resets the extra metrics
    with pytest.raises(ValueError, match="clashes"):
        StreamTrainer(_pipeline(), metrics={"accuracy": StreamingF1()})
    with pytest.raises(TypeError):
        StreamTrainer(_pipeline(), metrics={"bad": object()})
    for method, args in (("update", ([0], [0])), ("result", ()), ("reset", ())):
        with pytest.raises(NotImplementedError):
            getattr(StreamingMetric(), method)(*args)
