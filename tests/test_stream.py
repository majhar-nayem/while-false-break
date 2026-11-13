"""Tests for streaming pipelines and numcompute_stream.stream.

Run with:  python -m pytest tests/test_stream.py
"""

import numpy as np
import pytest

from numcompute_stream.pipeline import ColumnSelector, FeatureUnion, Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
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
    return Pipeline([
        ("impute", SimpleImputer()),
        ("scale", StandardScaler()),
        ("model", DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size)),
    ])


def test_pipeline_partial_fit_learns_across_chunks():
    chunks = _stream()
    pipe = _pipeline()
    for X, y in chunks[:-1]:
        pipe.partial_fit(X, y)
    X_last, y_last = chunks[-1]
    assert pipe["model"].n_retrains_ == 2  # 550 samples, buffer 200
    assert np.mean(pipe.predict(X_last) == y_last) > 0.8  # NaN rows imputed on the way
    assert pipe.predict_proba(X_last).shape == (50, 2)
    union = FeatureUnion([
        ("numbers", Pipeline([("pick", ColumnSelector([0])), ("scale", StandardScaler())])),
        ("codes", Pipeline([("pick", ColumnSelector([1])), ("encode", OneHotEncoder())])),
    ])
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
