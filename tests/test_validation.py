"""Input checks and rarely used paths across numcompute_stream.

Run with:  python -m pytest tests/test_validation.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from numcompute_stream import io, visualise
from numcompute_stream.benchmarking import _format_time, benchmark
from numcompute_stream.metrics import StreamingConfusionMatrix, auc, roc_curve
from numcompute_stream.pipeline import Estimator, FeatureUnion, Pipeline
from numcompute_stream.preprocessing import OneHotEncoder, SimpleImputer, StandardScaler
from numcompute_stream.sort_search import binary_search, topk
from numcompute_stream.stats import RunningStats, StreamingHistogram, histogram, mean
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import DecisionTreeClassifier, _impurity_loop, best_split
from numcompute_stream.utils import argtopk, batch_iterator


def test_stats_input_checks():
    with pytest.raises(ValueError):
        mean(np.ones((0, 3)), axis=0)  # empty axis
    with pytest.raises(TypeError):
        histogram(["a", "b"])
    with pytest.raises(ValueError):
        histogram([1.0, 2.0], range=(3, 1))
    assert histogram([np.nan], bins=2)[0].sum() == 0  # no values: default range, no counts
    running = RunningStats()
    with pytest.raises(ValueError):
        _ = running.mean  # nothing added yet
    with pytest.raises(TypeError):
        running.update(["a"])
    with pytest.raises(ValueError):
        running.update(np.ones((2, 2, 2)))
    with pytest.raises(ValueError):
        StreamingHistogram().update(np.ones((2, 2, 2)))
    with pytest.raises(RuntimeError):
        _ = StreamingHistogram().counts
    with pytest.raises(ValueError):
        running.var()


def test_metric_input_checks():
    with pytest.raises(TypeError):
        roc_curve([0, 1], ["a", "b"])
    with pytest.raises(ValueError):
        auc([0, 1], [0, 1, 2])
    assert StreamingConfusionMatrix().n_samples_ == 0


def test_pipeline_input_checks():
    with pytest.raises(NotImplementedError):
        Estimator().fit(np.ones((1, 1)), [0])
    with pytest.raises(NotImplementedError):
        Estimator().predict(np.ones((1, 1)))
    scaler_only = Pipeline([("scale", StandardScaler())]).fit(np.ones((2, 1)))
    with pytest.raises(TypeError):
        scaler_only.predict(np.ones((1, 1)))
    with pytest.raises(TypeError):
        scaler_only.predict_proba(np.ones((1, 1)))

    class BatchOnly:
        def fit(self, X, y=None):
            return self

        def transform(self, X):
            return X

    with pytest.raises(TypeError, match="batch"):
        FeatureUnion([("batch", BatchOnly())]).partial_fit(np.ones((2, 1)))


def test_preprocessing_paths():
    scaler = StandardScaler().partial_fit([[0.0], [2.0]])
    scaler.fit([[10.0], [10.0]])  # fit forgets earlier chunks
    assert scaler.mean_.tolist() == [10.0] and scaler.n_samples_seen_.tolist() == [2]
    with pytest.raises(ValueError):
        SimpleImputer(strategy="mode")
    with pytest.raises(ValueError):
        OneHotEncoder(handle_unknown="drop")
    text = np.array([["a", ""], ["b", ""]])
    frequent = SimpleImputer(strategy="most_frequent", missing_values="", fill_value="none")
    assert frequent.fit(text).statistics_.tolist() == ["a", "none"]  # empty column -> fill_value
    assert SimpleImputer(strategy="constant").fit(text).statistics_.tolist() == ["missing"] * 2
    assert SimpleImputer().fit(np.array([[1], [3]])).statistics_.tolist() == [2.0]  # ints
    nan_marker_on_text = SimpleImputer(strategy="most_frequent").fit(np.array([["x"], ["y"]]))
    assert nan_marker_on_text.transform(np.array([["x"]])).tolist() == [["x"]]


def test_tree_and_trainer_paths():
    assert best_split(np.ones((1, 2)), np.zeros(1, dtype=int), 1)[0] == -1  # one sample
    assert _impurity_loop([0, 0], 0, "gini") == 0.0
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=10)
    assert "buffer_size=10" in repr(tree)
    with pytest.raises(ValueError):
        tree.fit(np.ones(3), [0, 1, 0])  # 1-D X
    with pytest.raises(ValueError):
        tree.fit(np.ones((0, 2)), [])
    tree.fit([[0.0], [1.0]], ["a", "b"])
    trainer = StreamTrainer(tree, pos_label="zzz")  # positive class never seen
    record = trainer.process_chunk([[0.0], [1.0]], ["a", "b"])
    assert record["cumulative_auc"] is None and record["accuracy"] == 1.0


def test_visualise_paths(monkeypatch):
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))
    visualise.plot_metric_over_time([0.5, 0.6], label="tree", show=True)
    assert shown == [True]
    _, ax = visualise.plot_metric_over_time([0.5, 0.6], label="tree")
    assert ax.get_legend().get_texts()[0].get_text() == "tree"
    with pytest.raises(ValueError):
        visualise.compare_models([0.1], [0.2], labels=("only one",))
    plt.close("all")


def test_io_input_checks(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert io.read_header(empty) == []
    data = tmp_path / "data.csv"
    data.write_text("a,b\n1,2\n\n3,\n")  # a blank line and a missing cell
    assert io.load_csv(data).shape == (2, 2)  # the blank line is skipped
    with pytest.raises(ValueError, match="not found"):
        io.load_csv(data, usecols="c")
    with pytest.raises(ValueError, match="out of range"):
        io.load_csv(data, usecols=5)
    with pytest.raises(ValueError):
        io.load_csv(data, missing="ignore")
    with pytest.raises(ValueError, match="integer"):
        io.load_csv(data, dtype=int)
    with pytest.raises(ValueError):
        list(io.iter_csv_chunks(data, chunksize=0))
    ragged = tmp_path / "ragged.csv"
    ragged.write_text("a,b\n1,2,3\n")
    with pytest.raises(ValueError, match="expected 2 fields"):
        io.load_csv(ragged)
    header_only = tmp_path / "header.csv"
    header_only.write_text("a,b\n")
    assert io.load_csv(header_only).shape == (0, 2)
    assert io.load_csv(header_only, dtype=str).dtype.kind == "U"


def test_helper_input_checks():
    with pytest.raises(ValueError):
        benchmark(lambda: None, number=0)
    assert _format_time(5e-10).endswith("ns")
    values, indices = topk([3.0, 1.0], 0)
    assert values.size == 0 and indices.size == 0
    with pytest.raises(TypeError):
        binary_search([1.0, 2.0], ["x"])
    with pytest.raises(ValueError):
        argtopk(np.float64(1.0), 1)
    with pytest.raises(TypeError):
        argtopk([1.0, 2.0], 1.5)
    with pytest.raises(ValueError):
        next(batch_iterator(np.float64(1.0), 2))
