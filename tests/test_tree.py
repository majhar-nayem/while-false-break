"""Tests for numcompute_stream.tree. Run with:  python -m pytest tests/test_tree.py"""

import numpy as np
import pytest

from numcompute_stream.tree import DecisionTreeClassifier, best_split, best_split_loop

XOR_X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
XOR_y = np.array([0, 1, 1, 0])


def _linear_data(n=300, seed=0):
    """Two-class data separated mainly by column 1."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 4))
    y = (X[:, 1] + 0.3 * X[:, 2] > 0).astype(int)
    return X, y


def test_learns_xor_and_a_threshold():
    tree = DecisionTreeClassifier(max_depth=2).fit(XOR_X, XOR_y)  # needs a zero-gain first split
    assert tree.predict(XOR_X).tolist() == XOR_y.tolist()
    assert tree.depth_ == 2
    x = np.arange(10.0)[:, None]
    stump = DecisionTreeClassifier(max_depth=1).fit(x, x[:, 0] >= 6)
    assert stump.threshold_[0] == 5.5 and stump.n_leaves_ == 2
    assert stump.score(x, x[:, 0] >= 6) == 1.0


def test_vectorised_split_matches_loop_version():
    X, y = _linear_data(120)
    X[:30, 0] = 1.0  # some ties inside a column
    for criterion in ("gini", "entropy"):
        fast = best_split(X, y, 2, criterion)
        slow = best_split_loop(X, y, 2, criterion)
        assert fast[:2] == slow[:2]
        assert fast[2] == pytest.approx(slow[2])


def test_gini_and_entropy_agree_on_a_clear_split():
    X = np.array([[1.0, 5.0], [2.0, 1.0], [3.0, 4.0], [10.0, 2.0], [11.0, 3.0], [12.0, 0.0]])
    y = np.array(["a", "a", "a", "b", "b", "b"])
    gini = DecisionTreeClassifier(max_depth=1, criterion="gini").fit(X, y)
    entropy = DecisionTreeClassifier(max_depth=1, criterion="entropy").fit(X, y)
    assert (gini.feature_[0], gini.threshold_[0]) == (entropy.feature_[0], entropy.threshold_[0])
    assert gini.threshold_[0] == 6.5
    assert gini.predict([[0.0, 0.0], [20.0, 0.0]]).tolist() == ["a", "b"]  # string labels


def test_depth_and_min_samples_split_are_respected():
    X, y = _linear_data(400)
    for depth in (1, 2, 3):
        assert DecisionTreeClassifier(max_depth=depth).fit(X, y).depth_ <= depth
    tree = DecisionTreeClassifier(max_depth=None, min_samples_split=50).fit(X, y)
    inner = tree.feature_ >= 0
    assert np.all(tree.value_[inner].sum(axis=1) >= 50)
    deep = DecisionTreeClassifier(max_depth=None).fit(X, y)
    assert deep.score(X, y) == 1.0  # no depth limit: every training point fits


def test_pure_and_constant_nodes_become_leaves():
    pure = DecisionTreeClassifier().fit(np.random.default_rng(0).normal(size=(20, 3)), np.ones(20))
    assert pure.n_nodes_ == 1 and pure.predict([[0.0, 0.0, 0.0]]).tolist() == [1.0]
    constant = DecisionTreeClassifier().fit(np.ones((6, 2)), [0, 1, 0, 1, 1, 1])
    assert constant.n_nodes_ == 1  # no cut is possible
    assert constant.predict_proba([[1.0, 1.0]]).tolist() == [[2 / 6, 4 / 6]]
    assert best_split(np.ones((5, 2)), np.zeros(5, dtype=int), 1)[0] == -1


def test_tie_rule_is_deterministic():
    # Columns 0 and 1 are identical, so both give the same best split:
    # the lowest column wins, and refitting gives the same tree.
    X = np.repeat(np.arange(8.0)[:, None], 2, axis=1)
    y = (X[:, 0] > 3).astype(int)
    trees = [DecisionTreeClassifier(max_depth=1).fit(X, y) for _ in range(3)]
    assert all(tree.feature_[0] == 0 for tree in trees)
    # Two equally good thresholds in one column: the lower one wins.
    assert best_split(np.array([[0.0], [1.0], [2.0]]), np.array([0, 1, 0]), 2)[1] == 0.5
    # Neighbouring floats: the threshold must still separate them.
    a = 1.0
    b = np.nextafter(a, 2.0)
    tree = DecisionTreeClassifier(max_depth=1).fit([[a], [b]], [0, 1])
    assert tree.predict([[a], [b]]).tolist() == [0, 1]


def test_predict_before_first_retrain_raises():
    tree = DecisionTreeClassifier(buffer_size=10)
    tree.partial_fit(np.zeros((4, 2)), np.zeros(4))
    with pytest.raises(RuntimeError, match="not been trained"):
        tree.predict(np.zeros((1, 2)))
    tree.flush()  # train on the 4 buffered samples
    assert tree.predict(np.zeros((1, 2))).tolist() == [0.0]


def test_retrains_when_buffer_is_full_and_clears_it():
    X, y = _linear_data(250)
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=100)
    retrained_after = []
    for i, start in enumerate(range(0, 250, 40)):  # chunks of 40
        tree.partial_fit(X[start : start + 40], y[start : start + 40])
        retrained_after.append(tree.n_retrains_)
        assert tree.n_buffered_ < 100
    # 40, 80, 120 -> retrain (buffer empty), 40, 80, 120 -> retrain, 10 left over
    assert retrained_after == [0, 0, 1, 1, 1, 2, 2]
    assert tree.n_samples_trained_ == 120 and tree.n_buffered_ == 10


def test_predict_uses_last_tree_between_retrains():
    X, y = _linear_data(300)
    tree = DecisionTreeClassifier(max_depth=3, buffer_size=100).partial_fit(X[:100], y[:100])
    before = tree.predict(X)
    tree.partial_fit(X[100:150], 1 - y[100:150])  # buffered, not trained on yet
    assert np.array_equal(tree.predict(X), before)
    same = DecisionTreeClassifier(max_depth=3).fit(X[:100], y[:100])
    assert np.array_equal(before, same.predict(X))  # buffered retrain == fit on the buffer


def test_class_seen_in_a_later_chunk():
    tree = DecisionTreeClassifier(max_depth=2, buffer_size=4)
    tree.partial_fit([[0.0], [1.0], [2.0], [3.0]], [0, 0, 1, 1])
    assert tree.predict_proba([[0.0]]).shape == (1, 2)
    tree.partial_fit([[10.0], [11.0], [12.0], [13.0]], [2, 2, 1, 1])  # class 2 is new
    assert tree.classes_.tolist() == [0, 1, 2]
    assert tree.predict_proba([[0.0]]).shape == (1, 3)  # class 0 kept, with probability 0
    assert tree.predict([[10.0]]).tolist() == [2]


def test_input_and_parameter_errors():
    tree = DecisionTreeClassifier()
    with pytest.raises(ValueError, match="NaN"):
        tree.fit([[np.nan, 1.0], [2.0, 3.0]], [0, 1])
    with pytest.raises(ValueError):
        tree.fit(np.ones((3, 2)), [0, 1])  # y too short
    with pytest.raises(TypeError):
        tree.fit(np.array([["a"], ["b"]]), [0, 1])
    tree.fit(np.ones((3, 2)), [0, 1, 1])
    with pytest.raises(ValueError):
        tree.predict(np.ones((1, 3)))  # wrong number of columns
    with pytest.raises(ValueError):
        tree.partial_fit(np.ones((2, 5)), [0, 1])
    for bad in (
        {"max_depth": 0},
        {"min_samples_split": 1},
        {"criterion": "mse"},
        {"buffer_size": 0},
        {"max_features": 0},
        {"max_features": 1.5},
    ):
        with pytest.raises(ValueError):
            DecisionTreeClassifier(**bad)
    sampled = DecisionTreeClassifier(max_features="sqrt", random_state=0).fit(*_linear_data())
    again = DecisionTreeClassifier(max_features="sqrt", random_state=0).fit(*_linear_data())
    assert np.array_equal(sampled.feature_, again.feature_)  # same seed, same tree


def test_max_features_options_and_impurity_threshold():
    X, y = _linear_data(200)
    for option, n_tried in (("log2", 2), (0.5, 2), (3, 3), ("sqrt", 2)):
        tree = DecisionTreeClassifier(max_features=option, random_state=1)
        assert tree._n_features_to_try(4) == n_tried
        assert tree.fit(X, y).score(X, y) > 0.8
    with pytest.raises(ValueError, match="more than"):
        DecisionTreeClassifier(max_features=9).fit(X, y)  # only 4 features
    # XOR's first split has zero gain, so any positive threshold stops at the root.
    assert (
        DecisionTreeClassifier(max_depth=2, min_impurity_decrease=0.01).fit(XOR_X, XOR_y).n_nodes_
        == 1
    )
    with pytest.raises(ValueError):
        DecisionTreeClassifier(min_impurity_decrease=-0.1)


def test_label_nan_empty_chunk_and_memory():
    tree = DecisionTreeClassifier(buffer_size=4)
    with pytest.raises(ValueError, match="NaN"):
        tree.fit([[0.0], [1.0]], [0.0, np.nan])
    assert tree.nbytes_ == 0  # nothing buffered or trained yet
    tree.partial_fit(np.zeros((0, 1)), np.zeros(0))  # an empty chunk is ignored
    assert tree.n_buffered_ == 0
    tree.partial_fit([[0.0], [1.0]], [0, 1])
    assert tree.nbytes_ == 2 * 8 + 2 * 8  # buffered X and y only
    tree.partial_fit([[2.0], [3.0]], [0, 1])  # buffer full: retrain
    assert tree.n_buffered_ == 0 and tree.nbytes_ > 0
