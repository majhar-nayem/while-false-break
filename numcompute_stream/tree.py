"""Decision tree classifier (CART) with buffered retraining for streams.

The tree is stored as flat arrays indexed by node id instead of node objects:
    feature_[i]   column tested at node i (-1 for a leaf)
    threshold_[i] go left if x[feature] <= threshold
    left_[i], right_[i]  child node ids (-1 for a leaf)
    value_[i]     class counts of the training samples that reached node i
Because of this, prediction moves every sample down the tree at once, one
level per step, with NumPy indexing.

Streaming: partial_fit(X_chunk, y_chunk) adds the chunk to a buffer; once the
buffer holds at least buffer_size samples the tree is retrained from scratch
on the buffer, and the buffer is emptied. Between retrains predict uses the
last trained tree.
"""

import math

import numpy as np

from numcompute_stream.pipeline import Estimator

CRITERIA = ("gini", "entropy")

# Splits whose impurity decrease is below min_impurity_decrease by more than
# this are rejected; it absorbs rounding error in the impurity sums.
GAIN_TOLERANCE = 1e-12


def _impurity(counts, n, criterion):
    """
    Impurity of class counts (..., k) with totals n (broadcastable to counts[..., 0]).

    gini    = 1 - sum(p^2)
    entropy = -sum(p * log2(p)), with 0 * log2(0) taken as 0
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        p = counts / np.asarray(n)[..., None]
    p = np.nan_to_num(p)  # n == 0: no samples, impurity 0
    if criterion == "gini":
        return 1.0 - np.sum(p * p, axis=-1)
    safe_p = np.where(p > 0, p, 1.0)  # log2(1) = 0, so empty classes add nothing
    return -np.sum(p * np.log2(safe_p), axis=-1)


def _midpoint(low, high):
    """Threshold between two sorted values that sends low left and high right."""
    mid = low + (high - low) / 2
    # For neighbouring floats the midpoint can round up to high; use low then.
    return np.where(mid < high, mid, low)


def best_split(X, y_codes, n_classes, criterion="gini"):
    """
    Find the best (feature, threshold) split of one node, vectorised.

    Parameters:
        X : ndarray, shape (n, f), finite floats
        y_codes : ndarray of int, shape (n,), class codes in [0, n_classes)
        n_classes : int
        criterion : "gini" or "entropy"

    Returns:
        (feature, threshold, impurity): column index into X, the threshold,
        and the weighted impurity of the two children; (-1, nan, inf) if no
        split is possible (fewer than 2 samples or every column constant).

    Notes:
        1. Each column is sorted (argsort along axis 0, all columns at once)
           and the one-hot labels are put in the same order: shape (n, f, k).
        2. A cumulative sum down the rows gives, for every possible cut
           position and every column, the class counts on the left; the
           right side is the node total minus the left.
        3. The impurity of both sides is computed for every cut at once and
           weighted by the side sizes.
        4. Cuts between two equal values are not allowed (they cannot be
           expressed as x <= threshold). The smallest weighted impurity wins;
           np.argmin returns the first one, so ties go to the lowest column
           and then the lowest threshold.

    Complexity:
        O(f * n log n) time for the sorts, O(n * f * k) memory.
    """
    n, n_features = X.shape
    if n < 2 or n_features == 0:
        return -1, np.nan, np.inf
    order = np.argsort(X, axis=0, kind="stable")  # (n, f)
    x_sorted = np.take_along_axis(X, order, axis=0)
    one_hot = np.zeros((n, n_classes))
    one_hot[np.arange(n), y_codes] = 1.0

    left = np.cumsum(one_hot[order], axis=0)[:-1]  # (n - 1, f, k): cut after row i
    right = one_hot.sum(axis=0) - left
    n_left = np.arange(1, n)[:, None]  # (n - 1, 1)
    n_right = n - n_left
    weighted = (
        n_left * _impurity(left, n_left, criterion) + n_right * _impurity(right, n_right, criterion)
    ) / n  # (n - 1, f)

    can_cut = x_sorted[1:] > x_sorted[:-1]
    if not can_cut.any():
        return -1, np.nan, np.inf
    weighted = np.where(can_cut, weighted, np.inf)
    best = np.argmin(weighted.T)  # transposed: column first, so ties go to the lowest column
    feature, row = divmod(int(best), n - 1)
    threshold = float(_midpoint(x_sorted[row, feature], x_sorted[row + 1, feature]))
    return feature, threshold, float(weighted[row, feature])


def _impurity_loop(counts, n, criterion):
    """Impurity of a list of class counts with plain Python (used by best_split_loop)."""
    if n == 0:
        return 0.0
    if criterion == "gini":
        return 1.0 - sum((c / n) ** 2 for c in counts)
    return -sum((c / n) * math.log2(c / n) for c in counts if c > 0)


def best_split_loop(X, y_codes, n_classes, criterion="gini"):
    """
    Same result as best_split, written with plain Python loops (the benchmark baseline).

    It uses the same algorithm, so the comparison measures vectorisation
    only: each column is sorted once, then the cut moves down one row at a
    time and one sample's class count moves from the right side to the left.

    Complexity:
        O(f * n log n + f * n * k) time, O(n + k) memory.
    """
    n, n_features = X.shape
    labels_all = y_codes.tolist()
    total = [0] * n_classes
    for label in labels_all:
        total[label] += 1
    best = (-1, np.nan, np.inf)
    for feature in range(n_features):
        column = X[:, feature].tolist()
        order = sorted(range(n), key=column.__getitem__)  # stable, like kind="stable"
        left = [0] * n_classes
        right = list(total)
        for i in range(n - 1):
            label = labels_all[order[i]]
            left[label] += 1
            right[label] -= 1
            low, high = column[order[i]], column[order[i + 1]]
            if not low < high:
                continue  # cannot cut between equal values
            n_left = i + 1
            n_right = n - n_left
            score = (
                n_left * _impurity_loop(left, n_left, criterion)
                + n_right * _impurity_loop(right, n_right, criterion)
            ) / n
            if score < best[2]:
                best = (feature, float(_midpoint(low, high)), score)
    return best


class DecisionTreeClassifier(Estimator):
    """
    Depth-limited CART classification tree with buffered retraining for streams.

    Example:
        tree = DecisionTreeClassifier(max_depth=5, buffer_size=500)
        for X_chunk, y_chunk in stream:
            tree.partial_fit(X_chunk, y_chunk)
        tree.flush()  # train on what is left in the buffer, if anything
        tree.predict(X_new)

    Parameters:
        max_depth : int >= 1 or None, maximum depth (root = depth 0); None = no limit
        min_samples_split : int >= 2, a node with fewer samples becomes a leaf
        max_features : None (all columns), int in [1, d], float in (0, 1]
            (that fraction of d), "sqrt" or "log2". A new random subset of
            columns is searched at every node (drawn from random_state).
        criterion : "gini" (default) or "entropy"
        min_impurity_decrease : float >= 0, split only if the weighted
            impurity drops by at least this much
        buffer_size : int >= 1, samples to collect before partial_fit retrains
        random_state : int or None, seed for max_features

    Attributes (after the first training):
        classes_ : ndarray, every class seen so far, sorted
        n_features_in_ : int
        feature_, threshold_, left_, right_ : ndarray, shape (n_nodes,)
        value_ : ndarray, shape (n_nodes, n_classes), class counts per node
        depth_ : int, depth of the trained tree
        n_retrains_ : int, how many times the tree has been (re)trained
        n_samples_trained_ : int, samples used by the last training

    Raises:
        ValueError for bad parameters, X that is not 2-D or contains NaN or
            inf (impute first), y of the wrong length, or a chunk with a
            different number of columns.
        RuntimeError from predict before the tree has been trained once.

    Notes:
        Stopping rules: depth == max_depth, fewer than min_samples_split
        samples, a pure node, no possible cut (every column constant), or an
        impurity decrease below min_impurity_decrease. As in CART, a split
        with zero decrease is allowed (min_impurity_decrease=0), which is what
        lets a depth-2 tree learn XOR.

        Each retrain sees only the buffered samples, not the older ones: the
        tree always reflects the most recent buffer_size (or more) samples,
        which suits data that drifts over time, at the cost of forgetting.
        classes_ keeps every class seen in any chunk, so the class order and
        predict_proba's columns never change between retrains; a class that
        is missing from the current buffer just gets probability 0.

    Complexity:
        Training: O(depth * f * n log n) time, O(n * f * k) memory for the
        split search. Prediction: O(m * depth). partial_fit stores at most
        about buffer_size + chunk size samples.
    """

    # Split search used by _build; the benchmark swaps in best_split_loop.
    _split = staticmethod(best_split)

    def __init__(
        self,
        max_depth=5,
        min_samples_split=2,
        max_features=None,
        criterion="gini",
        min_impurity_decrease=0.0,
        buffer_size=500,
        random_state=None,
    ):
        valid_depth = isinstance(max_depth, (int, np.integer)) and max_depth >= 1
        if max_depth is not None and not valid_depth:
            raise ValueError(f"max_depth must be a positive int or None, got {max_depth!r}")
        if not isinstance(min_samples_split, (int, np.integer)) or min_samples_split < 2:
            raise ValueError(f"min_samples_split must be an int >= 2, got {min_samples_split!r}")
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")
        if min_impurity_decrease < 0:
            raise ValueError(f"min_impurity_decrease must be >= 0, got {min_impurity_decrease}")
        if not isinstance(buffer_size, (int, np.integer)) or buffer_size < 1:
            raise ValueError(f"buffer_size must be a positive int, got {buffer_size!r}")
        _check_max_features(max_features)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.min_impurity_decrease = min_impurity_decrease
        self.buffer_size = buffer_size
        self.random_state = random_state
        self._clear_buffer()
        self.n_retrains_ = 0
        self._seen_classes = None

    def __repr__(self):
        return (
            f"DecisionTreeClassifier(max_depth={self.max_depth}, criterion={self.criterion!r}, "
            f"buffer_size={self.buffer_size})"
        )

    # ----- input checks -------------------------------------------------

    def _check_X(self, X):
        """Return X as a finite 2-D float array with the expected number of columns."""
        X = np.asarray(X)
        if X.dtype.kind not in "biuf":
            raise TypeError(f"X must be numeric, got dtype {X.dtype}; encode text columns first")
        X = X.astype(float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        expected = getattr(self, "n_features_in_", None)
        if expected is None and self._buffer_X:
            expected = self._buffer_X[0].shape[1]
        if expected is not None and X.shape[1] != expected:
            raise ValueError(f"X has {X.shape[1]} features, but the tree uses {expected}")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or inf; impute or clip it before the tree")
        return X

    def _check_Xy(self, X, y):
        """Check X and y together."""
        X = self._check_X(X)
        y = np.asarray(y)
        if y.ndim != 1 or y.size != X.shape[0]:
            raise ValueError(f"y must be 1-D with {X.shape[0]} values, got shape {y.shape}")
        if y.dtype.kind == "f" and np.isnan(y).any():
            raise ValueError("y contains NaN")
        return X, y

    # ----- training -----------------------------------------------------

    def _n_features_to_try(self, d):
        """How many columns to search at each node."""
        m = self.max_features
        if m is None:
            return d
        if m == "sqrt":
            return max(1, int(np.sqrt(d)))
        if m == "log2":
            return max(1, int(np.log2(d)))
        if isinstance(m, float):
            return max(1, int(m * d))
        if m > d:
            raise ValueError(f"max_features={m} is more than the {d} features")
        return m

    def _choose_columns(self, rng, d, n_try):
        """Columns to search at one node: all d, or a sorted random subset of n_try."""
        if n_try < d:
            return np.sort(rng.choice(d, size=n_try, replace=False))
        return np.arange(d)

    def _node_split(self, X, y_codes, rows, counts, n_classes, columns):
        """
        Best split of one node, or None if the node should stay a leaf.

        Returns:
            (column, threshold), with column an index into X; None if every
            searched column is constant or the impurity decrease is below
            min_impurity_decrease.
        """
        best, threshold, child_impurity = self._split(
            X[np.ix_(rows, columns)], y_codes[rows], n_classes, self.criterion
        )
        if best < 0:
            return None
        gain = float(_impurity(counts, rows.size, self.criterion)) - child_impurity
        if gain < self.min_impurity_decrease - GAIN_TOLERANCE:
            return None
        return int(columns[best]), threshold

    def _build(self, X, y, classes):
        """Train a new tree on (X, y) from scratch, replacing the old one."""
        y_codes = np.searchsorted(classes, y)
        k = classes.size
        n, d = X.shape
        n_try = self._n_features_to_try(d)
        rng = np.random.default_rng(self.random_state)
        max_depth = np.inf if self.max_depth is None else self.max_depth
        feature, threshold, left, right, value = [], [], [], [], []

        def add_leaf(rows):
            """Append a leaf holding these sample rows and return its node id."""
            feature.append(-1)
            threshold.append(np.nan)
            left.append(-1)
            right.append(-1)
            value.append(np.bincount(y_codes[rows], minlength=k))
            return len(feature) - 1

        all_rows = np.arange(n)
        stack = [(add_leaf(all_rows), all_rows, 0)]  # (node id, sample rows, depth)
        depth_reached = 0
        while stack:  # one iteration per node: a loop over nodes is needed
            node, rows, depth = stack.pop()
            depth_reached = max(depth_reached, depth)
            counts = value[node]
            too_deep = depth >= max_depth
            too_small = rows.size < self.min_samples_split
            if too_deep or too_small or counts.max() == rows.size:  # last test: pure node
                continue
            columns = self._choose_columns(rng, d, n_try)
            split = self._node_split(X, y_codes, rows, counts, k, columns)
            if split is None:
                continue
            feature[node], threshold[node] = split
            go_left = X[rows, feature[node]] <= threshold[node]
            left[node] = add_leaf(rows[go_left])
            right[node] = add_leaf(rows[~go_left])
            stack.append((left[node], rows[go_left], depth + 1))
            stack.append((right[node], rows[~go_left], depth + 1))

        self.feature_ = np.array(feature, dtype=int)
        self.threshold_ = np.array(threshold, dtype=float)
        self.left_ = np.array(left, dtype=int)
        self.right_ = np.array(right, dtype=int)
        self.value_ = np.array(value, dtype=float)
        self.depth_ = depth_reached
        self.classes_ = classes
        self.n_features_in_ = d
        self.n_samples_trained_ = n
        self.n_retrains_ += 1

    def fit(self, X, y):
        """
        Train the tree from scratch on X (n, d) and y (n,). Clears the buffer
        and the retrain count. Returns self.
        """
        X, y = self._check_Xy(X, y)
        if X.shape[0] == 0:
            raise ValueError("cannot fit on X with 0 samples")
        self._clear_buffer()
        self.n_retrains_ = 0
        self._seen_classes = np.unique(y)
        self._build(X, y, self._seen_classes)
        return self

    def _clear_buffer(self):
        """Empty the buffer of waiting chunks."""
        self._buffer_X = []
        self._buffer_y = []
        self._n_buffered = 0

    def partial_fit(self, X_chunk, y_chunk):
        """
        Add a chunk to the buffer and retrain once it holds >= buffer_size samples.

        Parameters:
            X_chunk : array_like, shape (n, d)
            y_chunk : array_like, shape (n,)

        Returns:
            self (check n_retrains_ to see whether this call retrained)

        Notes:
            A chunk bigger than buffer_size triggers one retrain on the whole
            buffer, not several. An empty chunk is ignored.
        """
        X, y = self._check_Xy(X_chunk, y_chunk)
        if X.shape[0] == 0:
            return self
        chunk_classes = np.unique(y)
        if self._seen_classes is None:
            self._seen_classes = chunk_classes
        else:
            self._seen_classes = np.union1d(self._seen_classes, chunk_classes)
        self._buffer_X.append(X)
        self._buffer_y.append(y)
        self._n_buffered += X.shape[0]
        if self._n_buffered >= self.buffer_size:
            self._retrain_on_buffer()
        return self

    def flush(self):
        """Train on the samples left in the buffer, e.g. at the end of a stream. Returns self."""
        if self._n_buffered > 0:
            self._retrain_on_buffer()
        return self

    def _retrain_on_buffer(self):
        """Train a new tree on every buffered sample, then empty the buffer."""
        X = np.concatenate(self._buffer_X)
        y = np.concatenate(self._buffer_y)
        self._clear_buffer()
        self._build(X, y, self._seen_classes)

    @property
    def n_buffered_(self):
        """Samples waiting in the buffer."""
        return self._n_buffered

    @property
    def n_nodes_(self):
        """Number of nodes (inner nodes and leaves) in the trained tree."""
        self._check_trained()
        return self.feature_.size

    @property
    def n_leaves_(self):
        """Number of leaves in the trained tree."""
        self._check_trained()
        return int(np.sum(self.feature_ < 0))

    @property
    def nbytes_(self):
        """Memory held by the tree arrays and the buffer, in bytes."""
        buffered = sum(a.nbytes for a in self._buffer_X) + sum(a.nbytes for a in self._buffer_y)
        if not hasattr(self, "feature_"):
            return buffered
        arrays = (self.feature_, self.threshold_, self.left_, self.right_, self.value_)
        return buffered + sum(a.nbytes for a in arrays)

    # ----- prediction ---------------------------------------------------

    def _check_trained(self):
        """Raise RuntimeError if no tree has been trained yet."""
        if not hasattr(self, "feature_"):
            raise RuntimeError(
                "the tree has not been trained yet: call fit, or keep calling partial_fit "
                f"until {self.buffer_size} samples are buffered (or call flush)"
            )

    def apply(self, X):
        """
        Return the leaf id reached by every row of X, shape (m,).

        All rows move down together: at each level, rows still at an inner
        node jump to its left or right child. The loop runs depth_ times.
        """
        self._check_trained()
        X = self._check_X(X)
        rows = np.arange(X.shape[0])
        node = np.zeros(X.shape[0], dtype=int)
        for _ in range(self.depth_):  # one step per level, not per sample
            feature = self.feature_[node]
            at_leaf = feature < 0
            go_left = X[rows, np.maximum(feature, 0)] <= self.threshold_[node]
            child = np.where(go_left, self.left_[node], self.right_[node])
            node = np.where(at_leaf, node, child)
        return node

    def predict_proba(self, X):
        """Class probabilities, shape (m, len(classes_)): class shares of each row's leaf."""
        leaves = self.apply(X)  # checks the tree is trained
        counts = self.value_[leaves]
        return counts / counts.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Predicted class of each row, shape (m,); ties go to the first class in classes_."""
        leaves = self.apply(X)
        return self.classes_[np.argmax(self.value_[leaves], axis=1)]

    def score(self, X, y):
        """Accuracy of predict(X) against y."""
        y = np.asarray(y)
        return float(np.mean(self.predict(X) == y))


def _check_max_features(max_features):
    """Raise ValueError for an invalid max_features value."""
    m = max_features
    valid = (
        m is None
        or m in ("sqrt", "log2")
        or (isinstance(m, (int, np.integer)) and not isinstance(m, bool) and m >= 1)
        or (isinstance(m, float) and 0 < m <= 1)
    )
    if not valid:
        raise ValueError(
            "max_features must be None, 'sqrt', 'log2', an int >= 1 or a float in (0, 1], "
            f"got {m!r}"
        )
