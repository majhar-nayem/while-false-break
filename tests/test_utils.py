"""Tests for numcompute.utils."""

import numpy as np
import pytest

from numcompute import utils
from numcompute.utils import (
    NotFittedError,
    argtopk,
    batch_iterator,
    check_array,
    cosine_similarity,
    euclidean_distances,
    log_softmax,
    logsumexp,
    manhattan_distances,
    relu,
    sigmoid,
    softmax,
)


class TestCheckArray:
    def test_rejects_strings(self):
        with pytest.raises(TypeError, match="numeric"):
            check_array(["a", "b"])

    def test_object_numbers_become_float(self):
        out = check_array(np.array([1, 2.5], dtype=object))
        assert out.dtype == np.float64

    def test_ndim_message_suggests_reshape(self):
        with pytest.raises(ValueError, match="reshape"):
            check_array([1.0, 2.0], ndim=2)

    def test_empty_and_nan_flags(self):
        with pytest.raises(ValueError, match="empty"):
            check_array([], allow_empty=False)
        with pytest.raises(ValueError, match="NaN"):
            check_array([1.0, np.nan], allow_nan=False)

    def test_not_fitted_is_runtime_error(self):
        assert issubclass(NotFittedError, RuntimeError)


class TestDistances:
    def test_euclidean_matches_brute_force(self, rng):
        A, B = rng.normal(size=(6, 3)), rng.normal(size=(4, 3))
        brute = np.sqrt(((A[:, None] - B[None]) ** 2).sum(-1))
        np.testing.assert_allclose(euclidean_distances(A, B), brute, atol=1e-12)

    def test_euclidean_self_has_exact_zero_diagonal(self, rng):
        A = rng.normal(size=(5, 3)) * 1e6  # large values -> round-off risk
        D = euclidean_distances(A)
        assert np.all(np.diag(D) == 0)
        assert np.all(D >= 0)

    def test_manhattan(self):
        D = manhattan_distances([[0, 0], [1, 1]], [[1, 2]])
        np.testing.assert_array_equal(D, [[3], [1]])

    def test_cosine_zero_vector_is_zero_not_nan(self):
        S = cosine_similarity([[0.0, 0.0], [1.0, 0.0]], [[1.0, 0.0]])
        np.testing.assert_allclose(S, [[0.0], [1.0]])

    def test_feature_mismatch(self):
        with pytest.raises(ValueError, match="same number of features"):
            euclidean_distances(np.ones((2, 3)), np.ones((2, 4)))

    def test_non_contiguous(self, non_contiguous):
        D = euclidean_distances(non_contiguous)
        D2 = euclidean_distances(np.ascontiguousarray(non_contiguous))
        np.testing.assert_allclose(D, D2)


class TestActivations:
    def test_softmax_extreme_logits_do_not_overflow(self):
        p = softmax([1000.0, 1000.0, -1000.0])
        np.testing.assert_allclose(p, [0.5, 0.5, 0.0])

    def test_softmax_rows_sum_to_one(self, rng):
        p = softmax(rng.normal(size=(4, 7)) * 50, axis=1)
        np.testing.assert_allclose(p.sum(axis=1), 1.0)

    def test_softmax_shift_invariance(self):
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(softmax(x), softmax(x + 1e4))

    def test_log_softmax_stable(self):
        out = log_softmax([0.0, -1e4])
        np.testing.assert_allclose(out, [0.0, -1e4])

    def test_logsumexp_matches_naive_on_safe_input(self, rng):
        x = rng.normal(size=(3, 5))
        np.testing.assert_allclose(logsumexp(x, axis=1), np.log(np.exp(x).sum(axis=1)))

    def test_logsumexp_large_and_infinite(self):
        assert logsumexp([1000.0, 1000.0]) == pytest.approx(1000.0 + np.log(2))
        assert logsumexp([-np.inf, -np.inf]) == -np.inf
        assert logsumexp([np.inf, 1.0]) == np.inf

    def test_logsumexp_keepdims_and_empty(self):
        assert logsumexp(np.zeros((2, 3)), axis=1, keepdims=True).shape == (2, 1)
        with pytest.raises(ValueError):
            logsumexp([])

    def test_sigmoid_extremes(self):
        out = sigmoid([-1000.0, 0.0, 1000.0])
        np.testing.assert_allclose(out, [0.0, 0.5, 1.0])

    def test_relu(self):
        np.testing.assert_array_equal(relu([-1.0, 0.0, 2.0]), [0.0, 0.0, 2.0])


class TestTopKAndBatching:
    def test_argtopk_rows(self):
        x = np.array([[1, 5, 3], [9, 2, 7]])
        np.testing.assert_array_equal(argtopk(x, 2, axis=1), [[1, 2], [0, 2]])
        np.testing.assert_array_equal(argtopk(x, 1, axis=1, largest=False), [[0], [1]])

    def test_argtopk_k_bounds(self):
        assert argtopk([1, 2], 0).shape == (0,)
        with pytest.raises(ValueError):
            argtopk([1, 2], 3)

    def test_batches_cover_all_rows_in_order(self):
        X, y = np.arange(10).reshape(5, 2), np.arange(5)
        batches = list(batch_iterator(X, y, batch_size=2))
        assert [len(b[1]) for b in batches] == [2, 2, 1]
        np.testing.assert_array_equal(np.concatenate([b[1] for b in batches]), y)

    def test_batches_shuffle_reproducible_and_drop_last(self):
        a = list(batch_iterator(np.arange(10), batch_size=3, shuffle=True, seed=0))
        b = list(batch_iterator(np.arange(10), batch_size=3, shuffle=True, seed=0))
        assert all(np.array_equal(p, q) for p, q in zip(a, b, strict=True))
        dropped = list(batch_iterator(np.arange(10), batch_size=3, drop_last=True))
        assert len(dropped) == 3

    def test_batches_length_mismatch(self):
        with pytest.raises(ValueError, match="inconsistent"):
            list(batch_iterator(np.arange(3), np.arange(4), batch_size=2))


def test_public_api_listed():
    for name in utils.__all__:
        assert hasattr(utils, name)