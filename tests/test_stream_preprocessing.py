"""Tests for partial_fit in numcompute_stream.preprocessing.

Run with:  python -m pytest tests/test_stream_preprocessing.py
"""

import numpy as np
import pytest

from numcompute_stream.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)


def _stream(transformer, X, n_chunks):
    """partial_fit the transformer on X split into n_chunks pieces."""
    for chunk in np.array_split(X, n_chunks):
        transformer.partial_fit(chunk)
    return transformer


def test_streamed_scalers_match_batch_fit():
    rng = np.random.default_rng(0)
    X = rng.normal(loc=50.0, scale=[1.0, 10.0, 0.1], size=(300, 3))
    X[rng.random(X.shape) < 0.05] = np.nan
    for make in (StandardScaler, lambda: MinMaxScaler((-1, 1))):
        streamed = _stream(make(), X, 7)
        batch = make().fit(X)
        assert np.allclose(streamed.transform(X), batch.transform(X), equal_nan=True)
    assert np.allclose(_stream(StandardScaler(), X, 7).mean_, np.nanmean(X, axis=0))


def test_constant_columns_do_not_divide_by_zero():
    X = np.array([[5.0, 1.0], [5.0, 2.0]])
    scaler = StandardScaler().partial_fit(X[:1])  # one row: every std is 0
    assert np.isfinite(scaler.transform(X)).all()
    scaler.partial_fit(X[1:])
    assert scaler.scale_[0] == 1.0  # still constant
    assert np.allclose(scaler.transform(X)[:, 1], [-1.0, 1.0])
    minmax = _stream(MinMaxScaler(), np.full((4, 1), 3.0), 2)
    assert np.isfinite(minmax.transform([[3.0], [4.0]])).all()


def test_unfitted_empty_chunks_and_shape_errors():
    for transformer in (StandardScaler(), MinMaxScaler(), SimpleImputer(), OneHotEncoder()):
        with pytest.raises(RuntimeError):
            transformer.transform(np.ones((2, 2)))
        transformer.partial_fit(np.ones((0, 2)))  # an empty chunk is ignored
        with pytest.raises(RuntimeError):
            transformer.transform(np.ones((2, 2)))
        transformer.partial_fit(np.ones((3, 2)))
        with pytest.raises(ValueError):
            transformer.partial_fit(np.ones((3, 4)))  # different number of columns
        with pytest.raises(ValueError):
            transformer.partial_fit(np.ones(3))  # 1-D


def test_imputer_updates_with_each_chunk():
    imputer = SimpleImputer(strategy="mean")
    imputer.partial_fit([[1.0, np.nan], [3.0, np.nan]])
    assert imputer.statistics_[0] == 2.0 and np.isnan(imputer.statistics_[1])
    imputer.partial_fit([[np.nan, 10.0], [8.0, 20.0]])
    assert imputer.statistics_.tolist() == [4.0, 15.0]
    assert imputer.transform([[np.nan, np.nan]]).tolist() == [[4.0, 15.0]]

    frequent = SimpleImputer(strategy="most_frequent", missing_values="")
    frequent.partial_fit(np.array([["S"], ["C"], [""]]))
    assert frequent.statistics_.tolist() == ["C"]  # tie -> smallest value
    frequent.partial_fit(np.array([["S"], ["Q"]]))
    assert frequent.statistics_.tolist() == ["S"]
    assert frequent.transform(np.array([[""]])).tolist() == [["S"]]


def test_streaming_median_is_close_to_exact():
    x = np.random.default_rng(1).exponential(scale=20.0, size=(2000, 1))
    imputer = _stream(SimpleImputer(strategy="median"), x, 10)
    bin_width = (x[:200].max() - x[:200].min()) / imputer.median_bins
    assert abs(imputer.statistics_[0] - np.median(x)) <= bin_width
    assert SimpleImputer(strategy="median").fit(x).statistics_[0] == np.median(x)  # exact


def test_one_hot_categories_grow_and_unknowns_are_zero():
    encoder = OneHotEncoder(handle_unknown="ignore")
    encoder.partial_fit(np.array([["b"], ["a"]]))
    assert encoder.n_output_features_ == 2
    assert encoder.transform(np.array([["c"]])).tolist() == [[0.0, 0.0]]  # unknown -> zeros
    encoder.partial_fit(np.array([["c"], ["a"]]))  # "c" first seen in a later chunk
    assert encoder.categories_[0].tolist() == ["a", "b", "c"]
    assert encoder.transform(np.array([["c"], ["a"]])).tolist() == [[0, 0, 1], [1, 0, 0]]

    strict = OneHotEncoder().partial_fit(np.array([["a"]]))
    with pytest.raises(ValueError):
        strict.transform(np.array([["z"]]))


def test_fixed_categories_keep_the_output_width():
    encoder = OneHotEncoder(categories=[[1, 2, 3]], handle_unknown="ignore")
    widths = []
    for chunk in ([[1], [1]], [[2]], [[3], [9]]):
        encoder.partial_fit(chunk)
        widths.append(encoder.transform(chunk).shape[1])
    assert widths == [3, 3, 3]
    assert encoder.transform([[9]]).tolist() == [[0, 0, 0]]
    with pytest.raises(ValueError):
        OneHotEncoder(categories=[[1, 2]]).partial_fit([[1, 2]])  # 2 columns, 1 category list
