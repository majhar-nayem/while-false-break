"""Tests for numcompute.benchmarking."""

import numpy as np
import pytest

from numcompute.benchmarking import (
    BenchmarkResult,
    benchmark,
    compare,
    environment_info,
    format_table,
    save_results,
)


def loop_sum(x):
    total = 0.0
    for v in x:
        total += v
    return total


def test_benchmark_fields():
    r = benchmark(np.sum, np.ones(100), repeat=3, number=10, size=100)
    assert isinstance(r, BenchmarkResult)
    assert 0 < r.best <= r.median
    assert (r.repeat, r.number, r.size) == (3, 10, 100)


def test_autorange_picks_number():
    r = benchmark(lambda: None, repeat=2, min_time=0.001)
    assert r.number >= 1


def test_compare_speedup_and_check():
    x = np.random.default_rng(0).normal(size=20_000)
    res = compare({"loop": loop_sum, "numpy": np.sum}, x, repeat=3, number=3)
    assert res[0].speedup == pytest.approx(1.0)
    assert res[1].speedup > 1  # vectorised should win
    with pytest.raises(ValueError, match="disagrees"):
        compare({"a": np.sum, "b": np.mean}, x, number=1, repeat=1)


def test_table_env_and_csv(tmp_path):
    res = compare({"a": np.sum, "b": np.sum}, np.ones(10), number=1, repeat=1, size=10)
    table = format_table(res)
    assert table.count("\n") == 3 and "| a |" in table
    env = environment_info()
    assert env["numpy"] == np.__version__
    out = tmp_path / "r.csv"
    save_results(res, out)
    assert out.read_text().startswith("name,")


def test_bad_arguments():
    with pytest.raises(ValueError):
        benchmark(np.sum, [1], repeat=0)
    with pytest.raises(ValueError):
        compare({})


def test_time_units_and_empty_csv(tmp_path):
    rows = [
        BenchmarkResult(f"t{i}", t, t, t, 0.0, 1, 1)
        for i, t in enumerate([2.0, 2e-3, 2e-6, 2e-9])
    ]
    table = format_table(rows)
    for unit in ("2 s", "2 ms", "2 µs", "2 ns"):
        assert unit in table
    out = tmp_path / "empty.csv"
    save_results([], out)
    assert out.read_text() == ""


def test_environment_info_without_blas_details(monkeypatch):
    def old_show_config(*args, **kwargs):
        raise TypeError("mode= not supported")

    monkeypatch.setattr(np, "show_config", old_show_config)
    assert environment_info()["blas"] == "unknown"
