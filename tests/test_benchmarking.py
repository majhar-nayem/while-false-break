"""Tests for numcompute.benchmarking. Run with:  python -m pytest tests/test_benchmarking.py"""

import numpy as np
import pytest

from numcompute.benchmarking import benchmark, compare, environment_info, format_table


def loop_sum(x):
    total = 0.0
    for v in x:
        total += v
    return total


def test_benchmark_result():
    r = benchmark(np.sum, np.ones(100), repeat=3, number=10)
    assert 0 < r["best"] <= r["median"]
    assert (r["repeat"], r["number"]) == (3, 10)
    assert benchmark(lambda: None, repeat=2, min_time=0.001)["number"] >= 1  # auto number
    with pytest.raises(ValueError):
        benchmark(np.sum, [1], repeat=0)


def test_compare_speedup_and_check():
    x = np.random.default_rng(0).normal(size=20_000)
    rows = compare({"loop": loop_sum, "numpy": np.sum}, x, repeat=3)
    assert rows[0]["speedup"] == 1.0
    assert rows[1]["speedup"] > 1  # vectorised should win
    with pytest.raises(ValueError):  # different answers
        compare({"a": np.sum, "b": np.mean}, x)
    with pytest.raises(ValueError):
        compare({})


def test_table_and_environment():
    rows = [
        {"name": "loop", "best": 2e-3, "median": 2.1e-3, "speedup": 1.0},
        {"name": "numpy", "best": 4e-6, "median": 5e-6, "speedup": 500.0},
    ]
    table = format_table(rows)
    assert table.splitlines()[0].startswith("| Implementation")
    assert "2 ms" in table and "4 µs" in table and "500.0x" in table
    env = environment_info()
    assert env["numpy"] == np.__version__ and set(env) == {
        "python",
        "numpy",
        "platform",
        "processor",
    }
