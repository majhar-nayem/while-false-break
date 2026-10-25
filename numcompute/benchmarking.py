"""Micro-benchmark harness for comparing implementations (e.g. loop vs NumPy).

Timing methodology (the same idea as :mod:`timeit`):

* ``time.perf_counter`` - monotonic, highest-resolution clock available.
* **Warm-up** runs are discarded (caches, lazy imports, allocator).
* **Auto-ranging** - each measurement repeats the call ``number`` times,
  with ``number`` grown until one measurement takes at least
  ``min_time``, so fast functions are not dominated by timer resolution.
* **Repeats** - several measurements; ``best`` (minimum) is the least noisy
  estimate of the true cost, ``median`` shows typical behaviour.
* Results are reported per call, and :func:`environment_info` records the
  machine so that tables are reproducible.
"""

from __future__ import annotations

import csv
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

__all__ = [
    "BenchmarkResult",
    "benchmark",
    "compare",
    "environment_info",
    "format_table",
    "save_results",
]


@dataclass
class BenchmarkResult:
    """Timing summary of one implementation (all times in seconds per call).

    Attributes
    ----------
    name : str
        Label of the implementation.
    best, median, mean, stdev : float
        Statistics over the repeats.
    repeat : int
        Number of measurements.
    number : int
        Calls per measurement.
    speedup : float or None
        ``baseline.best / self.best`` when produced by :func:`compare`.
    size : int or None
        Problem size label, if given.

    Notes
    -----
    A plain record; build it with :func:`benchmark` or :func:`compare`.
    Complexity: O(1) memory.
    """

    name: str
    best: float
    median: float
    mean: float
    stdev: float
    repeat: int
    number: int
    speedup: float | None = None
    size: int | None = None


def benchmark(
    func: Callable,
    *args,
    name: str | None = None,
    repeat: int = 5,
    number: int | None = None,
    warmup: int = 1,
    min_time: float = 0.05,
    size: int | None = None,
    **kwargs,
) -> BenchmarkResult:
    """Time ``func(*args, **kwargs)``.

    Parameters
    ----------
    func : callable
        Function under test.
    *args, **kwargs
        Arguments passed to ``func`` on every call.
    name : str, optional
        Label (defaults to ``func.__name__``).
    repeat : int, default=5
        Number of measurements.
    number : int, optional
        Calls per measurement; auto-ranged if omitted.
    warmup : int, default=1
        Untimed calls before measuring.
    min_time : float, default=0.05
        Target duration of one measurement when auto-ranging (seconds).
    size : int, optional
        Problem size, stored in the result for tables.

    Returns
    -------
    BenchmarkResult

    Raises
    ------
    ValueError
        If ``repeat < 1``, ``number < 1`` or ``warmup < 0``.

    Notes
    -----
    Complexity: ``warmup + repeat * number`` calls of ``func``.
    """
    if repeat < 1 or warmup < 0 or (number is not None and number < 1):
        raise ValueError("need repeat >= 1, warmup >= 0 and number >= 1")
    for _ in range(warmup):
        func(*args, **kwargs)
    if number is None:
        number = 1
        while True:
            t0 = time.perf_counter()
            for _ in range(number):
                func(*args, **kwargs)
            if time.perf_counter() - t0 >= min_time or number >= 1_000_000:
                break
            number *= 10
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        for _ in range(number):
            func(*args, **kwargs)
        times.append((time.perf_counter() - t0) / number)
    return BenchmarkResult(
        name=name or getattr(func, "__name__", "func"),
        best=min(times),
        median=statistics.median(times),
        mean=statistics.fmean(times),
        stdev=statistics.stdev(times) if len(times) > 1 else 0.0,
        repeat=repeat,
        number=number,
        size=size,
    )


def compare(
    implementations: dict[str, Callable],
    *args,
    check: bool = True,
    rtol: float = 1e-7,
    atol: float = 1e-9,
    size: int | None = None,
    **bench_kwargs,
) -> list[BenchmarkResult]:
    """Benchmark several implementations of the same computation.

    The first implementation is the baseline; each result's ``speedup`` is
    ``baseline.best / result.best`` (so > 1 means faster than the baseline).

    Parameters
    ----------
    implementations : dict of str -> callable
        Ordered mapping; put the reference (e.g. the Python loop) first.
    *args
        Arguments passed to every implementation.
    check : bool, default=True
        Verify that all implementations return the same values as the
        baseline (``np.allclose``) before timing, so a fast but wrong
        version cannot win.
    rtol, atol : float
        Tolerances for ``check``.
    size : int, optional
        Problem size label.
    **bench_kwargs
        Forwarded to :func:`benchmark` (``repeat``, ``number``, ...).

    Returns
    -------
    list of BenchmarkResult

    Raises
    ------
    ValueError
        If ``implementations`` is empty or outputs disagree.

    Notes
    -----
    Complexity: one :func:`benchmark` per implementation.
    """
    if not implementations:
        raise ValueError("implementations must not be empty")
    items = list(implementations.items())
    if check:
        ref = np.asarray(items[0][1](*args), dtype=np.float64)
        for name, fn in items[1:]:
            got = np.asarray(fn(*args), dtype=np.float64)
            if got.shape != ref.shape or not np.allclose(
                got, ref, rtol=rtol, atol=atol, equal_nan=True
            ):
                raise ValueError(f"{name!r} disagrees with baseline {items[0][0]!r}")
    results = [
        benchmark(fn, *args, name=name, size=size, **bench_kwargs) for name, fn in items
    ]
    base = results[0].best
    for r in results:
        r.speedup = base / r.best if r.best > 0 else float("inf")
    return results


def environment_info() -> dict:
    """Describe the machine and library versions for reproducible reports.

    Returns
    -------
    dict
        ``python``, ``numpy``, ``platform``, ``machine``, ``processor``,
        ``cpu_count`` and ``blas`` (best effort).

    Notes
    -----
    Complexity: O(1) (reads interpreter and platform metadata).
    """
    blas = "unknown"
    try:
        cfg = np.show_config(mode="dicts")
        blas = cfg.get("Build Dependencies", {}).get("blas", {}).get("name", "unknown")
    except Exception:  # noqa: BLE001 - older numpy without mode="dicts"
        pass
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "blas": blas,
    }


def _fmt_time(t: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6)):
        if t >= scale:
            return f"{t / scale:.3g} {unit}"
    return f"{t / 1e-9:.3g} ns"


def format_table(results: Sequence[BenchmarkResult]) -> str:
    """Render results as a Markdown table (paste into README/report).

    Parameters
    ----------
    results : sequence of BenchmarkResult

    Returns
    -------
    str
        Markdown table with name, size, best, median and speed-up.

    Notes
    -----
    Complexity: O(r) for r results.
    """
    lines = [
        "| Implementation | n | Best | Median | Speed-up |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in results:
        sp = "—" if r.speedup is None else f"{r.speedup:.1f}×"
        n = "" if r.size is None else f"{r.size:,}"
        lines.append(
            f"| {r.name} | {n} | {_fmt_time(r.best)} | {_fmt_time(r.median)} | {sp} |"
        )
    return "\n".join(lines)


def save_results(results: Sequence[BenchmarkResult], path: str | os.PathLike) -> None:
    """Write results to CSV (one row per result, plus environment columns).

    Parameters
    ----------
    results : sequence of BenchmarkResult
    path : str or PathLike
        Output file; parent directories must exist.

    Returns
    -------
    None

    Notes
    -----
    Complexity: O(r) for r results.
    """
    env = environment_info()
    with open(path, "w", newline="") as fh:
        rows = [
            {**asdict(r), **{f"env_{k}": v for k, v in env.items()}} for r in results
        ]
        if not rows:
            return
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
