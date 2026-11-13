"""Micro-benchmark harness: time functions and compare loops with vectorised code."""

import platform
import statistics
import sys
import time

import numpy as np

# Stop growing the number of calls per round once it reaches this many.
MAX_CALLS_PER_ROUND = 10**6


def _time_calls(func, args, n_calls):
    """Call func(*args) n_calls times and return the total time in seconds."""
    start = time.perf_counter()
    for _ in range(n_calls):
        func(*args)
    return time.perf_counter() - start


def _choose_number(func, args, min_time):
    """Find a number of calls (1, 10, 100, ...) that takes at least min_time seconds."""
    number = 1
    while _time_calls(func, args, number) < min_time and number < MAX_CALLS_PER_ROUND:
        number *= 10
    return number


def benchmark(func, *args, repeat=5, number=None, min_time=0.05):
    """
    Time a function call.

    Parameters:
        func : callable, called as func(*args)
        args : arguments passed to func
        repeat : int >= 1, number of timing rounds
        number : int >= 1, calls per round; if None it is picked so that one
            round takes at least min_time seconds (fast functions are run many
            times so the timer's resolution does not matter)
        min_time : float, target length of one round in seconds

    Returns:
        dict with "best" and "median" (seconds per call), "number" and "repeat"

    Raises:
        ValueError if repeat < 1 or number < 1.

    Notes:
        One warm-up call is made first and not timed. "best" is the most useful
        number: slower rounds are usually caused by other programs, not by func.

    Complexity:
        About repeat * max(min_time, number * cost of func).
    """
    if repeat < 1:
        raise ValueError("repeat and number must be >= 1")
    if number is not None and number < 1:
        raise ValueError("repeat and number must be >= 1")

    func(*args)  # warm-up call, not timed

    if number is None:
        number = _choose_number(func, args, min_time)

    times = []
    for _ in range(repeat):
        total_time = _time_calls(func, args, number)
        times.append(total_time / number)

    return {
        "best": min(times),
        "median": statistics.median(times),
        "number": number,
        "repeat": repeat,
    }


def _check_same_results(funcs, args):
    """Raise ValueError if any function gives a different result from the first one."""
    names = list(funcs)
    baseline_name = names[0]
    expected = np.asarray(funcs[baseline_name](*args), dtype=float)

    for name in names[1:]:
        result = np.asarray(funcs[name](*args), dtype=float)
        same_shape = result.shape == expected.shape
        if not same_shape or not np.allclose(result, expected, equal_nan=True):
            raise ValueError(f"{name!r} disagrees with {baseline_name!r}")


def compare(funcs, *args, repeat=5, check=True):
    """
    Time several implementations of the same thing and report the speed-up.

    Parameters:
        funcs : dict {name: callable}; the FIRST one is the baseline (e.g. the Python loop)
        args : arguments passed to every function
        repeat : int, timing rounds per function
        check : bool, first check that every function returns the same result

    Returns:
        list of dicts with "name", "best", "median" and "speedup"
        (speedup = baseline best time / this best time, so the baseline is 1.0)

    Raises:
        ValueError if funcs is empty, or check=True and a result differs from the baseline.

    Complexity:
        The sum of the benchmark costs.
    """
    if not funcs:
        raise ValueError("funcs must not be empty")
    if check:
        _check_same_results(funcs, args)

    rows = []
    for name, func in funcs.items():
        result = benchmark(func, *args, repeat=repeat)
        rows.append({"name": name, "best": result["best"], "median": result["median"]})

    baseline_time = rows[0]["best"]
    for row in rows:
        row["speedup"] = baseline_time / row["best"]
    return rows


def _format_time(seconds):
    """Format a time with a readable unit, e.g. 0.0023 -> '2.3 ms'."""
    units = [("s", 1), ("ms", 1e-3), ("µs", 1e-6)]
    for unit, scale in units:
        if seconds >= scale:
            return f"{seconds / scale:.3g} {unit}"
    return f"{seconds / 1e-9:.3g} ns"


def format_table(rows):
    """
    Format compare() results as a Markdown table.

    Parameters:
        rows : list of dicts from compare()

    Returns:
        str, a Markdown table with columns Implementation, Best, Median, Speed-up
    """
    lines = [
        "| Implementation | Best | Median | Speed-up |",
        "|---|---:|---:|---:|",
    ]
    for row in rows:
        best = _format_time(row["best"])
        median = _format_time(row["median"])
        lines.append(f"| {row['name']} | {best} | {median} | {row['speedup']:.1f}x |")
    return "\n".join(lines)


def environment_info():
    """
    Details that affect timings, to report next to benchmark results.

    Returns:
        dict with "python", "numpy", "platform" and "processor"
    """
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
