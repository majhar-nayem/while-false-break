"""Benchmark pure-Python loops against the vectorised NumCompute functions.

Run from the repository root:
    python benchmark/run_benchmarks.py

Prints a table and writes it to benchmark/results.md. Every pair is checked
to give the same answer before it is timed. The random data uses a fixed
seed, so the inputs are identical on every run.
"""

import heapq
import math
from pathlib import Path

import numpy as np

from numcompute import rank, sort_search, stats, utils
from numcompute.benchmarking import compare, environment_info, format_table
from numcompute.preprocessing import StandardScaler

# ---- pure-Python versions -------------------------------------------------


def loop_std(values):
    """Population standard deviation with plain Python."""
    mean = sum(values) / len(values)
    return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))


def loop_histogram(values, bins=20):
    """Equal-width histogram counts with plain Python."""
    low, high = min(values), max(values)
    counts = [0] * bins
    for v in values:
        bin_index = int((v - low) / (high - low) * bins)
        counts[min(bin_index, bins - 1)] += 1  # the maximum goes in the last bin
    return counts


def loop_standardize(rows):
    """Standardise each column of a list of rows with plain Python."""
    columns = list(zip(*rows))
    means = [sum(column) / len(column) for column in columns]
    stds = []
    for column, mean in zip(columns, means):
        stds.append(math.sqrt(sum((v - mean) ** 2 for v in column) / len(column)))
    return [[(v - m) / s for v, m, s in zip(row, means, stds)] for row in rows]


def loop_distances(points):
    """All pairwise Euclidean distances with plain Python."""
    return [[math.dist(a, b) for b in points] for a in points]


def loop_topk(values, k=10):
    """The k largest values, using the standard library heap."""
    return heapq.nlargest(k, values)


def loop_median(values):
    """Median of an odd-length list by sorting it."""
    return sorted(values)[len(values) // 2]


def loop_rank(values):
    """Average ranks (ties share the mean of their positions) with plain Python."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        # Find the last position j that has the same value as position i.
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


# ---- benchmark cases ------------------------------------------------------


def make_cases(n):
    """Build the (title, {name: function}) pairs to time; the loop version comes first."""
    rng = np.random.default_rng(42)  # fixed seed, so every run uses the same data
    values = rng.normal(size=n)
    values_list = values.tolist()
    table = rng.normal(size=(n // 10, 5))
    table_list = table.tolist()
    points = rng.normal(size=(200, 8))
    points_list = points.tolist()
    ints = rng.integers(0, 1000, size=n).astype(float)  # many ties
    ints_list = ints.tolist()
    odd = values[: n - 1]  # odd length, so the median is one element
    odd_list = odd.tolist()

    return [
        (
            "std",
            {
                "loop": lambda: loop_std(values_list),
                "numcompute": lambda: stats.std(values),
            },
        ),
        (
            "histogram (20 bins)",
            {
                "loop": lambda: loop_histogram(values_list),
                "numcompute": lambda: stats.histogram(values, 20)[0],
            },
        ),
        (
            "StandardScaler",
            {
                "loop": lambda: loop_standardize(table_list),
                "numcompute": lambda: StandardScaler().fit_transform(table),
            },
        ),
        (
            "distances (200 x 8)",
            {
                "loop": lambda: loop_distances(points_list),
                "numcompute": lambda: utils.euclidean_distances(points),
            },
        ),
        (
            "top-10",
            {
                "loop (heapq)": lambda: loop_topk(values_list),
                "numcompute": lambda: sort_search.topk(values, 10, return_indices=False),
            },
        ),
        (
            "median",
            {
                "loop (sorted)": lambda: loop_median(odd_list),
                "quickselect": lambda: sort_search.quickselect(odd, len(odd) // 2),
            },
        ),
        (
            "rank (average, ties)",
            {
                "loop": lambda: loop_rank(ints_list),
                "numcompute": lambda: rank.rank(ints),
            },
        ),
    ]


def main(n=50_000):
    """Run every case, print the results table and save it to results.md."""
    rows = []
    for title, funcs in make_cases(n):
        for row in compare(funcs, repeat=5):
            row["name"] = f"{title}: {row['name']}"
            rows.append(row)

    env = environment_info()
    table = format_table(rows)
    text = (
        f"# Benchmarks (n = {n:,})\n\n{table}\n\n"
        f"Python {env['python']}, NumPy {env['numpy']}, {env['platform']}, {env['processor']}\n"
    )
    print(text)
    out_path = Path(__file__).resolve().parent / "results.md"
    out_path.write_text(text, encoding="utf-8")
    print(f"saved to {out_path}")


if __name__ == "__main__":
    main()
