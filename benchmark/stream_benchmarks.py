"""Streaming benchmarks for numcompute_stream (Assignment 2.2).

Run from the repository root:
    python benchmark/stream_benchmarks.py

Part A times pure-Python loops against the vectorised streaming code on the
same chunked stream; every pair is checked to give the same answer first.
Part B streams data through DecisionTreeClassifier with different buffer_size
values and records accuracy (test-then-train), retrains, retraining time and
memory. Writes benchmark/stream_results.md and two charts in benchmark/figures/.
All data uses fixed seeds, so the inputs are identical on every run.
"""

import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write the charts to files, no window

import matplotlib.pyplot as plt
import numpy as np

from numcompute_stream import io
from numcompute_stream.benchmarking import compare, environment_info, format_table
from numcompute_stream.metrics import StreamingConfusionMatrix
from numcompute_stream.pipeline import ColumnSelector, FeatureUnion, Pipeline
from numcompute_stream.preprocessing import (
    OneHotEncoder,
    SimpleImputer,
    StandardScaler,
)
from numcompute_stream.stats import StreamingStats
from numcompute_stream.stream import StreamTrainer
from numcompute_stream.tree import (
    DecisionTreeClassifier,
    best_split,
    best_split_loop,
)
from numcompute_stream.visualise import compare_models

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "benchmark" / "figures"
TITANIC = ROOT / "demo" / "data" / "train.csv"


# ---- data streams ---------------------------------------------------------


def sea_stream(n=20000, chunk_size=200, drift_every=5000, noise=0.1, seed=0):
    """
    SEA concepts stream (Street and Kim, 2001), a standard drifting benchmark.

    Three features uniform in [0, 10]; the label is x0 + x1 <= theta, and
    theta changes every drift_every samples (8, 9, 7, 9.5), so the concept
    drifts three times. A fraction `noise` of the labels is flipped.
    Returns a list of (X_chunk, y_chunk).
    """
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, 3))
    theta = np.array([8.0, 9.0, 7.0, 9.5])[(np.arange(n) // drift_every) % 4]
    y = (X[:, 0] + X[:, 1] <= theta).astype(int)
    flip = rng.random(n) < noise
    y[flip] = 1 - y[flip]
    return [(X[i : i + chunk_size], y[i : i + chunk_size]) for i in range(0, n, chunk_size)]


NUM_COLS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CAT_COLS = ["Sex", "Embarked"]


def titanic_stream(chunk_size=50):
    """Titanic read with io.iter_csv_chunks: (raw rows with 5 number + 2 text columns, Survived)."""
    numbers = io.iter_csv_chunks(TITANIC, chunksize=chunk_size, usecols=NUM_COLS)
    texts = io.iter_csv_chunks(TITANIC, chunksize=chunk_size, usecols=CAT_COLS, dtype=str)
    labels = io.iter_csv_chunks(TITANIC, chunksize=chunk_size, usecols="Survived", dtype=int)
    return [
        (np.hstack([num.astype(object), text.astype(object)]), label[:, 0])
        for num, text, label in zip(numbers, texts, labels)
    ]


def titanic_pipeline(buffer_size):
    """Impute + scale the numbers, impute + one-hot the text (fixed categories), then the tree."""
    numeric = Pipeline([
        ("select", ColumnSelector([0, 1, 2, 3, 4], dtype=float)),
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ])
    text = Pipeline([
        ("select", ColumnSelector([5, 6], dtype=str)),
        ("impute", SimpleImputer(strategy="most_frequent", missing_values="")),
        ("onehot", OneHotEncoder(
            categories=[["female", "male"], ["C", "Q", "S"]], handle_unknown="ignore"
        )),
    ])
    tree = DecisionTreeClassifier(max_depth=4, buffer_size=buffer_size, random_state=0)
    return Pipeline([("features", FeatureUnion([("num", numeric), ("text", text)])),
                     ("tree", tree)])


# ---- pure-Python versions -------------------------------------------------


def loop_running_stats(chunks):
    """Welford's one-value-at-a-time update for each column, with plain Python."""
    d = len(chunks[0][0])
    count = [0] * d
    mean = [0.0] * d
    m2 = [0.0] * d
    for chunk in chunks:
        for row in chunk:
            for j, x in enumerate(row):
                count[j] += 1
                delta = x - mean[j]
                mean[j] += delta / count[j]
                m2[j] += delta * (x - mean[j])
    return mean + [m2[j] / count[j] for j in range(d)]


def vectorised_running_stats(chunks):
    """StreamingStats.update_stats, one call per chunk."""
    stats = StreamingStats()
    for chunk in chunks:
        stats.update_stats(chunk)
    return np.concatenate([stats.mean, stats.var()])


def loop_confusion(pairs, k):
    """Accumulate a k x k confusion matrix pair by pair."""
    matrix = [[0] * k for _ in range(k)]
    for y_true, y_pred in pairs:
        for t, p in zip(y_true, y_pred):
            matrix[t][p] += 1
    return matrix


def vectorised_confusion(pairs, k):
    """StreamingConfusionMatrix.update, one call per chunk."""
    matrix = StreamingConfusionMatrix(labels=list(range(k)))
    for y_true, y_pred in pairs:
        matrix.update(y_true, y_pred)
    return matrix.result()


def loop_predict(tree, X):
    """Walk each row down the tree on its own."""
    feature = tree.feature_.tolist()
    threshold = tree.threshold_.tolist()
    left = tree.left_.tolist()
    right = tree.right_.tolist()
    winner = np.argmax(tree.value_, axis=1).tolist()
    out = []
    for row in X.tolist():
        node = 0
        while feature[node] >= 0:
            node = left[node] if row[feature[node]] <= threshold[node] else right[node]
        out.append(winner[node])
    return tree.classes_[out]


class LoopSplitTree(DecisionTreeClassifier):
    """The same tree, but every node's split is searched with Python loops."""

    _split = staticmethod(best_split_loop)


def run_tree_stream(tree_class, chunks):
    """Stream every chunk into a tree (buffer 1000) and return its final predictions."""
    tree = tree_class(max_depth=6, buffer_size=1000, random_state=0)
    for X, y in chunks:
        tree.partial_fit(X, y)
    return tree.predict(chunks[-1][0])


def part_a():
    """Loop vs vectorised on streaming workloads; returns Markdown."""
    rng = np.random.default_rng(42)
    sea = sea_stream()
    stats_chunks = [rng.normal(5.0, 2.0, size=(500, 5)) for _ in range(20)]
    stats_lists = [chunk.tolist() for chunk in stats_chunks]
    pairs = [(rng.integers(0, 3, 500), rng.integers(0, 3, 500)) for _ in range(40)]
    pair_lists = [(t.tolist(), p.tolist()) for t, p in pairs]
    big_pairs = [(rng.integers(0, 3, 5000), rng.integers(0, 3, 5000)) for _ in range(40)]
    big_pair_lists = [(t.tolist(), p.tolist()) for t, p in big_pairs]
    X_buffer = np.concatenate([X for X, _ in sea[:10]])  # one full buffer: 2,000 x 3
    y_buffer = np.concatenate([y for _, y in sea[:10]])
    tree = DecisionTreeClassifier(max_depth=8).fit(X_buffer, y_buffer)
    X_all = np.concatenate([X for X, _ in sea])

    cases = [
        (
            "split search, one buffer (2,000 x 3)",
            {
                "loop": lambda: best_split_loop(X_buffer, y_buffer, 2),
                "vectorised": lambda: best_split(X_buffer, y_buffer, 2),
            },
        ),
        (
            "tree on SEA stream (100 chunks, buffer 1,000)",
            {
                "loop splits": lambda: run_tree_stream(LoopSplitTree, sea),
                "vectorised splits": lambda: run_tree_stream(DecisionTreeClassifier, sea),
            },
        ),
        (
            "predict 20,000 rows (depth 8)",
            {
                "loop": lambda: loop_predict(tree, X_all),
                "vectorised": lambda: tree.predict(X_all),
            },
        ),
        (
            "running mean/var, 20 chunks x 500 x 5",
            {
                "loop (Welford)": lambda: loop_running_stats(stats_lists),
                "vectorised (Chan merge)": lambda: vectorised_running_stats(stats_chunks),
            },
        ),
        (
            "confusion matrix, 40 chunks x 500",
            {
                "loop": lambda: loop_confusion(pair_lists, 3),
                "vectorised": lambda: vectorised_confusion(pairs, 3),
            },
        ),
        (
            "confusion matrix, 40 chunks x 5,000",
            {
                "loop": lambda: loop_confusion(big_pair_lists, 3),
                "vectorised": lambda: vectorised_confusion(big_pairs, 3),
            },
        ),
    ]
    lines = []
    for title, funcs in cases:
        rows = compare(funcs, repeat=5)
        for row in rows:
            row["name"] = f"{title}: {row['name']}"
        lines.append(format_table(rows).split("\n", 2)[2] if lines else format_table(rows))
        print(f"{title}: {rows[-1]['speedup']:.1f}x")
    return "\n".join(lines)


# ---- part B: buffer_size sweep ---------------------------------------------


def sweep(make_model, chunks, buffer_sizes, rounds=3):
    """
    Stream the chunks through a fresh model for each buffer size.

    Accuracy is test-then-train (each chunk scored before it is trained on).
    Times are the best of `rounds` runs. Returns a list of result dicts and
    the trainer of each buffer size (for plotting).
    """
    results = []
    trainers = {}
    for size in buffer_sizes:
        best_total = np.inf
        for _ in range(rounds):
            trainer = StreamTrainer(make_model(size), window=5).run(chunks)
            retrained = trainer.history("retrained") == 1
            total = float(trainer.history("fit_time")[retrained].sum())
            if total < best_total:
                best_total, best = total, trainer
        retrained = best.history("retrained") == 1
        n_retrains = int(retrained.sum())
        results.append({
            "buffer_size": size,
            "accuracy": best.log_[-1]["cumulative_accuracy"],
            "n_retrains": n_retrains,
            "first_scored": int(np.argmax(~np.isnan(best.history("accuracy")))),
            "total_retrain": best_total,
            "mean_retrain": best_total / n_retrains,
            "peak_memory": int(best.history("memory_bytes").max()),
        })
        trainers[size] = best
    return results, trainers


def sweep_table(results):
    lines = [
        (
            "| buffer_size | Accuracy (test-then-train) | Retrains | First chunk scored "
            "| Total retrain time | Mean per retrain | Peak memory |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['buffer_size']} | {r['accuracy']:.3f} | {r['n_retrains']} | {r['first_scored']} "
            f"| {r['total_retrain'] * 1e3:.1f} ms | {r['mean_retrain'] * 1e3:.2f} ms "
            f"| {r['peak_memory'] / 1024:.0f} KiB |"
        )
    return "\n".join(lines)


def sweep_figure(results, path):
    """Accuracy and mean retrain time against buffer size, on two y-axes."""
    sizes = [r["buffer_size"] for r in results]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(sizes, [r["accuracy"] for r in results], marker="o", color="#1f6feb")
    ax.set_xscale("log")
    ax.minorticks_off()  # only label the tested sizes
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set_xlabel("buffer_size")
    ax.set_ylabel("Test-then-train accuracy", color="#1f6feb")
    ax.grid(alpha=0.3)
    other = ax.twinx()
    other.plot(sizes, [r["mean_retrain"] * 1e3 for r in results], marker="s", color="#d1495b")
    other.set_ylabel("Mean retrain time (ms)", color="#d1495b")
    ax.set_title("SEA stream: accuracy and retrain cost against buffer size")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    print("Part A: loop vs vectorised")
    table_a = part_a()

    print("Part B: buffer_size sweep")
    sea = sea_stream()
    sea_results, sea_trainers = sweep(
        lambda size: DecisionTreeClassifier(max_depth=6, buffer_size=size, random_state=0),
        sea,
        [200, 500, 1000, 2000, 5000],
    )
    titanic_results, _ = sweep(titanic_pipeline, titanic_stream(), [50, 100, 200, 400])
    sweep_figure(sea_results, FIGURES / "buffer_sweep.png")
    compare_models(
        sea_trainers[200].history("rolling_accuracy"),
        sea_trainers[2000].history("rolling_accuracy"),
        labels=("buffer_size=200", "buffer_size=2000"),
        title="SEA stream: rolling accuracy (last 5 chunks); drift every 25 chunks",
        ylabel="Accuracy",
        save_path=FIGURES / "sea_rolling_accuracy.png",
    )
    plt.close("all")

    env = environment_info()
    report = "\n".join([
        "# Streaming benchmarks (Assignment 2.2)",
        "",
        "Generated by `python benchmark/stream_benchmarks.py`.",
        "",
        "## A. Loop vs vectorised (best of 5 rounds; results checked equal first)",
        "",
        table_a,
        "",
        "## B. buffer_size sweep",
        "",
        (
            "Accuracy is test-then-train: each chunk is scored before the model trains on it, "
            "so the first chunks (before the first retrain) are not scored. Times are the best "
            "of 3 runs. Peak memory is what the model holds between chunks (buffer, tree "
            "arrays and running statistics)."
        ),
        "",
        (
            "### SEA stream (20,000 samples, chunks of 200, concept drift every 5,000, "
            "10% label noise; tree max_depth=6)"
        ),
        "",
        sweep_table(sea_results),
        "",
        (
            "### Titanic (891 passengers read with io.iter_csv_chunks, chunks of 50; "
            "impute + scale + one-hot pipeline, tree max_depth=4)"
        ),
        "",
        sweep_table(titanic_results),
        "",
        f"Python {env['python']}, NumPy {env['numpy']}, {env['platform']}, {env['processor']}",
        "",
    ])
    (ROOT / "benchmark" / "stream_results.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"done in {time.perf_counter() - start:.0f} s")


if __name__ == "__main__":
    main()
