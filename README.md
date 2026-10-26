# NumCompute

A small scientific computing toolkit written in plain Python and NumPy, built for
**Assignment 2.1: Programming Task 1**.

NumCompute provides reusable, vectorised building blocks for a simple machine
learning workflow: reading CSV files, preprocessing, sorting and ranking,
descriptive statistics, evaluation metrics, finite-difference gradients, a
pipeline API for chaining steps, and a benchmark harness that compares our
vectorised code against plain Python loops.

No external ML or data libraries (scikit-learn, pandas, PyTorch, matplotlib) are
used. The only dependency is NumPy.

**Team:** While-False-Break

| Member | Student ID | Modules | Tests | Other work |
|---|---|---|---|---|
| Shadid Chowdhury | a3232268 | `io`, `preprocessing`, `sort_search`, `rank` | `test_io`, `test_preprocessing`, `test_sort_search`, `test_rank` | benchmark script, README, final integration and bug fixes |
| Member Name 2 | a0000000 | `stats`, `optim`, `metrics` | `test_stats`, `test_optim`, `test_metrics` | Packaging (`pyproject.toml`) |
| Member Name 3 | a0000000 | `utils`, `pipeline`, `benchmarking` | `test_utils`, `test_pipeline`, `test_benchmarking` | Demo notebook (`demo/quickstart.ipynb`, `svgplot.py`) |

---

## Contents

1. [Installation](#installation)
2. [How to run](#how-to-run)
3. [Project structure](#project-structure)
4. [Quick examples](#quick-examples)
5. [API overview](#api-overview)
6. [Design choices](#design-choices)
7. [Testing](#testing)
8. [Performance](#performance)
9. [Sample output from the demo](#sample-output-from-the-demo)
10. [Use of generative AI](#use-of-generative-ai)

---

## Installation

You need Python 3.10 or newer.

```bash
# from the project root
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install -e ".[dev]"
```

This installs `numcompute` in editable mode, plus `pytest` and `jupyter` for the
tests and the demo. The package itself only needs `numpy>=1.24`.

## How to run

| Task | Command |
|---|---|
| Run the unit tests | `python -m pytest` |
| Run the benchmarks | `python benchmark/run_benchmarks.py` |
| Re-run the demo notebook | `jupyter nbconvert --to notebook --execute --inplace demo/quickstart.ipynb` |
| Open the demo interactively | `jupyter notebook demo/quickstart.ipynb` |
| Check code style | `python -m ruff check .` |

The benchmark script prints a table and also saves it to `benchmark/results.md`.

## Project structure

```
while-false-break/
├── numcompute/              the package
│   ├── __init__.py
│   ├── io.py                CSV reader: column selection, dtypes, missing values, chunked streaming
│   ├── preprocessing.py     StandardScaler, MinMaxScaler, SimpleImputer, OneHotEncoder
│   ├── sort_search.py       stable sort, top-k, quickselect, binary search
│   ├── rank.py              ranking with ties (average / dense / ordinal), percentiles
│   ├── stats.py             mean, var, std, min, max, quantiles, histogram, RunningStats (Welford)
│   ├── metrics.py           accuracy, precision, recall, F1, confusion matrix, MSE, ROC curve, AUC
│   ├── optim.py             finite-difference gradient and Jacobian
│   ├── pipeline.py          Estimator base class, Pipeline, Compose, FeatureUnion, ColumnSelector
│   ├── utils.py             distances, sigmoid, relu, logsumexp, softmax, argtopk, batching
│   └── benchmarking.py      timing harness, comparison tables, environment info
├── tests/                   69 unit tests (pytest), one file per module
├── demo/
│   ├── quickstart.ipynb     end-to-end demo on the Titanic dataset
│   ├── svgplot.py           tiny SVG chart helper (so the demo needs no matplotlib)
│   └── data/train.csv       Titanic passenger list from Kaggle
├── benchmark/
│   ├── run_benchmarks.py    loops vs vectorised benchmarks
│   └── results.md           latest benchmark table
└── pyproject.toml           packaging configuration
```

## Quick examples

```python
import numpy as np
from numcompute import io, preprocessing as pp, rank, stats, utils, optim

# Read a CSV; missing numeric cells become NaN
X = io.load_csv("demo/data/train.csv", usecols=["Age", "Fare"])

# Fill missing values, then scale
X = pp.SimpleImputer(strategy="median").fit_transform(X)
Z = pp.StandardScaler().fit_transform(X)

# Statistics that can skip NaN
stats.mean([1.0, np.nan, 3.0], ignore_nan=True)        # 2.0

# Ranking with ties
rank.rank([10, 20, 20, 30], method="average")          # [1.0, 2.5, 2.5, 4.0]

# Numerically stable softmax (no overflow on large inputs)
utils.softmax(np.array([1000.0, 1001.0, 1002.0]))       # [0.090, 0.245, 0.665]

# Finite-difference gradient of f(x, y) = x^2 + 3y
optim.grad(lambda v: v[0] ** 2 + 3 * v[1], np.array([2.0, 1.0]))   # ~[4.0, 3.0]
```

A pipeline chains preprocessing steps and a model:

```python
from numcompute.pipeline import Pipeline

model = Pipeline([
    ("impute", pp.SimpleImputer(strategy="median")),
    ("scale", pp.StandardScaler()),
    ("clf", MyClassifier()),          # any object with fit(X, y) and predict(X)
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## API overview

Every function has a docstring listing its parameters, return value and shape,
the exceptions it raises, and its time/space complexity. A short summary:

| Module | Main functions / classes |
|---|---|
| `io` | `read_header`, `load_csv`, `iter_csv_chunks` |
| `preprocessing` | `StandardScaler`, `MinMaxScaler`, `SimpleImputer`, `OneHotEncoder` |
| `sort_search` | `stable_sort`, `topk`, `quickselect`, `binary_search` |
| `rank` | `rank`, `percentile` |
| `stats` | `mean`, `var`, `std`, `amin`, `amax`, `quantile`, `median`, `histogram`, `RunningStats` |
| `metrics` | `accuracy`, `precision`, `recall`, `f1`, `confusion_matrix`, `mse`, `roc_curve`, `auc` |
| `optim` | `grad`, `jacobian` |
| `pipeline` | `Estimator`, `Pipeline`, `Compose`, `FeatureUnion`, `ColumnSelector` |
| `utils` | `euclidean_distances`, `manhattan_distances`, `sigmoid`, `relu`, `logsumexp`, `softmax`, `argtopk`, `batch_iterator` |
| `benchmarking` | `benchmark`, `compare`, `format_table`, `environment_info` |

## Design choices

**Vectorisation.** Core computations use NumPy array operations instead of Python
loops. For example, `euclidean_distances` uses broadcasting to compute all pairs
at once, and `topk` uses `np.argpartition` so it never sorts the whole array.
Loops are only used where they are naturally needed, such as the pivot steps in
`quickselect` or iterating over chunks of a file.

**One consistent API.** Every preprocessing class follows the same
`fit` / `transform` / `fit_transform` pattern, and models follow `fit` / `predict`.
The pipeline relies on these method names only (duck typing), so any object with
the right methods can be used as a step, including the demo's own classifier.
Reductions in `stats` take an `axis` argument that works like NumPy's
(`axis=0` gives one value per column).

**Numerical stability.**
- `softmax` and `logsumexp` subtract the maximum first, so `exp` never overflows.
- `RunningStats` uses Welford's method, which updates mean and variance one batch
  at a time without the cancellation error of the "sum of squares" formula.
- `optim.grad` defaults to central differences (error O(h²)) with `h = 1e-5`,
  a balance between truncation error and round-off error. The demo shows this
  trade-off in a chart.

**Missing values and ties.**
- `load_csv` can fill, skip or raise on missing cells (`missing="fill" | "skip" | "raise"`).
- `stats` functions propagate NaN by default, like NumPy, and skip it with
  `ignore_nan=True`. `rank` gives NaN a rank of NaN and ranks the other values
  among themselves.
- `rank` supports three tie methods (`average`, `dense`, `ordinal`). `topk` breaks
  ties in favour of the earlier position, so results are repeatable.

**Clear errors.** Inputs are checked at the start of each function. Wrong types
raise `TypeError` and wrong shapes or values raise `ValueError`, with a message
saying what was expected (for example `k must be in [0, 9], got 12`).

**Only NumPy for the demo too.** Because matplotlib is not allowed, the demo
draws its charts with `demo/svgplot.py`, a small helper that writes SVG text
directly.

## Testing

```bash
python -m pytest
```

There are **69 tests** across 10 files (one per module). Current result:
`69 passed`. Besides normal cases, the tests cover the edge cases asked for in
the brief:

- empty arrays and single-element arrays
- all-equal values (for example a column with zero standard deviation)
- duplicates and ties in sorting, top-k and ranking
- extreme `k` (0, n − 1, and out-of-range values)
- NaN handling, both propagated and ignored
- non-contiguous arrays (strided slices such as `x[::2]`)
- wrong types and shapes raising the right exception

## Performance

`benchmark/run_benchmarks.py` times each vectorised function against a plain
Python version on the same data (n = 50,000, fixed random seed). Each pair is
checked to give the same answer before it is timed, and the best of several
rounds is reported.

| Implementation | Best | Median | Speed-up |
|---|---:|---:|---:|
| std: loop | 5.41 ms | 5.47 ms | 1.0x |
| std: numcompute | 426 µs | 463 µs | 12.7x |
| histogram (20 bins): loop | 7.88 ms | 8.19 ms | 1.0x |
| histogram (20 bins): numcompute | 1.64 ms | 1.65 ms | 4.8x |
| StandardScaler: loop | 6.06 ms | 6.2 ms | 1.0x |
| StandardScaler: numcompute | 448 µs | 454 µs | 13.5x |
| distances (200 x 8): loop | 5.93 ms | 5.98 ms | 1.0x |
| distances (200 x 8): numcompute | 181 µs | 184 µs | 32.7x |
| top-10: loop (heapq) | 643 µs | 681 µs | 1.0x |
| top-10: numcompute | 468 µs | 478 µs | 1.4x |
| median: loop (sorted) | 5.37 ms | 5.39 ms | 1.0x |
| median: quickselect | 1.01 ms | 1.02 ms | 5.3x |
| rank (average, ties): loop | 20.1 ms | 20.2 ms | 1.0x |
| rank (average, ties): numcompute | 4.06 ms | 4.25 ms | 4.9x |

Environment: Python 3.14.7, NumPy 2.5.4, Windows 11, Intel CPU.

The biggest gain is for pairwise distances (about 33x), where broadcasting
replaces a double loop. Top-10 gains the least because Python's `heapq` is
already written in C. Exact timings depend on the machine, but the ordering of
the results stays the same when re-run.

## Sample output from the demo

The notebook `demo/quickstart.ipynb` walks through every module using the
Titanic passenger list (891 rows, 177 missing ages, 2 missing boarding ports).
A few excerpts:

**Reading a CSV with missing values (`io`)**
```
numeric (891, 5) | text (891, 2) | labels (891,)
missing per numeric column: {'Pclass': 0, 'Age': 177, 'SibSp': 0, 'Parch': 0, 'Fare': 0}
skip  -> (714, 5) (incomplete rows dropped)
chunk sizes: [250, 250, 250, 141]
```

**Ranking with ties (`rank`)**
```
Pclass:  [3. 1. 3. 1. 3. 3.]
 average: [4.5 1.5 4.5 1.5 4.5 4.5]
   dense: [2. 1. 2. 1. 2. 2.]
 ordinal: [3. 1. 4. 2. 5. 6.]
```

**Streaming statistics match the one-pass result (`stats`)**
```
streaming mean: [ 2.309 29.699  0.523  0.382 32.204]
matches one-pass mean: True | std: True
```

**Gradient of the Rosenbrock function (`optim`)**
```
analytic: [-126.999   46.62 ]
central : [-126.999   46.62 ]
forward : [-126.992   46.621]
```

**Pipeline and metrics (`pipeline`, `metrics`)** on a held-out 20% test set,
using a simple nearest-centroid classifier:
```
                  accuracy  precision  recall     F1
nearest centroid     0.793      0.750   0.696  0.722
   women survive     0.816      0.790   0.710  0.748

confusion matrix (rows = true, columns = predicted):
 [[94 16]
 [21 48]]
ROC AUC: 0.85
```

## Use of generative AI

This assignment allows **Level 3: AI-assisted ideas**. Our team used generative
AI in the following ways:

- **Documentation:** help with wording and structuring the README and the
  function docstrings.
- **Getting unstuck:** when we were stuck while writing code, we asked for ideas
  on how to approach the problem, then wrote and tested the implementation
  ourselves.
- **Clean code:** suggestions on how to write clearer, cleaner code (naming,
  structure, input checks and error messages).

All code was written, tested and verified by the team, and we take responsibility
for every part of the submission.