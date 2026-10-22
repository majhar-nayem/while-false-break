"""CSV reader: custom delimiters, dtype handling, missing values and chunked streaming."""

import csv
import itertools

import numpy as np

MISSING = ["", "NA", "N/A", "NaN", "nan", "null", "None", "?"]


def read_header(path, delimiter=","):
    """Return the column names from the first line of a CSV file."""
    with open(path, encoding="utf-8", newline="") as f:
        line = f.readline()
    if not line.strip():
        return []
    return [name.strip() for name in next(csv.reader([line], delimiter=delimiter))]


def _setup(path, delimiter, header, usecols, missing):
    """Validate arguments and return (number of columns, selected column indices)."""
    if missing not in ("fill", "skip", "raise"):
        raise ValueError(f"missing must be 'fill', 'skip' or 'raise', got {missing!r}")
    first = read_header(path, delimiter)
    n_cols = len(first)
    if usecols is None:
        return n_cols, list(range(n_cols))
    if isinstance(usecols, (int, str)):
        usecols = [usecols]
    cols = []
    for c in usecols:
        if isinstance(c, str):
            if not header or c not in first:
                raise ValueError(f"column {c!r} not found")
            cols.append(first.index(c))
        elif 0 <= c < n_cols:
            cols.append(c)
        else:
            raise ValueError(f"column index {c} out of range for {n_cols} columns")
    return n_cols, cols


def _parse(lines, delimiter, n_cols, cols, dtype, missing, fill_value):
    """Convert text lines into a 2-D array and apply the missing-value policy."""
    rows = [r for r in csv.reader(lines, delimiter=delimiter) if r]
    for r in rows:
        if len(r) != n_cols:
            raise ValueError(f"could not parse row {r}: expected {n_cols} fields, got {len(r)}")
    is_text = np.dtype(dtype).kind == "U"
    if not rows:
        return np.empty((0, len(cols)), dtype=str if is_text else dtype)

    text = np.array([[r[c].strip() for c in cols] for r in rows], dtype=str)
    is_missing = np.isin(text, MISSING)

    if is_text:
        data = np.where(is_missing, "", text)
    else:
        # quoted fields are already split by csv, so re-join and let genfromtxt convert
        joined = [",".join(r) for r in np.where(is_missing, "nan", text)]
        data = np.genfromtxt(joined, delimiter=",", dtype=float, comments=None)
        data = data.reshape(text.shape)
        is_missing = np.isnan(data)

    if missing == "raise" and is_missing.any():
        r, c = np.argwhere(is_missing)[0]
        raise ValueError(f"missing value at data row {r}, column {c}")
    if missing == "skip":
        data = data[~is_missing.any(axis=1)]
    elif fill_value is not None:
        data = np.where(is_missing, fill_value, data)

    if is_text:
        return data
    if np.dtype(dtype).kind in "iu" and np.isnan(data).any():
        raise ValueError("integer dtype cannot hold missing values; use fill_value or missing='skip'")
    return data.astype(dtype)


def load_csv(path, delimiter=",", header=True, usecols=None, dtype=float,
             missing="fill", fill_value=None):
    """Load a delimited text file into a 2-D NumPy array.

    Parameters
    ----------
    path : str or Path
    delimiter : str, single character such as "," or "\\t"
    header : bool, whether the first line holds column names
    usecols : int, str or list, columns to load by index or name (default: all)
    dtype : float, int or str
    missing : "fill" (NaN, "" or fill_value), "skip" (drop rows) or "raise"
    fill_value : value used for missing cells when missing="fill"

    Returns
    -------
    ndarray of shape (n_rows, n_selected_cols), always 2-D.

    Raises
    ------
    FileNotFoundError if the file does not exist; ValueError for an unknown
    column, ragged rows, a bad policy, or missing values with missing="raise"
    or an integer dtype.

    Complexity: O(file size) time, O(n_rows * n_cols) memory.
    """
    n_cols, cols = _setup(path, delimiter, header, usecols, missing)
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.readlines()
    if header:
        lines = lines[1:]
    return _parse(lines, delimiter, n_cols, cols, dtype, missing, fill_value)


def iter_csv_chunks(path, chunksize=10000, delimiter=",", header=True, usecols=None,
                    dtype=float, missing="fill", fill_value=None):
    """Yield the file as 2-D arrays of at most `chunksize` rows.

    Arguments are the same as load_csv. Only one chunk is held in memory,
    so memory use is O(chunksize * n_cols) regardless of file size.
    """
    if not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError(f"chunksize must be a positive int, got {chunksize!r}")
    n_cols, cols = _setup(path, delimiter, header, usecols, missing)
    with open(path, encoding="utf-8", newline="") as f:
        if header:
            f.readline()
        while True:
            chunk = list(itertools.islice(f, chunksize))
            if not chunk:
                return
            yield _parse(chunk, delimiter, n_cols, cols, dtype, missing, fill_value)
