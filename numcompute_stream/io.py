"""CSV reader: custom delimiters, dtype handling, missing values and chunked streaming."""

import csv
import itertools

import numpy as np

# Cell contents that count as a missing value.
MISSING = ["", "NA", "N/A", "NaN", "nan", "null", "None", "?"]

MISSING_POLICIES = ("fill", "skip", "raise")


def read_header(path, delimiter=","):
    """
    Return the column names from the first line of a CSV file.

    Parameters:
        path : str or Path
        delimiter : str, single character such as "," or "\\t"

    Returns:
        list of str, the stripped column names ([] for an empty file)

    Raises:
        FileNotFoundError if the file does not exist.

    Complexity:
        O(length of the first line).
    """
    with open(path, encoding="utf-8", newline="") as file:
        first_line = file.readline()
    if not first_line.strip():
        return []
    names = next(csv.reader([first_line], delimiter=delimiter))
    return [name.strip() for name in names]


def _column_index(column, header_names, header, n_cols):
    """Turn one usecols entry (a name or an index) into a column index."""
    if isinstance(column, str):
        if not header or column not in header_names:
            raise ValueError(f"column {column!r} not found")
        return header_names.index(column)
    if 0 <= column < n_cols:
        return column
    raise ValueError(f"column index {column} out of range for {n_cols} columns")


def _setup(path, delimiter, header, usecols, missing):
    """Validate arguments and return (number of columns, selected column indices)."""
    if missing not in MISSING_POLICIES:
        raise ValueError(f"missing must be 'fill', 'skip' or 'raise', got {missing!r}")

    header_names = read_header(path, delimiter)
    n_cols = len(header_names)

    if usecols is None:
        return n_cols, list(range(n_cols))
    if isinstance(usecols, (int, str)):
        usecols = [usecols]

    cols = [_column_index(column, header_names, header, n_cols) for column in usecols]
    return n_cols, cols


def _split_rows(lines, delimiter, n_cols):
    """Split text lines into lists of fields, skipping blank lines."""
    rows = []
    for row in csv.reader(lines, delimiter=delimiter):
        if not row:
            continue
        if len(row) != n_cols:
            raise ValueError(f"could not parse row {row}: expected {n_cols} fields, got {len(row)}")
        rows.append(row)
    return rows


def _first_bad_cell(text):
    """Return (row, col, value) of the first cell in text that is not a number."""
    for row, col in np.ndindex(text.shape):  # only runs after conversion has failed
        try:
            float(text[row, col])
        except ValueError:
            return row, col, text[row, col]
    return None


def _text_to_float(text, is_missing, cols):
    """
    Convert a 2-D array of strings to floats, with NaN for missing cells.

    The csv module has already split the fields (including quoted fields that
    contain the delimiter), so each cell is converted on its own. A cell that
    is not a number raises ValueError instead of silently becoming NaN.
    """
    text = np.where(is_missing, "nan", text)
    try:
        return text.astype(float)
    except ValueError:
        row, col, value = _first_bad_cell(text)
        raise ValueError(
            f"could not convert {str(value)!r} to a number at data row {row}, "
            f"column {cols[col]}; use dtype=str to load text columns"
        ) from None


def _parse(lines, delimiter, n_cols, cols, dtype, missing, fill_value):
    """Convert text lines into a 2-D array and apply the missing-value policy."""
    rows = _split_rows(lines, delimiter, n_cols)
    is_text = np.dtype(dtype).kind == "U"
    if not rows:
        empty_dtype = str if is_text else dtype
        return np.empty((0, len(cols)), dtype=empty_dtype)

    selected = [[row[c].strip() for c in cols] for row in rows]
    text = np.array(selected, dtype=str)
    is_missing = np.isin(text, MISSING)

    if is_text:
        data = np.where(is_missing, "", text)
    else:
        data = _text_to_float(text, is_missing, cols)
        is_missing = np.isnan(data)  # also catches cells such as "nan" in the file

    if missing == "raise" and is_missing.any():
        row, col = np.argwhere(is_missing)[0]
        raise ValueError(f"missing value at data row {row}, column {col}")
    if missing == "skip":
        complete_rows = ~is_missing.any(axis=1)
        data = data[complete_rows]
    elif fill_value is not None:
        data = np.where(is_missing, fill_value, data)

    if is_text:
        return data
    if np.dtype(dtype).kind in "iu" and np.isnan(data).any():
        raise ValueError(
            "integer dtype cannot hold missing values; use fill_value or missing='skip'"
        )
    return data.astype(dtype)


def load_csv(
    path, delimiter=",", header=True, usecols=None, dtype=float, missing="fill", fill_value=None
):
    """
    Load a delimited text file into a 2-D NumPy array.

    Parameters:
        path : str or Path
        delimiter : str, single character such as "," or "\\t"
        header : bool, whether the first line holds column names
        usecols : int, str or list, columns to load by index or name (default: all)
        dtype : float, int or str
        missing : "fill" (NaN, "" or fill_value), "skip" (drop rows) or "raise"
        fill_value : value used for missing cells when missing="fill"

    Returns:
        ndarray of shape (n_rows, n_selected_cols), always 2-D

    Raises:
        FileNotFoundError if the file does not exist.
        ValueError for an unknown column, ragged rows, a bad policy, a cell
            that is not a number when dtype is numeric, or missing values with
            missing="raise" or an integer dtype.

    Complexity:
        O(file size) time, O(n_rows * n_cols) memory.
    """
    n_cols, cols = _setup(path, delimiter, header, usecols, missing)
    with open(path, encoding="utf-8", newline="") as file:
        lines = file.readlines()
    if header:
        lines = lines[1:]
    return _parse(lines, delimiter, n_cols, cols, dtype, missing, fill_value)


def iter_csv_chunks(
    path,
    chunksize=10000,
    delimiter=",",
    header=True,
    usecols=None,
    dtype=float,
    missing="fill",
    fill_value=None,
):
    """
    Yield the file as 2-D arrays of at most chunksize rows.

    Parameters:
        chunksize : int >= 1, maximum number of rows per chunk
        other parameters : the same as load_csv

    Yields:
        ndarray of shape (<= chunksize, n_selected_cols)

    Raises:
        ValueError if chunksize is not a positive int, plus the errors of load_csv.

    Complexity:
        O(file size) time in total. Only one chunk is held in memory, so memory
        use is O(chunksize * n_cols) regardless of file size.
    """
    if not isinstance(chunksize, int) or chunksize < 1:
        raise ValueError(f"chunksize must be a positive int, got {chunksize!r}")
    n_cols, cols = _setup(path, delimiter, header, usecols, missing)

    with open(path, encoding="utf-8", newline="") as file:
        if header:
            file.readline()  # skip the header line
        while True:
            chunk = list(itertools.islice(file, chunksize))
            if not chunk:
                return
            yield _parse(chunk, delimiter, n_cols, cols, dtype, missing, fill_value)
