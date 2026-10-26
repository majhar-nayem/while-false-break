"""Tests for numcompute.io. Run with:  python -m pytest tests/test_io.py"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from numcompute import io

SAMPLE = 'id,name,age,city\n1,"Smith, John",25,adl\n2,"Lee, Ann",,syd\n3,Bob,40,NA\n'


def make_file(text, name="sample.csv"):
    """Write text to a new temporary file and return its path."""
    path = Path(tempfile.mkdtemp()) / name
    path.write_text(text)
    return path


def test_header():
    assert io.read_header(make_file(SAMPLE)) == ["id", "name", "age", "city"]


def test_numeric_and_missing_fill():
    X = io.load_csv(make_file(SAMPLE), usecols=["id", "age"])
    assert X.shape == (3, 2)
    assert np.isnan(X[1, 1])


def test_quoted_text_column():
    names = io.load_csv(make_file(SAMPLE), usecols="name", dtype=str)
    assert names[:, 0].tolist() == ["Smith, John", "Lee, Ann", "Bob"]


def test_skip_fill_raise():
    p = make_file(SAMPLE)
    assert io.load_csv(p, usecols=["age"], missing="skip").shape == (2, 1)
    assert io.load_csv(p, usecols=["age"], fill_value=-1)[1, 0] == -1
    try:
        io.load_csv(p, usecols=["age"], missing="raise")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_int_dtype_and_tab_delimiter():
    p = make_file("a\tb\n1\t2\n3\t4\n", "data.tsv")
    X = io.load_csv(p, delimiter="\t", dtype=int)
    assert X.dtype.kind == "i" and X.tolist() == [[1, 2], [3, 4]]


def test_chunks():
    p = make_file("a,b\n" + "".join(f"{i},{i * 2}\n" for i in range(23)))
    chunks = list(io.iter_csv_chunks(p, chunksize=5))
    assert [len(c) for c in chunks] == [5, 5, 5, 5, 3]
    assert np.vstack(chunks).shape == (23, 2)


def test_quoted_delimiter_does_not_shift_columns():
    # "Smith, John" contains the delimiter; the number columns must still line up.
    X = io.load_csv(make_file(SAMPLE), usecols=["id", "age"])
    assert X[0].tolist() == [1.0, 25.0]
    p = make_file('a,b\n"1,5",2\n')
    with pytest.raises(ValueError, match="'1,5'"):  # clear error, not a reshape error
        io.load_csv(p)


def test_non_numeric_text_raises():
    p = make_file("a,b\n1,2\nabc,4\n")
    with pytest.raises(ValueError, match="'abc'.*data row 1, column 0"):
        io.load_csv(p)  # must not silently become NaN
    with pytest.raises(ValueError, match="'Smith, John'.*column 1"):
        io.load_csv(make_file(SAMPLE))  # a text column read as numbers
    # Missing markers and literal "nan" are still treated as missing.
    X = io.load_csv(make_file("a,b\nNA,nan\n1,\n"))
    assert np.isnan(X).sum() == 3


def test_titanic(titanic_path):
    assert io.read_header(titanic_path)[:3] == ["PassengerId", "Survived", "Pclass"]

    X = io.load_csv(titanic_path, usecols=["Age", "Fare"])
    assert X.shape[1] == 2
    assert np.isnan(X[:, 0]).any()  # some ages are missing
    assert not np.isnan(X[:, 1]).any()

    X_complete = io.load_csv(titanic_path, usecols=["Age", "Fare"], missing="skip")
    assert X_complete.shape[0] == X.shape[0] - np.isnan(X[:, 0]).sum()

    with pytest.raises(ValueError, match="use dtype=str"):  # Name/Sex columns are text
        io.load_csv(titanic_path)

    names = io.load_csv(titanic_path, usecols="Name", dtype=str)
    assert names[0, 0] == "Braund, Mr. Owen Harris"  # quoted field with a comma

    chunks = list(io.iter_csv_chunks(titanic_path, 200, usecols=["Age"]))
    assert all(len(chunk) <= 200 for chunk in chunks)
    assert np.array_equal(np.vstack(chunks), X[:, [0]], equal_nan=True)
