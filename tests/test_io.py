"""Tests for numcompute.io. Run with:  python test_io.py   (or: python -m pytest test_io.py)"""

import os
import tempfile

import numpy as np

from numcompute import io

SAMPLE = 'id,name,age,city\n1,"Smith, John",25,adl\n2,"Lee, Ann",,syd\n3,Bob,40,NA\n'


def make_file(text, name="sample.csv"):
    path = os.path.join(tempfile.mkdtemp(), name)
    with open(path, "w") as f:
        f.write(text)
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


def test_titanic():
    path = "data/titanic.csv"
    if not os.path.exists(path):
        print("  (skipped: put titanic.csv in the data folder)")
        return
    print("  columns:", io.read_header(path))
    X = io.load_csv(path, usecols=["Age", "Fare"])
    print("  Age/Fare shape:", X.shape, "| missing ages:", int(np.isnan(X[:, 0]).sum()))
    print("  rows after skip:", io.load_csv(path, usecols=["Age", "Fare"], missing="skip").shape[0])
    print("  first 3 names:", io.load_csv(path, usecols="Name", dtype=str)[:3, 0].tolist())
    print("  chunks of 200:", [len(c) for c in io.iter_csv_chunks(path, 200, usecols=["Age"])])


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("All tests passed.")
