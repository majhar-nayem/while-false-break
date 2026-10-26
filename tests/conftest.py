"""Shared pytest fixtures for the NumCompute tests."""

from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parent.parent / "demo" / "data"


@pytest.fixture
def titanic_path():
    """Path to the Titanic CSV (train.csv) used in the demo; skips the test if it is missing."""
    path = DATA_DIR / "train.csv"
    if not path.exists():
        pytest.skip("demo/data/train.csv not found")
    return path
