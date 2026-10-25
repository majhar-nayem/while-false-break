.PHONY: install test cov lint fmt bench demo report check

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

cov:
	python -m pytest --cov=numcompute --cov-report=term-missing

lint:
	python -m ruff check .

fmt:
	python -m ruff format .
	python -m ruff check --fix .

bench:
	python benchmark/run_benchmarks.py

demo:
	python -m jupyter nbconvert --to notebook --execute --inplace demo/quickstart.ipynb

# Re-run the notebook first (it regenerates report/figures/), then print to PDF.
CHROME ?= /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
report:
	"$(CHROME)" --headless=new --disable-gpu --no-pdf-header-footer \
		--allow-file-access-from-files --print-to-pdf=report/report.pdf \
		"file://$(CURDIR)/report/report.html"

# Run this before pushing.
check: lint test
