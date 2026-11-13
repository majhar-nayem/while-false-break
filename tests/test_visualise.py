"""Tests for numcompute_stream.visualise. Run with:  python -m pytest tests/test_visualise.py"""

import matplotlib.pyplot as plt  # conftest.py selects the Agg backend (no windows)
import numpy as np
import pytest

from numcompute_stream.visualise import (
    compare_models,
    plot_metric_over_time,
    plot_predictions_vs_ground_truth,
)


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_metric_and_comparison_plots_draw_the_data():
    _, ax = plot_metric_over_time(
        [None, 0.5, 0.75, 0.8], "Accuracy", "acc", marks=[False, True, False, True]
    )
    line = ax.get_lines()[0]
    assert np.isnan(line.get_ydata()[0]) and line.get_ydata()[-1] == 0.8  # None -> gap
    assert len(ax.get_lines()) == 3  # the metric plus two retrain markers
    assert ax.get_title() == "Accuracy" and ax.get_ylabel() == "acc"

    _, ax = compare_models([0.6, 0.7], [0.5, 0.6, 0.9], labels=("small", "large"))
    assert [text.get_text() for text in ax.get_legend().get_texts()] == ["small", "large"]
    assert len(ax.get_lines()[1].get_xdata()) == 3


def test_predictions_plot_kinds():
    _, ax = plot_predictions_vs_ground_truth([0, 1, 1, 0], [0, 1, 0, 0])
    counts = [text.get_text() for text in ax.texts]
    assert counts == ["2", "0", "1", "1"]  # confusion matrix, rows = true
    _, ax = plot_predictions_vs_ground_truth(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
    assert len(ax.collections) == 1  # scatter for continuous values
    shared = plt.subplots(1, 2)[1]
    _, ax = plot_predictions_vs_ground_truth(["a", "b"], ["a", "a"], ax=shared[1])
    assert ax is shared[1]  # draws on the given axes


def test_save_to_file_and_input_errors(tmp_path):
    path = tmp_path / "accuracy.png"
    plot_metric_over_time([0.1, 0.2], save_path=path)
    assert path.exists() and path.stat().st_size > 0
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([0, 1, 1], [0, 1])
    with pytest.raises(ValueError):
        plot_metric_over_time([])
    with pytest.raises(TypeError):
        plot_metric_over_time(["high", "low"])
    with pytest.raises(ValueError):
        plot_metric_over_time([0.1, 0.2], marks=[True])
    with pytest.raises(ValueError):
        plot_predictions_vs_ground_truth([0], [0], kind="pie")
