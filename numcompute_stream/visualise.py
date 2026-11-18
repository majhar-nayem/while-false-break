"""Reusable matplotlib plots for streaming metrics and predictions.

Every function:
    - draws on `ax` if given (so plots can share a figure), else on a new figure;
    - saves the figure to `save_path` if given (format from the extension);
    - calls plt.show() if show=True (in a notebook the returned figure also
      displays inline on its own);
    - returns (fig, ax).

Typical use with a StreamTrainer:
    plot_metric_over_time(trainer.history("cumulative_accuracy"),
                          "Cumulative accuracy", "Accuracy")
"""

import matplotlib.pyplot as plt
import numpy as np

COLORS = ("#1f6feb", "#d1495b", "#2a9d8f", "#e9a23b")


def _metric_array(values, name):
    """Return values as a 1-D float array (None becomes NaN, which leaves a gap)."""
    array = np.array([np.nan if v is None else v for v in np.ravel(values)], dtype=object)
    try:
        array = array.astype(float)
    except (TypeError, ValueError):
        raise TypeError(f"{name} must contain numbers") from None
    if np.ndim(values) != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence, got shape {np.shape(values)}")
    return array


def _axes(ax, figsize):
    """Return (fig, ax): the given axes and their figure, or a new figure."""
    if ax is not None:
        return ax.figure, ax
    return plt.subplots(figsize=figsize)


def _finish(fig, save_path, show):
    """Save and/or show the figure."""
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()


def plot_metric_over_time(
    metric_values,
    title="Metric over time",
    ylabel="Value",
    xlabel="Chunk",
    label=None,
    marks=None,
    ax=None,
    save_path=None,
    show=False,
):
    """
    Plot one metric across chunks as a line.

    Parameters:
        metric_values : sequence of numbers, shape (n_chunks,); None/NaN leave a gap
            (e.g. chunks scored before the model was first trained)
        title, ylabel, xlabel : str
        label : str, optional legend entry
        marks : sequence of bool, shape (n_chunks,), optional; chunks to mark
            with a dotted vertical line (e.g. the chunks where the tree retrained)
        ax : matplotlib Axes, optional
        save_path : str or path, optional
        show : bool

    Returns:
        (fig, ax)

    Raises:
        TypeError if the values are not numbers.
        ValueError if metric_values is empty or not 1-D, or marks has another length.
    """
    values = _metric_array(metric_values, "metric_values")
    fig, ax = _axes(ax, (7, 3.5))
    chunks = np.arange(values.size)
    ax.plot(chunks, values, marker="o", markersize=3, color=COLORS[0], label=label)
    if marks is not None:
        marks = np.asarray(marks, dtype=bool)
        if marks.shape != values.shape:
            raise ValueError(f"marks has {marks.size} values but there are {values.size} chunks")
        for chunk in chunks[marks]:
            ax.axvline(chunk, color="grey", linestyle=":", linewidth=1)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    if label is not None:
        ax.legend()
    _finish(fig, save_path, show)
    return fig, ax


def compare_models(
    metric1,
    metric2,
    labels=("Model 1", "Model 2"),
    title="Model comparison",
    ylabel="Value",
    xlabel="Chunk",
    ax=None,
    save_path=None,
    show=False,
):
    """
    Plot the same streaming metric for two models on one chart.

    Parameters:
        metric1, metric2 : sequences of numbers, one value per chunk (they may
            differ in length, e.g. different chunk sizes)
        labels : (str, str), legend entries, e.g. ("buffer_size=100", "buffer_size=500")
        title, ylabel, xlabel : str
        ax, save_path, show : as in plot_metric_over_time

    Returns:
        (fig, ax)

    Raises:
        TypeError / ValueError as plot_metric_over_time, or if labels is not two strings.
    """
    first = _metric_array(metric1, "metric1")
    second = _metric_array(metric2, "metric2")
    if len(labels) != 2:
        raise ValueError(f"labels must have 2 entries, got {len(labels)}")
    fig, ax = _axes(ax, (7, 3.5))
    for values, label, color in ((first, labels[0], COLORS[0]), (second, labels[1], COLORS[1])):
        ax.plot(np.arange(values.size), values, marker="o", markersize=3, color=color, label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    _finish(fig, save_path, show)
    return fig, ax


def plot_predictions_vs_ground_truth(
    y_true,
    y_pred,
    title="Predictions vs ground truth",
    kind="auto",
    labels=None,
    ax=None,
    save_path=None,
    show=False,
):
    """
    Show how predictions compare with the true values, e.g. on the latest chunk.

    Parameters:
        y_true, y_pred : array_like, shape (n,)
        title : str
        kind : "confusion" (class labels: a count matrix, rows = true,
            columns = predicted), "scatter" (numbers: predicted against true,
            with the y = x line) or "auto" (confusion for at most 20 distinct
            values or for text labels, otherwise scatter)
        labels : class order for "confusion" (default: sorted labels)
        ax, save_path, show : as in plot_metric_over_time

    Returns:
        (fig, ax)

    Raises:
        ValueError if the inputs are not 1-D, differ in length, are empty, or
            kind is unknown.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.ndim != 1 or y_pred.ndim != 1 or y_true.size != y_pred.size or y_true.size == 0:
        raise ValueError(
            f"y_true and y_pred must be non-empty, 1-D and the same length, "
            f"got {y_true.shape} and {y_pred.shape}"
        )
    if kind not in ("auto", "confusion", "scatter"):
        raise ValueError(f"kind must be 'auto', 'confusion' or 'scatter', got {kind!r}")
    if kind == "auto":
        is_text = y_true.dtype.kind not in "biuf"
        n_values = np.unique(np.concatenate([y_true, y_pred])).size
        kind = "confusion" if is_text or n_values <= 20 else "scatter"

    fig, ax = _axes(ax, (4.5, 4) if kind == "confusion" else (5, 4.5))
    if kind == "confusion":
        from numcompute_stream.metrics import confusion_matrix

        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        matrix = confusion_matrix(y_true, y_pred, labels=labels)
        ax.imshow(matrix, cmap="Blues")
        high = matrix.max() / 2 if matrix.max() > 0 else 1
        for (i, j), count in np.ndenumerate(matrix):
            ax.text(
                j,
                i,
                str(count),
                ha="center",
                va="center",
                color="white" if count > high else "black",
            )
        ticks = np.arange(len(labels))
        ax.set_xticks(ticks, [str(label) for label in labels])
        ax.set_yticks(ticks, [str(label) for label in labels])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        accuracy = float(np.mean(y_true == y_pred))
        ax.set_title(f"{title}\naccuracy {accuracy:.3f} on {y_true.size} samples")
    else:
        y_true = y_true.astype(float)
        y_pred = y_pred.astype(float)
        ax.scatter(y_true, y_pred, s=12, alpha=0.7, color=COLORS[0])
        low = np.nanmin([y_true.min(), y_pred.min()])
        high = np.nanmax([y_true.max(), y_pred.max()])
        ax.plot([low, high], [low, high], color="grey", linestyle="--", linewidth=1)
        ax.set_xlabel("True value")
        ax.set_ylabel("Predicted value")
        ax.set_title(title)
        ax.grid(alpha=0.3)
    _finish(fig, save_path, show)
    return fig, ax
