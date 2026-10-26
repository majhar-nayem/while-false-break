"""Small SVG chart helper for the demo, using only NumPy and the standard library.

The brief allows only Python and NumPy, so instead of matplotlib the charts
are written directly as SVG text. Each function returns a Figure, which
Jupyter shows inline and which can be saved with fig.save("name.svg").
"""

from html import escape
from pathlib import Path

import numpy as np

COLORS = ["#2a6fdb", "#e4572e", "#29a36a", "#8e5cd9"]
W, H = 640, 360  # image size in pixels
LEFT, RIGHT, TOP, BOTTOM = 70, 30, 45, 60


class Figure:
    """An SVG image. Jupyter calls _repr_svg_ to display it."""

    def __init__(self, svg):
        self.svg = svg

    def _repr_svg_(self):
        return self.svg

    def save(self, path):
        Path(path).write_text(self.svg, encoding="utf-8")
        return self


def _text(x, y, s, size=12, anchor="middle", rotate=False, bold=False):
    rot = f' transform="rotate(-90 {x} {y})"' if rotate else ""
    weight = ' font-weight="bold"' if bold else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}"{weight}{rot}>{escape(str(s))}</text>'


def _ticks(lo, hi, log):
    """Tick values: powers of 10 on a log axis, otherwise 'nice' steps (1, 2 or 5 x 10^k)."""
    if log:
        ticks = 10.0 ** np.arange(np.floor(np.log10(lo)), np.ceil(np.log10(hi)) + 1)
        ticks = ticks[(ticks >= lo) & (ticks <= hi)]
        return ticks[:: max(1, len(ticks) // 7)]
    step = 10 ** np.floor(np.log10((hi - lo) / 5))
    for m in (1, 2, 5, 10):
        if (hi - lo) / (step * m) <= 6:
            step *= m
            break
    return np.arange(np.ceil(lo / step) * step, hi + step * 1e-9, step)


def _scale(lo, hi, log, p0, p1):
    """Return a function that maps data values to pixel positions."""
    if log:
        lo, hi = np.log10(lo), np.log10(hi)
    return lambda v: p0 + ((np.log10(v) if log else v) - lo) / (hi - lo) * (p1 - p0)


def _fmt(v, log):
    return f"{v:.0e}" if log else f"{round(v, 10) + 0:g}"  # + 0 turns -0.0 into 0


def _frame(title, xlabel, ylabel, ylim, logy, xlim=None, logx=False):
    """Background, title, axis labels, y grid (and x ticks if xlim is given)."""
    sy = _scale(*ylim, logy, H - BOTTOM, TOP)
    parts = [
        f'<rect width="{W}" height="{H}" fill="white"/>',
        _text(W / 2, 25, title, 15, bold=True),
    ]
    for t in _ticks(*ylim, logy):
        y = sy(t)
        parts.append(
            f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y:.1f}" y2="{y:.1f}" stroke="#e5e5e5"/>'
        )
        parts.append(_text(LEFT - 8, y + 4, _fmt(t, logy), anchor="end"))
    sx = None
    if xlim is not None:
        sx = _scale(*xlim, logx, LEFT, W - RIGHT)
        for t in _ticks(*xlim, logx):
            parts.append(_text(sx(t), H - BOTTOM + 18, _fmt(t, logx)))
    parts.append(
        f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{H - BOTTOM}" y2="{H - BOTTOM}" stroke="#444"/>'
    )
    parts.append(_text(W / 2, H - 15, xlabel))
    parts.append(_text(18, (TOP + H - BOTTOM) / 2, ylabel, rotate=True))
    return parts, sx, sy


def _svg(parts):
    body = "".join(parts)
    return Figure(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'viewBox="0 0 {W} {H}" font-family="sans-serif" fill="#222">{body}</svg>'
    )


def bar_chart(labels, values, title="", xlabel="", ylabel=""):
    """
    Vertical bar chart.

    Parameters:
        labels : list of str, one per bar
        values : list or array of numbers >= 0, one per bar
        title, xlabel, ylabel : str

    Returns:
        Figure
    """
    values = np.asarray(values, dtype=float)
    parts, _, sy = _frame(title, xlabel, ylabel, (0, values.max() * 1.1 or 1), False)
    slot = (W - LEFT - RIGHT) / len(values)
    for i, (label, v) in enumerate(zip(labels, values)):
        x = LEFT + i * slot
        parts.append(
            f'<rect x="{x + slot * 0.1:.1f}" y="{sy(v):.1f}" width="{slot * 0.8:.1f}" '
            f'height="{sy(0) - sy(v):.1f}" fill="{COLORS[0]}"><title>{escape(str(label))}: {v:g}</title></rect>'
        )
        parts.append(_text(x + slot / 2, H - BOTTOM + 16, label, 10))
    return _svg(parts)


def histogram(counts, edges, title="", xlabel="", ylabel="Count"):
    """
    Histogram from stats.histogram output: one bar per bin, labelled by its range.

    Parameters:
        counts : array of int, shape (bins,)
        edges : array of float, shape (bins + 1,)

    Returns:
        Figure
    """
    labels = [f"{lo:.0f}-{hi:.0f}" for lo, hi in zip(edges[:-1], edges[1:])]
    return bar_chart(labels, counts, title, xlabel, ylabel)


def line_chart(
    series, title="", xlabel="", ylabel="", logx=False, logy=False, legend="bottom-right"
):
    """
    Line chart with one or more lines and a legend.

    Parameters:
        series : list of (name, x, y) tuples
        title, xlabel, ylabel : str
        logx, logy : bool, use a log scale (all values on that axis must be > 0)
        legend : "bottom-right" or "top-left", where to put the legend box

    Returns:
        Figure
    """
    xs = np.concatenate([np.asarray(x, dtype=float) for _, x, _ in series])
    ys = np.concatenate([np.asarray(y, dtype=float) for _, _, y in series])
    xlim, ylim = (xs.min(), xs.max()), (ys.min(), ys.max())
    # leave a little space above and below the lines
    if logy:
        ylim = (ylim[0] / 2, ylim[1] * 2)
    else:
        pad = (ylim[1] - ylim[0]) * 0.05 or 1
        ylim = (ylim[0] - pad, ylim[1] + pad)
    parts, sx, sy = _frame(title, xlabel, ylabel, ylim, logy, xlim, logx)
    for i, (name, x, y) in enumerate(series):
        color = COLORS[i % len(COLORS)]
        points = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in zip(x, y))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    # legend on a white box so lines do not run through it
    if legend == "top-left":
        left, top = LEFT + 10, TOP + 5
    else:
        left, top = W - RIGHT - 185, H - BOTTOM - 12 - 18 * len(series)
    parts.append(
        f'<rect x="{left}" y="{top}" width="175" height="{18 * len(series) + 6}" '
        f'fill="white" stroke="#ccc"/>'
    )
    for i, (name, _, _) in enumerate(series):
        ly = top + 12 + 18 * i
        color = COLORS[i % len(COLORS)]
        parts.append(
            f'<line x1="{left + 10}" x2="{left + 30}" y1="{ly}" y2="{ly}" stroke="{color}" stroke-width="3"/>'
        )
        parts.append(_text(left + 36, ly + 4, name, 11, anchor="start"))
    return _svg(parts)
