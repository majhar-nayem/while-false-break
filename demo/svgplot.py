"""Tiny SVG charting helper for the demo and report, using only NumPy + stdlib.

The brief allows only Python and NumPy, so instead of matplotlib the charts
are written directly as SVG text. Every chart returns a :class:`Figure`,
which Jupyter displays inline (``_repr_svg_``) and which can be saved for
the report with :meth:`Figure.save`. Hovering a mark in a browser shows its
exact value (an SVG ``<title>`` tooltip).

Design rules: thin marks, a recessive grid, text in neutral ink (never the
series colour), a legend whenever there are two or more series, and a fixed,
colour-blind-validated categorical order (blue, orange, aqua, yellow).
"""

from __future__ import annotations

import math
from html import escape
from pathlib import Path

import numpy as np

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
BLUE_RAMP = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#256abf",
    "#184f95",
    "#0d366b",
]
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984"
GRID, AXIS = "#e6e5e1", "#b9b8b2"
FONT = "Inter, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


class Figure:
    """An SVG drawing with a title, plot area and helpers for axes and marks."""

    def __init__(
        self,
        width=640,
        height=360,
        title="",
        subtitle="",
        margin=None,
        legend_row=False,
    ):
        self.w, self.h = width, height
        top = 64 if subtitle else 48
        self.legend_y = top + 4
        self.m = {"l": 64, "r": 24, "t": top + (26 if legend_row else 0), "b": 52}
        self.m.update(margin or {})
        self.parts: list[str] = []
        if title:
            self.text(16, 26, title, size=15, weight=600, anchor="start")
        if subtitle:
            self.text(16, 46, subtitle, size=12, color=INK2, anchor="start")

    # plot-area geometry
    @property
    def x0(self):
        return self.m["l"]

    @property
    def x1(self):
        return self.w - self.m["r"]

    @property
    def y0(self):
        return self.h - self.m["b"]

    @property
    def y1(self):
        return self.m["t"]

    def text(
        self, x, y, s, size=11, color=INK2, anchor="middle", weight=400, rotate=None
    ):
        """Add a text element."""
        rot = f' transform="rotate({rotate} {x:.1f} {y:.1f})"' if rotate else ""
        self.parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" '
            f'font-weight="{weight}" text-anchor="{anchor}"{rot}>'
            f"{escape(str(s))}</text>"
        )

    def add(self, svg: str):
        """Append raw SVG markup."""
        self.parts.append(svg)

    def legend(self, names, colors, dashed=()):
        """Draw a one-row legend on its own line below the subtitle."""
        x, y = 16.0, self.legend_y
        for name, color in zip(names, colors, strict=True):
            dash = ' stroke-dasharray="4 3"' if name in dashed else ""
            self.add(
                f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x + 18:.1f}" y2="{y:.1f}" '
                f'stroke="{color}" stroke-width="2.5" stroke-linecap="round"{dash}/>'
            )
            self.text(x + 24, y + 4, name, size=11, color=INK2, anchor="start")
            x += 24 + 6.8 * len(name) + 20

    def svg(self) -> str:
        """Return the complete SVG document."""
        body = "\n".join(self.parts)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.w} {self.h}" '
            f'width="{self.w}" height="{self.h}" font-family="{FONT}">\n'
            f'<rect width="100%" height="100%" fill="{SURFACE}"/>\n{body}\n</svg>\n'
        )

    def _repr_svg_(self):
        return self.svg()

    def save(self, path) -> Path:
        """Write the SVG to ``path`` (parent folders are created)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.svg())
        return path


# --------------------------------------------------------------------------
# Scales and ticks
# --------------------------------------------------------------------------
def nice_ticks(lo, hi, n=5):
    """Round tick values (steps of 1, 2 or 5 x 10^k) covering [lo, hi]."""
    if hi <= lo:
        hi = lo + 1
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 5, 10) if s * mag >= raw)
    start = math.floor(lo / step + 1e-9) * step  # first tick at or below lo
    stop = math.ceil(hi / step - 1e-9) * step  # last tick at or above hi
    return np.arange(start, stop + step * 0.5, step)


def log_ticks(lo, hi):
    """Powers of ten spanning [lo, hi] (both > 0)."""
    a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
    return 10.0 ** np.arange(a, b + 1)


def fmt(v):
    """Compact tick label: 1k, 2.5, 1e-08, 0.05..."""
    v = float(v)
    if v == 0:
        return "0"
    a = abs(v)
    if a >= 1e4 or a < 1e-3:
        e = int(round(math.log10(a)))
        return f"1e{e}" if math.isclose(a, 10**e) else f"{v:.0e}"
    if a >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


class Scale:
    """Map data values to pixels (linear or log10)."""

    def __init__(self, lo, hi, p0, p1, log=False):
        self.log = log
        self.lo, self.hi = (math.log10(lo), math.log10(hi)) if log else (lo, hi)
        self.p0, self.p1 = p0, p1

    def __call__(self, v):
        v = np.asarray(v, dtype=float)
        t = np.log10(v) if self.log else v
        return self.p0 + (t - self.lo) / (self.hi - self.lo) * (self.p1 - self.p0)


def _axes(fig, sx, sy, xticks, yticks, xlabel, ylabel, xfmt=fmt, yfmt=fmt):
    for t in yticks:
        y = float(sy(t))
        fig.add(
            f'<line x1="{fig.x0}" x2="{fig.x1}" y1="{y:.1f}" y2="{y:.1f}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        fig.text(fig.x0 - 8, y + 4, yfmt(t), anchor="end", color=MUTED)
    for t in xticks:
        x = float(sx(t))
        fig.add(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{fig.y0}" y2="{fig.y0 + 4}" '
            f'stroke="{AXIS}" stroke-width="1"/>'
        )
        fig.text(x, fig.y0 + 17, xfmt(t), color=MUTED)
    fig.add(
        f'<line x1="{fig.x0}" x2="{fig.x1}" y1="{fig.y0}" y2="{fig.y0}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    if xlabel:
        fig.text((fig.x0 + fig.x1) / 2, fig.h - 12, xlabel, size=12, color=INK2)
    if ylabel:
        cy = (fig.y0 + fig.y1) / 2
        fig.text(16, cy, ylabel, size=12, color=INK2, rotate=-90)


def _bar_path(x, w, ytop, ybase, r=4.0):
    """Vertical bar with only the data end (top) rounded."""
    h = ybase - ytop
    r = max(0.0, min(r, w / 2, h))
    return (
        f"M{x:.1f},{ybase:.1f} V{ytop + r:.1f} "
        f"Q{x:.1f},{ytop:.1f} {x + r:.1f},{ytop:.1f} "
        f"H{x + w - r:.1f} Q{x + w:.1f},{ytop:.1f} {x + w:.1f},{ytop + r:.1f} "
        f"V{ybase:.1f} Z"
    )


def _hbar_path(xbase, xend, y, h, r=4.0):
    """Horizontal bar with only the data end (right) rounded."""
    w = xend - xbase
    r = max(0.0, min(r, h / 2, w))
    return (
        f"M{xbase:.1f},{y:.1f} H{xend - r:.1f} "
        f"Q{xend:.1f},{y:.1f} {xend:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} Q{xend:.1f},{y + h:.1f} {xend - r:.1f},{y + h:.1f} "
        f"H{xbase:.1f} Z"
    )


# --------------------------------------------------------------------------
# Chart types
# --------------------------------------------------------------------------
def histogram(counts, edges, title="", subtitle="", xlabel="", ylabel="Count", **kw):
    """Bars spanning ``edges`` with heights ``counts`` (e.g. stats.histogram)."""
    counts, edges = np.asarray(counts), np.asarray(edges, dtype=float)
    fig = Figure(title=title, subtitle=subtitle, **kw)
    yt = nice_ticks(0, counts.max() * 1.05)
    sx = Scale(edges[0], edges[-1], fig.x0, fig.x1)
    sy = Scale(0, yt[-1], fig.y0, fig.y1)
    _axes(fig, sx, sy, nice_ticks(edges[0], edges[-1], 8), yt, xlabel, ylabel)
    for c, a, b in zip(counts, edges[:-1], edges[1:], strict=True):
        xa, xb = float(sx(a)) + 1, float(sx(b)) - 1  # 2px gap between bars
        yt_ = float(sy(c))
        fig.add(
            f'<path d="{_bar_path(xa, xb - xa, yt_, fig.y0)}" fill="{SERIES[0]}">'
            f"<title>[{a:g}, {b:g}): {int(c)}</title></path>"
        )
    return fig


def line_chart(
    series,
    title="",
    subtitle="",
    xlabel="",
    ylabel="",
    logx=False,
    logy=False,
    xlim=None,
    ylim=None,
    reference=None,
    markers=False,
    step=False,
    **kw,
):
    """Line chart of ``[(name, x, y), ...]`` with an optional dashed reference.

    ``reference`` is ``(name, x, y)`` drawn as a dashed neutral line (e.g. the
    chance diagonal of a ROC plot). ``step=True`` draws a staircase.
    """
    n_items = len(series) + (reference is not None)
    fig = Figure(title=title, subtitle=subtitle, legend_row=n_items >= 2, **kw)
    xs = np.concatenate([np.asarray(s[1], float) for s in series])
    ys = np.concatenate([np.asarray(s[2], float) for s in series])
    xlo, xhi = xlim or (xs.min(), xs.max())
    ylo, yhi = ylim or (ys.min(), ys.max())
    xt = log_ticks(xlo, xhi) if logx else nice_ticks(xlo, xhi)
    yt = log_ticks(ylo, yhi) if logy else nice_ticks(ylo, yhi)
    if not xlim:
        xlo, xhi = xt[0], xt[-1]
    if not ylim:
        ylo, yhi = yt[0], yt[-1]
    sx = Scale(xlo, xhi, fig.x0, fig.x1, log=logx)
    sy = Scale(ylo, yhi, fig.y0, fig.y1, log=logy)
    _axes(fig, sx, sy, xt, yt, xlabel, ylabel)

    def path(x, y):
        px, py = sx(x), sy(y)
        if step:
            px, py = np.repeat(px, 2)[1:], np.repeat(py, 2)[:-1]
        return "M" + " L".join(f"{a:.1f},{b:.1f}" for a, b in zip(px, py, strict=True))

    names, colors, dashed = [], [], []
    if reference is not None:
        name, x, y = reference
        fig.add(
            f'<path d="{path(np.asarray(x, float), np.asarray(y, float))}" fill="none" '
            f'stroke="{MUTED}" stroke-width="1.5" stroke-dasharray="4 3"/>'
        )
        names.append(name)
        colors.append(MUTED)
        dashed.append(name)
    for i, (name, x, y) in enumerate(series):
        x, y = np.asarray(x, float), np.asarray(y, float)
        color = SERIES[i]
        fig.add(
            f'<path d="{path(x, y)}" fill="none" stroke="{color}" stroke-width="2" '
            f'stroke-linejoin="round" stroke-linecap="round"/>'
        )
        if markers:
            for a, b in zip(sx(x), sy(y), strict=True):
                fig.add(
                    f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{color}" '
                    f'stroke="{SURFACE}" stroke-width="2"/>'
                )
            for a, b, xv, yv in zip(sx(x), sy(y), x, y, strict=True):
                fig.add(
                    f'<circle cx="{a:.1f}" cy="{b:.1f}" r="10" fill="transparent">'
                    f"<title>{escape(name)}: x={xv:g}, y={yv:.3g}</title></circle>"
                )
        names.append(name)
        colors.append(color)
    if len(names) >= 2:
        fig.legend(names, colors, dashed)
    return fig


def hbar_chart(
    labels,
    values,
    title="",
    subtitle="",
    xlabel="",
    logx=False,
    value_fmt="{:.0f}×",
    baseline=None,
    **kw,
):
    """Single-series horizontal bars (sorted by the caller), labelled at the end."""
    values = np.asarray(values, float)
    n = len(values)
    left = 8 + 6.4 * max(len(s) for s in labels)
    kw.setdefault("height", 64 + 48 + n * 26)
    fig = Figure(title=title, subtitle=subtitle, margin={"l": left, "r": 56}, **kw)
    if logx:
        lo = baseline or 10 ** math.floor(math.log10(values.min()))
        xt = log_ticks(lo, values.max())
    else:
        lo = 0
        xt = nice_ticks(0, values.max())
    sx = Scale(xt[0], xt[-1], fig.x0, fig.x1, log=logx)
    band = (fig.y0 - fig.y1) / n
    for t in xt:
        x = float(sx(t))
        fig.add(
            f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{fig.y1}" y2="{fig.y0}" '
            f'stroke="{GRID}" stroke-width="1"/>'
        )
        fig.text(
            x, fig.y0 + 17, fmt(t) + ("×" if "×" in value_fmt else ""), color=MUTED
        )
    if xlabel:
        fig.text((fig.x0 + fig.x1) / 2, fig.h - 12, xlabel, size=12, color=INK2)
    for i, (lab, v) in enumerate(zip(labels, values, strict=True)):
        y = fig.y1 + i * band + band * 0.18
        h = band * 0.64
        xe = float(sx(max(v, xt[0])))
        fig.add(
            f'<path d="{_hbar_path(fig.x0, xe, y, h)}" fill="{SERIES[0]}">'
            f"<title>{escape(lab)}: {value_fmt.format(v)}</title></path>"
        )
        fig.text(fig.x0 - 8, y + h / 2 + 4, lab, anchor="end", color=INK)
        fig.text(xe + 6, y + h / 2 + 4, value_fmt.format(v), anchor="start", color=INK2)
    fig.add(
        f'<line x1="{fig.x0}" x2="{fig.x0}" y1="{fig.y1}" y2="{fig.y0}" '
        f'stroke="{AXIS}" stroke-width="1"/>'
    )
    return fig


def heatmap(
    matrix,
    row_labels,
    col_labels,
    title="",
    subtitle="",
    xlabel="",
    ylabel="",
    value_fmt="{:d}",
    **kw,
):
    """Annotated matrix (e.g. a confusion matrix) on a single-hue blue ramp."""
    M = np.asarray(matrix)
    r, c = M.shape
    kw.setdefault("width", 420)
    kw.setdefault("height", 120 + 70 * r)
    fig = Figure(title=title, subtitle=subtitle, margin={"l": 110, "b": 56}, **kw)
    cw, ch = (fig.x1 - fig.x0) / c, (fig.y0 - fig.y1) / r
    lo, hi = M.min(), M.max()
    for i in range(r):
        for j in range(c):
            t = 0 if hi == lo else (M[i, j] - lo) / (hi - lo)
            k = int(round(t * (len(BLUE_RAMP) - 1)))
            fill = BLUE_RAMP[k]
            ink = SURFACE if k >= 3 else INK
            x, y = fig.x0 + j * cw, fig.y1 + i * ch
            fig.add(
                f'<rect x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cw - 2:.1f}" '
                f'height="{ch - 2:.1f}" rx="4" fill="{fill}">'
                f"<title>{row_labels[i]} → {col_labels[j]}: {M[i, j]}</title></rect>"
            )
            fig.text(
                x + cw / 2,
                y + ch / 2 + 6,
                value_fmt.format(M[i, j]),
                size=16,
                color=ink,
                weight=600,
            )
    for j, lab in enumerate(col_labels):
        fig.text(fig.x0 + (j + 0.5) * cw, fig.y0 + 18, lab, color=INK)
    for i, lab in enumerate(row_labels):
        fig.text(fig.x0 - 10, fig.y1 + (i + 0.5) * ch + 4, lab, anchor="end", color=INK)
    if xlabel:
        fig.text((fig.x0 + fig.x1) / 2, fig.h - 12, xlabel, size=12, color=INK2)
    if ylabel:
        fig.text(16, (fig.y0 + fig.y1) / 2, ylabel, size=12, color=INK2, rotate=-90)
    return fig
