#!/usr/bin/env python3
"""
scripts/fig_style.py

Shared drawing primitives for the manuscript figures.

The one thing every figure in this paper has to guarantee is that no label
ever spills out of the shape that contains it. Matplotlib will happily draw a
string ten times wider than its box, and the result only becomes visible after
the PDF is compiled, so the helpers here measure the rendered extent and shrink
the font until it fits. Nothing calls `ax.text` directly.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

# The figures are set in the same face as the body text. STIX is metrically
# compatible with Times New Roman and, unlike Nimbus Roman or DejaVu Serif,
# carries the full math alphabet the labels need (script L and G, blackboard R,
# bold H, the floor brackets), so a symbol in a figure and the same symbol in an
# equation are the same shape. `mathtext.fontset` has to be set too, or every
# $...$ label silently falls back to DejaVu and the figure ends up in two fonts.
_SERIF_PREF = ["STIX Two Text", "STIXGeneral", "Times New Roman",
               "Nimbus Roman", "DejaVu Serif"]


def _installed(prefs):
    """Keep the preferred order but drop families this machine does not have.

    Matplotlib warns once per text object for a missing family, which buries
    real output under thousands of findfont lines; it also means the figure
    silently falls back to whatever it lands on. Resolving the list up front
    makes the choice explicit and quiet.
    """
    import matplotlib.font_manager as fm

    have = {f.name for f in fm.fontManager.ttflist}
    keep = [p for p in prefs if p in have]
    return keep or list(prefs)


SERIF = _installed(_SERIF_PREF)
FONT = SERIF
PPTX_FONT = "Times New Roman"  # what the editable deck asks PowerPoint for

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": SERIF,
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
})

# Colours lifted from the slide deck the original Fig. 1 was drawn in, so the
# revision is recognisably the same figure.
GREEN = "#CCFFCC"   # data entering or leaving the system
YELLOW = "#FFFFCC"  # a block whose internals are shown elsewhere
CYAN = "#C3FFF4"    # the trainable adapter
BLUE = "#DAE8FC"    # ordinary processing block
GREY = "#E4E4E4"    # frozen parameters
ORANGE = "#FBDEC4"  # symbolic component
RED_TX = "#B03A2E"
GREEN_TX = "#1E7B4D"
EDGE = "#3B4A5A"


def figure(w_in: float, h_in: float, dpi: int = 300):
    """A figure whose data coordinates are inches, y measured downward."""
    fig = plt.figure(figsize=(w_in, h_in), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_in)
    ax.set_ylim(h_in, 0)  # y-down, matching the slide the layout came from
    ax.axis("off")
    ax.set_facecolor("white")
    fig.patch.set_facecolor("white")
    return fig, ax


def _fits(fig, txt, w_in: float, h_in: float) -> bool:
    fig.canvas.draw()
    bb = txt.get_window_extent(fig.canvas.get_renderer())
    return (bb.width / fig.dpi) <= w_in and (bb.height / fig.dpi) <= h_in


def fit_text(fig, ax, x: float, y: float, s: str, max_w: float, max_h: float,
             size: float = 9.0, min_size: float = 4.5, weight: str = "normal",
             color: str = "black", style: str = "normal", ha: str = "center",
             va: str = "center", linespacing: float = 1.32, zorder: float = 5):
    """Place `s` at (x, y) and shrink it until it fits inside max_w x max_h."""
    t = ax.text(x, y, s, ha=ha, va=va, fontsize=size, fontweight=weight,
                color=color, style=style, family=FONT, zorder=zorder,
                linespacing=linespacing)
    while size > min_size and not _fits(fig, t, max_w, max_h):
        size -= 0.25
        t.set_fontsize(size)
    return t


def box(fig, ax, x: float, y: float, w: float, h: float, lines: Sequence[str],
        fill: str = BLUE, size: float = 9.0, head_size: float | None = None,
        radius: float = 0.06, edge: str = EDGE, lw: float = 0.9,
        pad: float = 0.10, zorder: float = 3, dashed: bool = False):
    """A rounded box whose first line is bold and whose text is guaranteed to fit.

    `x, y` is the top-left corner, matching the slide coordinates the layout
    was measured from.
    """
    ax.add_patch(FancyBboxPatch(
        (x + radius, y + radius), w - 2 * radius, h - 2 * radius,
        boxstyle=f"round,pad={radius}", linewidth=lw, edgecolor=edge,
        facecolor=fill, zorder=zorder,
        linestyle=(0, (3, 2)) if dashed else "solid"))

    head, rest = lines[0], list(lines[1:])
    hs = head_size if head_size is not None else size
    cx = x + w / 2
    if not rest:
        fit_text(fig, ax, cx, y + h / 2, head, w - 2 * pad, h - 2 * pad,
                 size=hs, weight="bold", zorder=zorder + 1)
        return

    # Reserve the head's own line, then give the remainder to the body so a
    # long body can never push the title out of the box.
    hh = min(0.28, h * 0.42)
    th = fit_text(fig, ax, cx, y + pad + hh / 2, head, w - 2 * pad, hh,
                  size=hs, weight="bold", zorder=zorder + 1)
    fit_text(fig, ax, cx, y + pad + hh + (h - 2 * pad - hh) / 2, "\n".join(rest),
             w - 2 * pad, h - 2 * pad - hh, size=size, zorder=zorder + 1)
    return th


def group(ax, x: float, y: float, w: float, h: float, color: str = EDGE,
          fill: str = "none", lw: float = 0.9, zorder: float = 1):
    ax.add_patch(FancyBboxPatch(
        (x + 0.05, y + 0.05), w - 0.10, h - 0.10, boxstyle="round,pad=0.05",
        linewidth=lw, edgecolor=color, facecolor=fill,
        linestyle=(0, (5, 3)), zorder=zorder))


def plain(ax, x: float, y: float, w: float, h: float, fill: str = "white",
          edge: str = EDGE, lw: float = 0.9, zorder: float = 3):
    ax.add_patch(Rectangle((x, y), w, h, linewidth=lw, edgecolor=edge,
                           facecolor=fill, zorder=zorder))


def arrow(ax, p0: Tuple[float, float], p1: Tuple[float, float],
          color: str = EDGE, lw: float = 1.1, style: str = "-|>",
          conn: str = "arc3,rad=0", zorder: float = 2, dashed: bool = False):
    ax.annotate("", xy=p1, xytext=p0, zorder=zorder, annotation_clip=False,
                arrowprops=dict(arrowstyle=style, color=color, linewidth=lw,
                                shrinkA=0, shrinkB=0, connectionstyle=conn,
                                linestyle=(0, (3, 2)) if dashed else "solid"))


def image(ax, path: str, x: float, y: float, w: float, h: float,
          zorder: float = 3):
    import matplotlib.image as mpimg

    ax.imshow(mpimg.imread(path), extent=(x, x + w, y + h, y),
              aspect="auto", zorder=zorder, interpolation="antialiased")


def panel_tag(fig, ax, x: float, y: float, s: str, size: float = 13.0):
    ax.text(x, y, s, fontsize=size, fontweight="bold", family=FONT,
            ha="left", va="top", zorder=10)


def save_rgb(fig, path: str, dpi: int = 300):
    """Write a flattened RGB PNG.

    pdflatex turns a PNG alpha channel into a soft mask and composites it pixel
    by pixel; the figure this replaces was 4372x2042 RGBA and that alone took
    pdflatex over ten minutes on a single page.
    """
    from PIL import Image

    fig.savefig(path, dpi=dpi, facecolor="white", edgecolor="none")
    plt.close(fig)
    im = Image.open(path)
    if im.mode != "RGB":
        bg = Image.new("RGB", im.size, (255, 255, 255))
        bg.paste(im, mask=im.split()[-1] if im.mode == "RGBA" else None)
        bg.save(path)
        im = bg
    print(f"  {path}  {im.size[0]}x{im.size[1]}  {im.mode}")


def label_outline(t):
    t.set_path_effects([withStroke(linewidth=2.6, foreground="white")])
    return t
