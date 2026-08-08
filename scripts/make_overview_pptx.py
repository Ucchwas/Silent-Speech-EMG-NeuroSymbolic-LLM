#!/usr/bin/env python3
"""
scripts/make_overview_pptx.py  ->  Figures/Overview.pptx, slide 2

Rewrite slide 2 of the deck, the slide that holds Fig. 1, as native editable
PowerPoint shapes: rounded rectangles with real text frames, connectors, and the
four bitmaps the original slide used. Slide 1 is not touched.

The shapes come from `scripts/fig1_layout.py`, the same list of drawing
operations that `scripts/make_fig1.py` renders into the PNG the manuscript
includes. Editing the deck and re-exporting therefore starts from exactly the
figure that is in the paper, and a change made here can be carried back by
editing the layout module and running both scripts.

Text is set in Times New Roman so the deck matches the manuscript body face;
the PNG uses STIX, which is metrically compatible and carries the math glyphs.
Mathtext in the layout is converted to Unicode with real sub- and superscript
runs by `scripts/pptx_text.py`.

The first run makes a copy of the untouched deck next to it, so the submitted
figure is still recoverable.

Usage
-----
python scripts/make_overview_pptx.py
python scripts/make_overview_pptx.py --pptx PATH --slide 2
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.dml import MSO_LINE_DASH_STYLE
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fig1_layout import OPS, SLIDE_H, SLIDE_W  # noqa: E402
from fig_style import EDGE, PPTX_FONT  # noqa: E402
from pptx_text import split_runs, unknown_commands  # noqa: E402

FIGDIR = ROOT / "TASLP_NS_Silent_Speech" / "Figures"
ASSETS = FIGDIR / "assets"
PPTX = FIGDIR / "Overview.pptx"
BACKUP = FIGDIR / "Overview_submitted.pptx"

EMU_PER_IN = 914400


def inch(v: float) -> Emu:
    return Emu(int(round(v * EMU_PER_IN)))


NAMED = {"black": "#000000", "white": "#FFFFFF"}


def rgb(spec):
    """Accept the matplotlib colour spellings the layout uses."""
    if not spec or spec == "none":
        return None
    spec = NAMED.get(spec, spec)
    return RGBColor.from_string(spec.lstrip("#").upper())


# -----------------------------------------------------------------------------
# text
# -----------------------------------------------------------------------------
# Times New Roman has no script or blackboard-bold alphabet and no floor
# brackets. Left in a Times run, these become whatever PowerPoint happens to
# fall back to, which differs between machines. Cambria Math ships with Office
# and draws them correctly, so those characters get their own runs.
MATH_ONLY = set("ℒ𝒢ℋ𝒟ℛ𝒞𝒱𝒯𝒲ℝℕℤℚ⌊⌋∅′…")
MATH_FONT = "Cambria Math"


def _by_font(text: str):
    """Split a run into (text, font) pieces so rare glyphs keep a font that has them."""
    out, buf, cur = [], "", None
    for ch in text:
        f = MATH_FONT if ch in MATH_ONLY else PPTX_FONT
        if f != cur and buf:
            out.append((buf, cur))
            buf = ""
        buf, cur = buf + ch, f
    if buf:
        out.append((buf, cur))
    return out


def _set_runs(par, s: str, size: float, bold: bool, color: str,
              italic: bool = False) -> None:
    """Fill one paragraph, splitting mathtext into baseline-shifted runs."""
    for text, base in split_runs(s):
        for piece, fontname in _by_font(text):
            r = par.add_run()
            r.text = piece
            f = r.font
            f.name, f.size, f.bold, f.italic = fontname, Pt(size), bold, italic
            f.color.rgb = rgb(color) or RGBColor(0, 0, 0)
            if base:
                # python-pptx has no subscript property; PowerPoint stores it
                # as a per-mille baseline shift on the run properties.
                r.font._rPr.set("baseline",
                                "-25000" if base < 0 else "30000")


def _fill_frame(tf, lines, sizes, bolds, color="#000000", italic=False,
                align=PP_ALIGN.CENTER) -> None:
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = inch(0.03)
    tf.margin_top = tf.margin_bottom = inch(0.02)
    for i, line in enumerate(lines):
        par = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        par.alignment = align
        par.line_spacing = 0.95
        if i:
            par.space_before = Pt(1)
        _set_runs(par, line, sizes[i], bolds[i], color, italic)


# -----------------------------------------------------------------------------
# shapes
# -----------------------------------------------------------------------------
def _outline(shape, color, width_pt=0.9, dashed=False):
    ln = shape.line
    if color is None:
        ln.fill.background()
        return
    ln.color.rgb = rgb(color)
    ln.width = Pt(width_pt)
    if dashed:
        ln.dash_style = MSO_LINE_DASH_STYLE.DASH


def add_box(slide, e):
    """A rounded rectangle whose first line is the bold heading."""
    sh = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, inch(e["x"]), inch(e["y"]),
        inch(e["w"]), inch(e["h"]))
    # PowerPoint's corner adjustment is a fraction of the shorter side; the PNG
    # uses a fixed 0.06 in radius, so convert rather than leave the default 0.16
    # which would round a short box into a lozenge.
    sh.adjustments[0] = min(0.5, 0.06 / max(0.01, min(e["w"], e["h"])))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(e["fill"])
    _outline(sh, e["edge"], 0.9)
    sh.shadow.inherit = False

    lines = e["lines"]
    head = e["head"] or e["size"]
    sizes = [head] + [e["size"]] * (len(lines) - 1)
    bolds = [True] + [False] * (len(lines) - 1)
    _fill_frame(sh.text_frame, lines, sizes, bolds)
    return sh


def add_plain(slide, e):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, inch(e["x"]), inch(e["y"]),
                                inch(e["w"]), inch(e["h"]))
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(e["fill"])
    _outline(sh, e["edge"], e["lw"])
    sh.shadow.inherit = False
    sh.text_frame.word_wrap = True
    return sh


def add_group(slide, e):
    """The dashed container that names a channel."""
    sh = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, inch(e["x"]), inch(e["y"]),
        inch(e["w"]), inch(e["h"]))
    sh.adjustments[0] = 0.04
    sh.fill.background()
    _outline(sh, e["color"], 1.0, dashed=True)
    sh.shadow.inherit = False
    return sh


def add_text(slide, e):
    w, h = max(e["w"], 0.25), max(e["h"], 0.18)
    if e["rotate"]:
        w, h = h, w  # the box is drawn upright, then rotated
    left = e["x"] - (w / 2 if e["ha"] == "center" else 0.0)
    top = e["y"] - (h / 2 if e["va"] == "center" else 0.0)
    sh = slide.shapes.add_textbox(inch(left), inch(top), inch(w), inch(h))
    align = {"center": PP_ALIGN.CENTER, "left": PP_ALIGN.LEFT,
             "right": PP_ALIGN.RIGHT}[e["ha"]]
    lines = e["s"].split("\n")
    _fill_frame(sh.text_frame, lines, [e["size"]] * len(lines),
                [e["weight"] == "bold"] * len(lines), color=e["color"],
                italic=e["style"] == "italic", align=align)
    if e["rotate"]:
        sh.rotation = 360.0 - e["rotate"]
    return sh


def add_tag(slide, e):
    sh = slide.shapes.add_textbox(inch(e["x"]), inch(e["y"]), inch(0.42),
                                  inch(0.28))
    _fill_frame(sh.text_frame, [e["s"]], [e["size"]], [True],
                align=PP_ALIGN.LEFT)
    return sh


def _arrowhead(shape):
    el = shape.line._get_or_add_ln()
    for old in el.findall(qn("a:tailEnd")):
        el.remove(old)
    el.append(el.makeelement(qn("a:tailEnd"),
                             {"type": "triangle", "w": "sm", "len": "sm"}))


def _curvature(conn: str) -> float:
    """The arc3 radius the layout asked for, 0 for a straight arrow."""
    m = re.search(r"rad\s*=\s*(-?\d*\.?\d+)", conn)
    return float(m.group(1)) if m else 0.0


def add_arrow(slide, e):
    (x0, y0), (x1, y1) = e["p0"], e["p1"]
    kind = (MSO_CONNECTOR.CURVE if _curvature(e["conn"])
            else MSO_CONNECTOR.STRAIGHT)
    sh = slide.shapes.add_connector(kind, inch(x0), inch(y0), inch(x1),
                                    inch(y1))
    _outline(sh, EDGE, 1.1, dashed=e["dashed"])
    _arrowhead(sh)
    return sh


def add_image(slide, e):
    return slide.shapes.add_picture(str(ASSETS / e["path"]), inch(e["x"]),
                                    inch(e["y"]), inch(e["w"]), inch(e["h"]))


def add_divider(slide, e):
    sh = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, inch(e["x0"]),
                                    inch(e["y"]), inch(e["x1"]), inch(e["y"]))
    _outline(sh, "#B8C2CC", 0.9, dashed=True)
    return sh


BUILD = {
    "divider": add_divider, "tag": add_tag, "image": add_image,
    "plain": add_plain, "box": add_box, "group": add_group,
    "arrow": add_arrow, "text": add_text,
}


# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pptx", default=str(PPTX))
    ap.add_argument("--slide", type=int, default=2,
                    help="1-based slide number to rewrite (default: 2)")
    args = ap.parse_args()

    path = Path(args.pptx)
    if not BACKUP.exists() and path == PPTX:
        shutil.copy2(path, BACKUP)
        print(f"  kept the untouched deck at {BACKUP.name}")

    prs = Presentation(str(path))
    if prs.slide_width != inch(SLIDE_W) or prs.slide_height != inch(SLIDE_H):
        prs.slide_width, prs.slide_height = inch(SLIDE_W), inch(SLIDE_H)

    idx = args.slide - 1
    if not 0 <= idx < len(prs.slides):
        raise SystemExit(f"slide {args.slide} does not exist "
                         f"({len(prs.slides)} in the deck)")
    slide = prs.slides[idx]

    removed = 0
    for sh in list(slide.shapes):
        sh._element.getparent().remove(sh._element)
        removed += 1

    for e in OPS:
        BUILD[e["op"]](slide, e)

    missing = unknown_commands()
    if missing:
        raise SystemExit(f"unmapped mathtext commands: {sorted(missing)} "
                         f"(add them to scripts/pptx_text.py)")

    prs.save(str(path))
    print(f"  slide {args.slide}: replaced {removed} shape(s) with "
          f"{len(OPS)} from fig1_layout")
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
