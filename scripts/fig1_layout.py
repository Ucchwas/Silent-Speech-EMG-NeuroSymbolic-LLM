#!/usr/bin/env python3
"""
scripts/fig1_layout.py

The single description of Fig. 1, as a flat list of drawing operations in slide
inches (x right, y down, origin at the top-left of the slide).

Two things render this list and they must not disagree:

  scripts/make_fig1.py           -> Figures/Overview_Pipeline_Adapter.png
  scripts/make_overview_pptx.py  -> slide 2 of Figures/Overview.pptx

The deck is the editable copy. If the layout lived in the matplotlib script and
were transcribed into PowerPoint by hand, an edit to one would leave the other
stale and nothing would catch it: the PNG in the paper and the deck the figure
is supposed to come from would quietly say different things. Keeping the
geometry here means both are regenerated from the same numbers.

Coordinates follow the slide the submitted Fig. 1 was drawn in, so the revision
is recognisably the same figure. Every measured value printed in a label is a
test-split number recorded in Results_reproduced/*.csv.

Label text is written in matplotlib mathtext. `scripts/pptx_text.py` converts it
to Unicode runs for PowerPoint, which has no mathtext.
"""

from __future__ import annotations

from typing import Any, Dict, List

from fig_style import BLUE, CYAN, EDGE, GREEN, GREY, ORANGE, YELLOW

# Slide geometry. The PNG is cropped to the used height; the deck keeps the
# full 13.333 x 7.5 slide so PowerPoint opens it at its native size.
W, H = 13.333, 6.62
SLIDE_W, SLIDE_H = 13.333, 7.5
DIVIDER = 3.90

LANG_C = "#4A6FA5"   # language channel
ALIGN_C = "#2F7D52"  # alignment channel
RED_FILL, RED_EDGE = "#FBEAEA", "#B03A2E"
GRN_FILL = "#E6F5EC"


def _ops() -> List[Dict[str, Any]]:
    o: List[Dict[str, Any]] = []

    def image(path, x, y, w, h):
        o.append(dict(op="image", path=path, x=x, y=y, w=w, h=h))

    def plain(x, y, w, h, fill="white", edge=EDGE, lw=0.9):
        o.append(dict(op="plain", x=x, y=y, w=w, h=h, fill=fill, edge=edge,
                      lw=lw))

    def box(x, y, w, h, lines, fill=BLUE, size=9.0, head=None, edge=EDGE):
        o.append(dict(op="box", x=x, y=y, w=w, h=h, lines=list(lines),
                      fill=fill, size=size, head=head, edge=edge))

    def group(x, y, w, h, color=EDGE):
        o.append(dict(op="group", x=x, y=y, w=w, h=h, color=color))

    def arrow(p0, p1, conn="arc3,rad=0", dashed=False):
        o.append(dict(op="arrow", p0=p0, p1=p1, conn=conn, dashed=dashed))

    def text(x, y, s, w, h, size=8.0, weight="normal", color="black",
             ha="center", va="center", style="normal", outline=False,
             rotate=0.0):
        o.append(dict(op="text", x=x, y=y, s=s, w=w, h=h, size=size,
                      weight=weight, color=color, ha=ha, va=va, style=style,
                      outline=outline, rotate=rotate))

    def tag(x, y, s):
        o.append(dict(op="tag", x=x, y=y, s=s, size=13.0))

    # ------------------------------------------------------------------ (a)
    o.append(dict(op="divider", y=DIVIDER, x0=0.10, x1=W - 0.10))
    tag(0.10, 0.10, "(a)")

    # acquisition
    plain(0.16, 0.30, 1.26, 1.60, fill=GREEN)
    image("electrodes.png", 0.38, 0.38, 0.82, 0.82)
    text(0.79, 1.32, "EMG input", 1.10, 0.20, size=8.0, weight="bold")
    text(0.79, 1.62, "8-channel facial sEMG\nrecorded at 1 kHz", 1.10, 0.40,
         size=7.0)

    # front end
    arrow((1.44, 1.10), (1.72, 1.10))
    box(1.74, 0.74, 1.32, 0.72,
        ["Preprocess", "band-pass + notch filtering,",
         "27 ms window / 10 ms hop"], size=7.2)

    arrow((3.08, 1.10), (3.22, 1.10))
    box(3.24, 0.74, 1.14, 0.72,
        ["Features", "14 descriptors per channel,", "112-d per frame"],
        size=7.2)

    arrow((4.40, 1.10), (4.54, 1.10))
    box(4.56, 0.66, 1.50, 0.88,
        ["EMG adapter $G_\\phi$", "trainable, 16.86 M parameters",
         "conv + Transformer, $\\times4$ subsampling",
         "detailed in panel (b)"], fill=CYAN, size=7.0)

    # training objective
    box(0.16, 2.30, 5.90, 1.10,
        ["Training objective   (LLM backbone frozen)",
         "$\\mathcal{L}=\\mathcal{L}_{\\mathrm{AR}}+\\lambda_{\\mathrm{CTC}}\\,"
         "\\mathcal{L}_{\\mathrm{CTC}}$,    $\\lambda_{\\mathrm{CTC}}=0.2$",
         "update: adapter, prompts, AR and CTC heads       "
         "freeze: LLM backbone $\\theta$",
         "both terms carry weight: dropping $\\mathcal{L}_{\\mathrm{AR}}$ "
         "costs the CTC decoder 0.235 $\\rightarrow$ 0.311 WER"],
        fill=ORANGE, size=7.2, head=8.2)
    arrow((5.31, 2.28), (5.31, 1.56), dashed=True)

    # language channel
    group(6.56, 0.16, 6.62, 1.46, color=LANG_C)
    text(6.74, 0.32, "Language channel   (autoregressive)", 3.2, 0.20,
         size=8.6, weight="bold", color=LANG_C, ha="left", outline=True)
    box(6.76, 0.56, 1.70, 0.86,
        ["Frozen decoder-only LLM", "Llama-3.2-1B",
         "$\\theta$ fixed, never updated"], fill=GREY, size=7.2)
    arrow((8.48, 0.99), (8.64, 0.99))
    box(8.66, 0.56, 1.66, 0.86,
        ["AR character head", "beam search, $K=48$",
         "next-character posteriors"], size=7.2)
    arrow((10.34, 0.99), (10.50, 0.99))
    box(10.52, 0.52, 2.54, 0.94,
        ["Baseline output, measured on test",
         "already grammar-valid on 100% of utterances,",
         "yet the year field is correct only 10.5% of the time",
         "WER 0.333; adding a validity constraint here",
         "leaves every hypothesis byte-identical"],
        fill=RED_FILL, edge=RED_EDGE, size=6.6, head=7.4)

    # alignment channel
    group(6.56, 1.80, 6.62, 2.00, color=ALIGN_C)
    text(6.74, 1.96, "Alignment channel   (CTC head, bypasses the LLM)", 3.9,
         0.20, size=8.6, weight="bold", color=ALIGN_C, ha="left", outline=True)
    box(6.76, 2.18, 1.70, 0.82,
        ["CTC head", "frame-synchronous posteriors",
         "$p_t(\\cdot)$ over characters + blank"], size=7.2)
    arrow((8.48, 2.59), (8.64, 2.59))
    box(8.66, 2.18, 1.66, 0.82,
        ["Grammar-constrained", "CTC prefix beam search", "(Algorithm 2)"],
        fill=ORANGE, size=7.2)
    arrow((10.34, 2.59), (10.50, 2.59))
    box(10.52, 2.12, 2.54, 0.94,
        ["Final transcript   (reported system)",
         "60% grammar-valid before the constraint, 100% after",
         "year field correct 42.1% of the time",
         "WER 0.235, at no measurable decoding cost"],
        fill=GRN_FILL, edge=ALIGN_C, size=6.6, head=7.4)

    box(7.62, 3.14, 3.00, 0.54,
        ["Utterance grammar $\\mathcal{G}$",
         "54,510 admissible transcripts over three templates,",
         "induced from the 400 training transcripts alone"],
        fill=YELLOW, size=6.6, head=7.2)
    arrow((9.49, 3.12), (9.49, 3.02))

    # adapter fan-out
    arrow((6.08, 0.94), (6.72, 0.99), conn="arc3,rad=-0.20")
    arrow((6.08, 1.26), (6.72, 2.59), conn="arc3,rad=0.14")
    text(6.42, 1.72, "$\\mathbf{H}^{\\mathrm{emg}}$", 0.5, 0.24, size=9.0,
         outline=True)

    # ------------------------------------------------------------------ (b)
    y0 = 4.50
    mid = y0 + 0.56
    lab, lh = 5.66, 0.60
    tag(0.10, 3.99, "(b)")

    text(7.00, 4.06,
         "EMG adapter $G_\\phi$:   112-d frame features $\\rightarrow$ "
         "LLM-width embeddings, subsampled $\\times4$ in time",
         8.4, 0.22, size=8.6, weight="bold")

    image("featstack.png", 0.52, y0 + 0.24, 0.41, 0.65)
    box(0.07, lab, 1.24, lh,
        ["Input features", "$\\mathbf{E}\\in\\mathbb{R}^{T\\times112}$"],
        fill=GREEN, size=7.2)
    arrow((1.00, mid), (1.62, mid))

    image("cube.png", 1.72, y0 + 0.28, 0.52, 0.56)
    box(1.40, lab, 1.28, lh, ["Linear projection", "$112\\rightarrow512$"],
        fill=BLUE, size=7.2)
    arrow((2.30, mid), (2.86, mid))

    plain(2.88, y0 - 0.30, 2.02, 1.36, fill="#F4F7FB", edge="#8FA5BE", lw=0.8)
    for i, (nm, top, hgt) in enumerate([
            ("DS-Conv1D ($k$=5, $s$=2)", -0.24, 1.24),
            ("BatchNorm", -0.15, 1.06),
            ("GELU", 0.01, 0.72),
            ("Dropout 0.1", -0.04, 0.83)]):
        bx = 2.98 + i * 0.51
        plain(bx, y0 + top, 0.37, hgt, fill=BLUE, lw=0.8)
        text(bx + 0.185, y0 + top + hgt / 2, nm, hgt - 0.06, 0.33, size=6.0,
             rotate=90.0)
        if i < 3:
            arrow((bx + 0.37, mid), (bx + 0.49, mid))
    box(3.02, lab, 1.76, lh,
        ["Depthwise-separable conv block $\\times2$",
         "overall temporal subsampling $\\times4$"], fill=YELLOW, size=7.2)
    arrow((4.92, mid), (5.10, mid))

    plain(5.12, y0 + 0.06, 1.78, 0.46, fill=BLUE, lw=0.8)
    text(6.01, y0 + 0.29, "DS dilated conv $\\times3$  ($d$ = 1, 2, 4) + skip",
         1.66, 0.38, size=6.8)
    plain(5.30, y0 + 0.60, 1.42, 0.40, fill=BLUE, lw=0.8)
    text(6.01, y0 + 0.80, "Squeeze-and-excitation", 1.30, 0.32, size=6.8)
    box(5.20, lab, 1.62, lh,
        ["Residual temporal stack", "multi-resolution EMG dynamics"],
        fill=CYAN, size=7.2)
    arrow((6.92, mid), (7.10, mid))

    image("sinusoid.png", 7.12, y0 + 0.36, 1.52, 0.40)
    box(7.10, lab, 1.56, lh, ["Positional encoding", "sinusoidal, additive"],
        fill=BLUE, size=7.2)
    arrow((8.66, mid), (8.92, mid))

    for k in range(3):
        plain(8.98 + 0.045 * k, y0 + 0.16 + 0.055 * k, 1.28, 0.62, fill=YELLOW,
              lw=0.8)
    text(9.69, y0 + 0.52, "Transformer\nencoder", 1.12, 0.50, size=7.0,
         weight="bold")
    box(8.90, lab, 1.62, lh,
        ["4 layers, 8 heads, FFN $4\\times512$",
         "self-attention is bidirectional"], fill=YELLOW, size=7.2)
    arrow((10.38, mid), (10.60, mid))

    plain(10.62, y0 + 0.22, 1.16, 0.66, fill=GREY, lw=0.8)
    text(11.20, y0 + 0.55, "Output\nprojection\n$512\\rightarrow d$", 1.04,
         0.58, size=7.0)
    arrow((11.80, mid), (12.20, mid))
    box(11.72, lab, 1.54, lh,
        ["Adapter output",
         "$\\mathbf{H}^{\\mathrm{emg}}\\in\\mathbb{R}^{T'\\times d}$,"
         "  $T'=\\lfloor T/4\\rfloor$"], fill=GREEN, size=7.2)
    arrow((12.22, mid), (12.49, lab - 0.04), conn="arc3,rad=-0.3")

    text(6.66, 6.46,
         "Self-attention is bidirectional, so the adapter is non-causal: the "
         "system decodes complete segmented utterances offline and does not "
         "stream.", 10.4, 0.22, size=7.6, style="italic", color="#444444")

    return o


OPS = _ops()
