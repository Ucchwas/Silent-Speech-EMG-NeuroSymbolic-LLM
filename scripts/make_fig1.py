#!/usr/bin/env python3
"""
scripts/make_fig1.py  ->  Figures/Overview_Pipeline_Adapter.png

Render Fig. 1 for the manuscript from the shared layout in
`scripts/fig1_layout.py`. The same layout is written into slide 2 of
Figures/Overview.pptx by `scripts/make_overview_pptx.py`, so the deck and the
figure in the paper cannot disagree.

Panel (b) keeps the geometry and the four bitmaps of the slide the submitted
version was drawn in, so the adapter panel stays the reader's familiar figure.
Panel (a) is reorganised around what the revision measures: one adapter feeding
two channels, with the symbolic constraint attached to the channel whose output
actually violates it.

Every number printed here is a measured test-split value recorded in
Results_reproduced/*.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fig1_layout import DIVIDER, H, OPS, W  # noqa: E402
from fig_style import (  # noqa: E402
    arrow, box, figure, fit_text, group, image, label_outline, panel_tag,
    plain, save_rgb,
)

ASSETS = ROOT / "TASLP_NS_Silent_Speech" / "Figures" / "assets"
OUT = ROOT / "TASLP_NS_Silent_Speech" / "Figures" / "Overview_Pipeline_Adapter.png"


def render(fig, ax) -> None:
    for e in OPS:
        kind = e["op"]

        if kind == "divider":
            ax.plot([e["x0"], e["x1"]], [e["y"], e["y"]], color="#B8C2CC",
                    lw=0.9, linestyle=(0, (6, 4)), zorder=0)

        elif kind == "tag":
            panel_tag(fig, ax, e["x"], e["y"], e["s"], size=e["size"])

        elif kind == "image":
            image(ax, str(ASSETS / e["path"]), e["x"], e["y"], e["w"], e["h"])

        elif kind == "plain":
            plain(ax, e["x"], e["y"], e["w"], e["h"], fill=e["fill"],
                  edge=e["edge"], lw=e["lw"])

        elif kind == "box":
            box(fig, ax, e["x"], e["y"], e["w"], e["h"], e["lines"],
                fill=e["fill"], size=e["size"], head_size=e["head"],
                edge=e["edge"])

        elif kind == "group":
            group(ax, e["x"], e["y"], e["w"], e["h"], color=e["color"])

        elif kind == "arrow":
            arrow(ax, e["p0"], e["p1"], conn=e["conn"], dashed=e["dashed"])

        elif kind == "text":
            if e["rotate"]:
                # Rotated labels sit inside a box drawn to hold them, so the
                # fitting pass would only measure the unrotated extent.
                ax.text(e["x"], e["y"], e["s"], rotation=e["rotate"],
                        ha="center", va="center", fontsize=e["size"],
                        zorder=6)
                continue
            t = fit_text(fig, ax, e["x"], e["y"], e["s"], e["w"], e["h"],
                         size=e["size"], weight=e["weight"], color=e["color"],
                         style=e["style"], ha=e["ha"], va=e["va"])
            if e["outline"]:
                label_outline(t)

        else:  # pragma: no cover
            raise ValueError(f"unknown layout op: {kind}")


def main() -> None:
    fig, ax = figure(W, H)
    render(fig, ax)
    save_rgb(fig, str(OUT), dpi=300)


if __name__ == "__main__":
    main()
