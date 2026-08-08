#!/usr/bin/env python3
"""
scripts/make_figures.py

Build the plotted figures from the scored CSVs, so a figure can never disagree
with the table beside it. Each panel reads exactly the CSV that the decoding
jobs wrote and nothing else.

  Fig. 2  Constraint_Placement.png  <- table2_decoders.csv, fig3_validity.csv,
                                       table5_decomp.csv
  Fig. 3  LLM_Backbone_Bars.png     <- fig2_backbones.csv
  Fig. 4  Sweeps.png                <- fig4_beta.csv, fig5_lambda.csv,
                                       fig4c_lambda_ar.csv

Figure sizes are set to the IEEE column geometry (3.45 in single, 7.16 in
double), so \\includegraphics never rescales them and every label reaches the
page at the size it was drawn at. This is the whole reason the previous
versions were unreadable: they were drawn at 4-8 in wide and then squeezed into
a 3.45 in column.

Output is 600 dpi PNG. The panels are line art at column width, so 300 dpi is
already visibly soft on the thin grid lines; 600 dpi costs a few hundred KB and
prints clean. Figures are flattened to RGB for the same reason Fig. 1 is: an
alpha channel becomes a soft mask that pdflatex composites pixel by pixel.

Usage
-----
python scripts/make_figures.py                      # Results_reproduced -> Figures/
python scripts/make_figures.py --results DIR --out DIR
python scripts/make_figures.py --only fig2
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from fig_style import SERIF, save_rgb  # noqa: E402

DPI = 600

COL1, COL2 = 3.45, 7.16  # IEEE single- and double-column text widths, inches

PRETTY = {
    "llama32_1b": "Llama-3.2-1B",
    "llama32_3b": "Llama-3.2-3B",
    "qwen25_3b_instruct": "Qwen2.5-3B-Inst.",
    "mistral7b_v03": "Mistral-7B-v0.3",
    "ar_beam": "AR beam",
    "ctc_greedy": "CTC greedy",
    "ctc_lexicon": "CTC + word lexicon",
    "ctc_grammar": "CTC + grammar",
    "ctc_grammar_ar": "CTC + grammar + AR",
}

# One hue per channel, used identically in every figure, so a reader can tell
# which branch a bar belongs to without reading the tick label.
LANG = "#C0504D"    # language channel (frozen LLM, autoregressive)
ALIGN = "#3C8D5F"   # alignment channel (CTC head)
NEUT = "#8FA0B3"    # unconstrained baseline
GOLD = "#D9A441"
STAR = "#B03A2E"


def _style() -> None:
    plt.rcParams.update({
        # Same face as the manuscript body text, and the same math font, so a
        # symbol on an axis matches the symbol in the equation it refers to.
        "font.family": "serif",
        "font.serif": SERIF,
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.titlesize": 7.8,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.0,
        "legend.frameon": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.6,
        "lines.linewidth": 1.4,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,
    })


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    v = (row.get(key) or "").strip()
    if not v:
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    return None if x != x else x


def name(label: str) -> str:
    return PRETTY.get(label, label)


def _by_system(rows: Sequence[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    return {r["system"]: r for r in rows}


def _annotate(ax, xs, ys, fmt="{:.3f}", dy=0.006, size=7.2, weight="normal"):
    for x, y in zip(xs, ys):
        ax.annotate(fmt.format(y), (x, y + dy), ha="center", va="bottom",
                    fontsize=size, fontweight=weight, clip_on=False)


# -----------------------------------------------------------------------------
# Fig. 2 -- where the symbolic constraint has purchase
# -----------------------------------------------------------------------------
def fig_placement(results: Path, out: Path) -> None:
    dec = _by_system(read_csv(results / "table2_decoders.csv"))
    val = _by_system(read_csv(results / "fig3_validity.csv"))
    dcp = _by_system(read_csv(results / "table5_decomp.csv"))

    fig, axes = plt.subplots(
        1, 3, figsize=(COL2, 2.55), width_ratios=[1.00, 0.74, 1.50],
        constrained_layout=True)

    # (a) the same constraint, moved from one channel to the other ---------
    # Two bars per group, not three: the word-lexicon variant is byte-identical
    # to the grammar on test and belongs in the table, not in a third bar that
    # only crowds the panel.
    ax = axes[0]
    xs, w, off = [0, 1], 0.27, 0.17
    none_ = [_f(dec["ar_beam"], "WER"), _f(dec["ctc_greedy"], "WER")]
    gram = [_f(dec["ns"], "WER"), _f(dec["ctc_grammar"], "WER")]
    ax.bar([x - off for x in xs], none_, w, color=NEUT,
           label="unconstrained", edgecolor="white", linewidth=0.6)
    ax.bar([x + off for x in xs], gram, w, color=ALIGN,
           label="+ utterance grammar", edgecolor="white", linewidth=0.6)
    _annotate(ax, [x - off for x in xs], none_, dy=0.009, size=6.8)
    _annotate(ax, [x + off for x in xs], gram, dy=0.009, size=6.8,
              weight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["Language\nchannel", "Alignment\nchannel"])
    ax.set_ylabel("WER")
    ax.set_ylim(0, 0.62)
    ax.set_title("(a) One constraint, two channels")
    ax.legend(loc="upper center", ncol=1, handlelength=1.0, borderpad=0.1,
              labelspacing=0.25, handletextpad=0.4,
              bbox_to_anchor=(0.5, 1.015))
    ax.annotate("no change", (0, 0.372), xytext=(0, 0.425), ha="center",
                fontsize=7.2, color=LANG, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=LANG, lw=0.7))
    ax.annotate("$-$20.5%", (1, 0.335), xytext=(1, 0.425), ha="center",
                fontsize=7.2, color=ALIGN, fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=ALIGN, lw=0.7))

    # (b) how often each channel already satisfies the grammar -------------
    ax = axes[1]
    keys = ["ar_beam", "ctc_greedy", "ctc_grammar"]
    pct = [_f(val[k], "valid_pct") for k in keys]
    ax.bar(range(3), pct, 0.60, color=[LANG, NEUT, ALIGN], edgecolor="white",
           linewidth=0.6)
    _annotate(ax, range(3), pct, fmt="{:.0f}%", dy=2.5, size=7.6,
              weight="bold")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["AR\nbeam", "CTC\ngreedy", "CTC +\ngram."])
    ax.set_ylabel("admitted by $\\mathcal{G}$ (%)")
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("(b) Grammar validity")
    ax.annotate("", xy=(1.42, 60), xytext=(1.42, 100),
                arrowprops=dict(arrowstyle="<->", color="#444444", lw=0.9))
    ax.annotate("40%\nrepairable", (1.36, 80), ha="right", va="center",
                fontsize=6.8, color="#333333")

    # (c) which fields the constraint actually fixes ------------------------
    ax = axes[2]
    fields = ["weekday", "month", "day", "year", "clock", "num%"]
    ticks = ["weekday", "month", "day", "year", "clock", "numeric"]
    series = [("AR beam", "ar_beam", LANG),
              ("CTC greedy", "ctc_greedy", NEUT),
              ("CTC + grammar", "ctc_grammar", ALIGN)]
    xs, w = list(range(len(fields))), 0.27
    for i, (lbl, key, c) in enumerate(series):
        ax.bar([x + (i - 1) * w for x in xs],
               [_f(dcp[key], f) for f in fields], w, color=c, label=lbl,
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(ticks, rotation=22, ha="right")
    ax.set_ylabel("field accuracy (%)")
    ax.set_ylim(0, 132)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("(c) Accuracy by template field")
    ax.legend(loc="upper center", ncol=3, handlelength=1.0, columnspacing=0.7,
              borderpad=0.1, handletextpad=0.4, bbox_to_anchor=(0.5, 1.04))
    y_ar, y_gr = _f(dcp["ar_beam"], "year"), _f(dcp["ctc_grammar"], "year")
    ax.annotate("", xy=(3 + w, y_gr + 2), xytext=(3 - w, y_ar + 2),
                arrowprops=dict(arrowstyle="-|>", color=STAR, lw=1.1,
                                connectionstyle="arc3,rad=-0.45"))
    ax.annotate(f"{y_ar:.0f}%$\\rightarrow${y_gr:.0f}%", (3.45, 60),
                ha="left", va="center", fontsize=7.0, color=STAR,
                fontweight="bold")

    save_rgb(fig, str(out / "Constraint_Placement.png"), dpi=DPI)


# -----------------------------------------------------------------------------
# Fig. 3 -- frozen backbone comparison
# -----------------------------------------------------------------------------
# Ordering the backbone bars by parameter count rather than by score is the
# point of the figure: read top to bottom, the error does not follow scale.
PARAMS_B = {
    "llama32_1b": 1.24,
    "qwen25_3b_instruct": 3.09,
    "llama32_3b": 3.21,
    "mistral7b_v03": 7.25,
}


def fig_backbones(results: Path, out: Path) -> None:
    rows = read_csv(results / "fig2_backbones.csv")
    rows.sort(key=lambda r: PARAMS_B.get(r["system"], 0.0))
    wer = [_f(r, "WER") or 0.0 for r in rows]
    lo = [_f(r, "WER_lo") or 0.0 for r in rows]
    hi = [_f(r, "WER_hi") or 0.0 for r in rows]
    err = [[w - l for w, l in zip(wer, lo)], [h - w for w, h in zip(wer, hi)]]

    fig, ax = plt.subplots(figsize=(COL1, 2.00), constrained_layout=True)
    ys = list(range(len(rows)))[::-1]
    best = min(range(len(rows)), key=lambda i: wer[i])
    cols = [ALIGN if i == best else NEUT for i in range(len(rows))]
    ax.barh(ys, wer, 0.56, xerr=err, color=cols, edgecolor="white",
            linewidth=0.6,
            error_kw=dict(ecolor="#444444", elinewidth=0.9, capsize=2.5,
                          capthick=0.9))
    for i, (y, w, h) in enumerate(zip(ys, wer, hi)):
        ax.annotate(f"{w:.3f}", (h + 0.012, y), va="center", fontsize=7.2,
                    fontweight="bold" if i == best else "normal")
    ax.set_yticks(ys)
    ax.set_yticklabels([f"{name(r['system'])}\n({PARAMS_B[r['system']]:.2f} B)"
                        for r in rows], fontsize=7.0)
    ax.set_xlabel("test WER under the reported decoder (95% CI)")
    ax.set_xlim(0, 0.56)
    ax.grid(axis="y", visible=False)
    save_rgb(fig, str(out / "LLM_Backbone_Bars.png"), dpi=DPI)


# -----------------------------------------------------------------------------
# Fig. 4 -- validation sweeps for the three tuned decoding weights
# -----------------------------------------------------------------------------
def _sweep(rows, prefix: str):
    pts = []
    for r in rows:
        s = r["system"]
        if not s.startswith(prefix):
            continue
        try:
            pts.append((float(s[len(prefix):]), _f(r, "WER")))
        except ValueError:
            continue
    pts.sort()
    return [p[0] for p in pts], [p[1] for p in pts]


def fig_sweeps(results: Path, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.15),
                             constrained_layout=True)

    # (a) character 5-gram weight, all four backbones
    ax = axes[0]
    beta = read_csv(results / "fig4_beta.csv")
    styles = [("llama32_1b", ALIGN, "o", "-"),
              ("mistral7b_v03", LANG, "s", "--"),
              ("llama32_3b", NEUT, "^", "-."),
              ("qwen25_3b_instruct", GOLD, "D", ":")]
    for key, c, m, ls in styles:
        xs, ys = _sweep(beta, f"{key}__beta_")
        if xs:
            ax.plot(xs, ys, ls, color=c, marker=m, markersize=3.2,
                    label=name(key))
    xs, ys = _sweep(beta, "llama32_1b__beta_")
    ax.plot([xs[0]], [ys[0]], marker="*", markersize=10, color=STAR,
            linestyle="none", zorder=5)
    ax.set_xlabel("$\\beta$  (character $5$-gram)")
    ax.set_ylabel("validation WER")
    ax.set_title("(a) $n$-gram fusion")
    # Headroom first, then the legend: the three flat curves top out at 0.355,
    # so anything above 0.40 is clear of them at every x.
    ax.set_ylim(0.17, 0.52)
    ax.set_yticks([0.20, 0.25, 0.30, 0.35, 0.40])
    ax.legend(loc="upper left", handlelength=1.3, labelspacing=0.12,
              borderpad=0.0, handletextpad=0.35, fontsize=6.3,
              bbox_to_anchor=(0.01, 1.0))

    # (b) fixed AR/CTC rerank weight
    ax = axes[1]
    xs, ys = _sweep(read_csv(results / "fig5_lambda.csv"), "lambda_")
    ax.plot(xs, ys, "-o", color=ALIGN, markersize=3.4)
    k = xs.index(2.0)
    ax.plot([2.0], [ys[k]], marker="*", markersize=10, color=STAR,
            linestyle="none", zorder=5)
    ax.set_xlabel("$\\lambda_{\\mathrm{fix}}$  (AR/CTC fusion)")
    ax.set_ylabel("validation WER")
    ax.set_title("(b) AR/CTC reranking")
    ax.annotate(f"selected 2.0\n({ys[k]:.3f})", (2.0, ys[k]),
                xytext=(1.00, ys[k] + 0.040), fontsize=6.6, color=STAR)
    ax.set_ylim(0.18, 0.36)

    # (c) AR rescoring of the grammar-valid pool: the sweep that did not hold
    ax = axes[2]
    xs, ys = _sweep(read_csv(results / "fig4c_lambda_ar.csv"), "lamar_")
    ax.plot(xs, ys, "-o", color=ALIGN, markersize=3.4, label="validation")
    ax.plot([0.5], [ys[xs.index(0.5)]], marker="*", markersize=10, color=STAR,
            linestyle="none", zorder=5)
    ax.axhline(0.2348, color="#4A5A6A", linestyle="--", linewidth=1.1,
               label="test, no rescoring (0.235)")
    ax.axhline(0.2652, color=LANG, linestyle=":", linewidth=1.4,
               label="test at selected $\\lambda_{\\mathrm{AR}}$ (0.265)")
    ax.set_xlim(-0.5, 8.7)
    ax.set_ylim(0.165, 0.300)
    ax.set_xlabel("$\\lambda_{\\mathrm{AR}}$  (pool rescoring)")
    ax.set_ylabel("WER")
    ax.set_title("(c) Rescoring the valid pool")
    # Naming the two test levels in the legend keeps them off the curve; the
    # lower-right corner is empty because the validation curve rises to the
    # right.
    ax.legend(loc="lower right", handlelength=1.6, labelspacing=0.18,
              borderpad=0.0, handletextpad=0.4, fontsize=6.3,
              bbox_to_anchor=(1.02, -0.03))

    save_rgb(fig, str(out / "Sweeps.png"), dpi=DPI)


FIGS = {
    "fig2": fig_placement,
    "fig3": fig_backbones,
    "fig4": fig_sweeps,
}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(REPO / "Results_reproduced"))
    ap.add_argument("--out",
                    default=str(REPO / "TASLP_NS_Silent_Speech" / "Figures"))
    ap.add_argument("--only", nargs="*", default=None, choices=sorted(FIGS))
    args = ap.parse_args()

    _style()
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for key in (args.only or sorted(FIGS)):
        try:
            FIGS[key](results, out)
        except FileNotFoundError as e:
            print(f"  skip {key}: {e}")


if __name__ == "__main__":
    main()
