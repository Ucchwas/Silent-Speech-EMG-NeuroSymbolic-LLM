#!/usr/bin/env python3
"""
scripts/make_fig_representation.py  ->  Figures/Target_Representation.png

Fig. 5 of the manuscript: what the target representation is worth. The panel
this script draws was previously the one figure with no generator in the repo,
so it could not be rebuilt when a label changed.

Every value comes from Results_Current/fig_representation.csv:

  wer_mean / wer_sd / em_pct   Table III (three training seeds: 1234, 7, 2024)
  seed_k                       the individual seed WERs, stored as integer word
                               error counts k over the 132 reference words of
                               the test split, so a WER is exactly k/132
  day / year / clock / num     Table IV(b), per-field accuracy at the tuned seed

Storing the seeds as k rather than as a rounded decimal keeps the bars, the
whiskers and the printed means consistent to the last digit: the mean of
31|36|40 over 132 is 0.2702, which rounds to the 0.270 the table prints.

Panel (b) shows the three systems that share the written-form grammar. The
no-backbone control appears only in panel (a), set apart by a dotted rule,
because it answers a different question (Sec. IV-G).

Usage:
    python scripts/make_fig_representation.py \
        --results Results_Current \
        --out TASLP_Silent_Speech_Recognition/Figures
"""

from __future__ import annotations

import argparse
import csv
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from fig_style import SERIF, save_rgb  # noqa: E402

DPI = 600
COL1 = 3.45          # IEEE single-column text width, inches
N_REF_WORDS = 132    # reference words in the silent test split

# Same hues as the other figures: the written-form system is the neutral bar,
# verbalisation is the gold accent, phonemes the alignment-channel green. The
# no-backbone control is the same green, lightened, because it is the phoneme
# system with a part removed rather than a different representation.
NEUT = "#8FA0B3"
GOLD = "#D9A441"
ALIGN = "#3C8D5F"
ALIGN_LT = "#A9C9B4"
RED_TX = "#B03A2E"

BAR_COLORS = [NEUT, GOLD, ALIGN, ALIGN_LT]
PANEL_B = ["characters", "verbalized_letters", "phonemes"]
FIELDS = [("day", "day"), ("year", "year"), ("clock", "clock"),
          ("num", "numeric overall")]


def _style() -> None:
    plt.rcParams.update({
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


def read_rows(path: Path) -> dict:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = {}
    for r in rows:
        r["seeds"] = [int(k) / N_REF_WORDS for k in r["seed_k"].split("|")]
        for key in ("wer_mean", "wer_sd", "em_pct", "day", "year", "clock",
                    "num"):
            r[key] = float(r[key])
        out[r["system"]] = r
    return out


def panel_a(ax, data: dict) -> None:
    keys = list(data)
    xs = [0, 1, 2, 3.35]          # the control sits apart from the three targets
    means = [data[k]["wer_mean"] for k in keys]
    sds = [data[k]["wer_sd"] for k in keys]

    ax.bar(xs, means, 0.72, color=BAR_COLORS, edgecolor="white", linewidth=0.6)
    ax.errorbar(xs, means, yerr=sds, fmt="none", ecolor="#555555",
                elinewidth=1.0, capsize=2.6, capthick=1.0, zorder=4)

    for x, k in zip(xs, keys):
        ax.plot([x] * len(data[k]["seeds"]), data[k]["seeds"], "o",
                mfc="white", mec="#444444", mew=0.8, ms=3.2, zorder=5,
                clip_on=False)

    for x, m, s, k in zip(xs, means, sds, keys):
        ax.annotate("{:.3f}".format(m), (x, m + s + 0.012), ha="center",
                    va="bottom", fontsize=7.4,
                    fontweight="bold" if k == "phonemes" else "normal",
                    clip_on=False)

    ax.axvline(2.68, color="#BBBBBB", lw=0.8, ls=(0, (2, 2)), zorder=0)
    ax.set_xticks(xs)
    # Wrap at 11 characters so "verbalized letters" and "phonemes, no
    # backbone" each break onto two lines and no label runs past the axes.
    ax.set_xticklabels(
        ["{}\nEM {:.1f}%".format("\n".join(textwrap.wrap(data[k]["label"], 11)),
                                 data[k]["em_pct"]) for k in keys])
    ax.set_ylabel("test WER")
    ax.set_ylim(0, max(m + s for m, s in zip(means, sds)) + 0.075)
    ax.set_xlim(-0.62, 3.97)
    ax.set_title("(a) Re-targeting the same system, identical grammar")


def panel_b(ax, data: dict) -> None:
    width, offs = 0.26, (-0.27, 0.0, 0.27)
    for off, key, color in zip(offs, PANEL_B, [NEUT, GOLD, ALIGN]):
        ax.bar([i + off for i in range(len(FIELDS))],
               [data[key][f] for f, _ in FIELDS], width, color=color,
               edgecolor="white", linewidth=0.6, label=data[key]["label"])

    ax.set_xticks(range(len(FIELDS)))
    ax.set_xticklabels([lbl for _, lbl in FIELDS])
    ax.set_ylabel("field accuracy (%)")
    ax.set_ylim(0, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("(b) Field accuracy at the tuned seed", pad=17)
    ax.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.005),
              handlelength=1.0, borderpad=0.1, labelspacing=0.25,
              handletextpad=0.4, columnspacing=1.2)

    # The year field is where the whole representation effect lives.
    y0 = data["characters"]["year"]
    y1 = data["phonemes"]["year"]
    ax.annotate("", xy=(1 + offs[2], y1 + 4), xytext=(1 + offs[0], y0 + 4),
                arrowprops=dict(arrowstyle="-|>", color=RED_TX, lw=1.1,
                                connectionstyle="arc3,rad=-0.42",
                                shrinkA=1.5, shrinkB=1.5))
    # Truncate rather than round: the manuscript quotes these fields as 42%
    # and 89%, and 89.5 would otherwise print as 90%.
    ax.annotate(r"$\mathbf{{{:d}\%\rightarrow{:d}\%}}$".format(int(y0), int(y1)),
                (1 + offs[2] + 0.34, y1 - 24), ha="left", va="center",
                fontsize=7.4, color=RED_TX, fontweight="bold")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(ROOT / "Results_Current"))
    ap.add_argument("--out",
                    default=str(ROOT / "TASLP_Silent_Speech_Recognition" /
                                "Figures"))
    args = ap.parse_args()

    _style()
    data = read_rows(Path(args.results) / "fig_representation.csv")

    fig, axes = plt.subplots(2, 1, figsize=(COL1, 3.62),
                             height_ratios=[1.0, 0.92],
                             constrained_layout=True)
    panel_a(axes[0], data)
    panel_b(axes[1], data)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_rgb(fig, str(out / "Target_Representation.png"), dpi=DPI)


if __name__ == "__main__":
    main()
