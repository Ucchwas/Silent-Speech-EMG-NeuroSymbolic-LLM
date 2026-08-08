#!/usr/bin/env python3
"""
scripts/pptx_text.py

Turn the matplotlib mathtext used in the figure labels into something
PowerPoint can show.

The figure labels are authored once, in mathtext, because the manuscript is the
primary deliverable and its symbols have to match the equations. PowerPoint has
no mathtext, so the deck needs the same strings as Unicode with real sub- and
superscript runs. Doing that conversion here, rather than keeping a second set
of hand-written labels in the deck, is what stops the two from drifting apart.

`split_runs("$\\lambda_{\\mathrm{CTC}}=0.2$")` gives
`[("λ", 0), ("CTC", -1), ("=0.2", 0)]`, where the second element is 0 for
baseline, -1 for subscript and +1 for superscript.

Anything unrecognised is passed through unchanged and reported by
`unknown_commands()`, so a symbol added to a label later cannot silently reach
the deck as a raw backslash command.
"""

from __future__ import annotations

import re
from typing import List, Set, Tuple

# Only the commands the figure labels actually use. Keeping this closed, rather
# than reaching for a general LaTeX engine, means an unmapped symbol is a loud
# failure instead of a stray "\lambda" printed on a slide.
SYMBOL = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "phi": "ϕ", "sigma": "σ",
    "theta": "θ", "epsilon": "ε",
    "times": "×", "cdot": "·", "rightarrow": "→", "to": "→",
    "in": "∈", "leq": "≤", "geq": "≥", "approx": "≈", "pm": "±",
    "ldots": "…", "lfloor": "⌊", "rfloor": "⌋", "varnothing": "∅",
    "quad": "  ", "qquad": "    ", ",": " ", ";": " ", " ": " ", "%": "%",
}

# Script and blackboard letters, so the deck shows the same glyph the equations
# do rather than a plain capital.
CAL = {"L": "ℒ", "G": "𝒢", "H": "ℋ", "D": "𝒟", "R": "ℛ", "C": "𝒞",
       "V": "𝒱", "T": "𝒯", "W": "𝒲"}
BBB = {"R": "ℝ", "N": "ℕ", "Z": "ℤ", "Q": "ℚ"}

_TOKEN = re.compile(
    r"\\(?P<cmd>[A-Za-z]+|[,;%\s])"   # \lambda, \times, \, ...
    r"|(?P<sub>_)"
    r"|(?P<sup>\^)"
    r"|(?P<open>\{)"
    r"|(?P<close>\})"
    r"|(?P<other>.)",
    re.S)

_UNKNOWN: Set[str] = set()


def unknown_commands() -> Set[str]:
    """Commands seen since import that had no mapping."""
    return set(_UNKNOWN)


def _emit(runs: List[Tuple[str, int]], s: str, base: int) -> None:
    if not s:
        return
    if runs and runs[-1][1] == base:
        runs[-1] = (runs[-1][0] + s, base)
    else:
        runs.append((s, base))


def _convert_math(body: str, runs: List[Tuple[str, int]], base: int) -> None:
    """Convert one $...$ body into runs, appending to `runs`."""
    i, n = 0, len(body)
    # `stack` holds the baseline to return to when a { } group closes.
    stack: List[int] = []
    pending = 0  # baseline that the next atom or group should take

    while i < n:
        m = _TOKEN.match(body, i)
        i = m.end()

        if m.group("cmd") is not None:
            cmd = m.group("cmd")
            cur = base + sum(stack) + pending
            if len(cmd) > 1:
                # In LaTeX the space after a multi-letter command only ends the
                # name; emitting it would put a gap in front of every symbol.
                while i < n and body[i] == " ":
                    i += 1
            if cmd in ("mathcal", "mathbb", "mathbf", "mathrm", "mathit",
                       "mathsf", "text", "textrm", "operatorname"):
                # Read the braced argument and map it letter by letter.
                if i < n and body[i] == "{":
                    depth, j = 1, i + 1
                    while j < n and depth:
                        depth += (body[j] == "{") - (body[j] == "}")
                        j += 1
                    arg, i = body[i + 1:j - 1], j
                else:
                    arg, i = body[i], i + 1
                table = {"mathcal": CAL, "mathbb": BBB}.get(cmd)
                _emit(runs, "".join(table.get(c, c) for c in arg) if table
                      else arg, cur)
            elif cmd in SYMBOL:
                _emit(runs, SYMBOL[cmd], cur)
            else:
                _UNKNOWN.add("\\" + cmd)
                _emit(runs, cmd, cur)
            pending = 0

        elif m.group("sub") is not None:
            pending = -1
        elif m.group("sup") is not None:
            pending = +1
        elif m.group("open") is not None:
            stack.append(pending)
            pending = 0
        elif m.group("close") is not None:
            if stack:
                stack.pop()
            pending = 0
        else:
            ch = m.group("other")
            if ch == "'":
                _emit(runs, "′", base + sum(stack) + pending)
            else:
                _emit(runs, ch, base + sum(stack) + pending)
            pending = 0


def split_runs(s: str) -> List[Tuple[str, int]]:
    """Split a mathtext-bearing label into (text, baseline) runs."""
    runs: List[Tuple[str, int]] = []
    for k, part in enumerate(s.split("$")):
        if not part:
            continue
        if k % 2 == 0:            # outside $...$
            _emit(runs, part, 0)
        else:                     # inside $...$
            _convert_math(part, runs, 0)
    return runs


def to_plain(s: str) -> str:
    """Flatten to a single string, for logging and for width estimation."""
    return "".join(t for t, _ in split_runs(s))


if __name__ == "__main__":
    for probe in [
        "$\\mathcal{L}=\\mathcal{L}_{\\mathrm{AR}}+\\lambda_{\\mathrm{CTC}}\\,"
        "\\mathcal{L}_{\\mathrm{CTC}}$,    $\\lambda_{\\mathrm{CTC}}=0.2$",
        "$\\mathbf{E}\\in\\mathbb{R}^{T\\times112}$",
        "$\\mathbf{H}^{\\mathrm{emg}}\\in\\mathbb{R}^{T'\\times d}$,"
        "  $T'=\\lfloor T/4\\rfloor$",
        "EMG adapter $G_\\phi$",
        "conv + Transformer, $\\times4$ subsampling",
    ]:
        print(f"{probe}\n  -> {to_plain(probe)}\n     {split_runs(probe)}")
    print("unknown:", unknown_commands() or "none")
