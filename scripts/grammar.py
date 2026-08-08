"""
Utterance-level grammar for the closed date/time benchmark.

The flat word lexicon is a very weak constraint on this corpus: 889 of its 910
entries are numeral literals, admitting 31% of all 2-digit and 8.6% of all
4-digit strings. Worse, a word-level trie resets at every space, so it accepts
any word in any position -- "1045 am am" and "august 06 1925 august" are both
legal. Measured on the test split, disabling the trie entirely changed nothing
(wo_trie 0.4470 == ns_full 0.4470).

This module replaces it with a trie over *complete valid utterances*, which
enforces structure across the whole transcript rather than word by word.

Two rules govern how the grammar is built, and both matter:

1. **Templates are induced from TRAIN transcripts only.** The three observed
   there -- MONTH DD YYYY, WEEKDAY MONTH DD, HHMM {am,pm} -- cover 100% of the
   validation and test references as well.

2. **Fields use semantic ranges, never observed literals.** Train contains only
   89 distinct years, and 4 validation years (1904, 1958, 1969, 2013) do not
   appear among them. Enumerating observed values would make those references
   unreachable -- a grammar that scores perfectly on train and silently destroys
   held-out data. Years therefore span the full train min..max interval
   (1883-2020), days respect month length, hours are 01-12 and minutes 00-59.

   Both the rule and its bounds come from train alone: `induce` reads only the
   training transcripts, and validation is used solely to confirm the rule was
   necessary. The test split is never inspected to build or widen the grammar.

`verify_coverage` re-checks property 2 against any split and is called by the
lexicon build step, so an over-tight grammar fails loudly instead of quietly.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

MONTHS: List[str] = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]
WEEKDAYS: List[str] = [
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
]
# February is given 29 so leap dates remain generatable.
DAYS_IN_MONTH: Dict[str, int] = {
    "january": 31, "february": 29, "march": 31, "april": 30, "may": 31, "june": 30,
    "july": 31, "august": 31, "september": 30, "october": 31, "november": 30, "december": 31,
}

DEFAULT_YEAR_MIN, DEFAULT_YEAR_MAX = 1883, 2020


# -----------------------------------------------------------------------------
# Template induction (train only)
# -----------------------------------------------------------------------------
def _shape(text: str) -> str:
    out = []
    for w in text.split():
        if re.fullmatch(r"\d+", w):
            out.append(f"D{len(w)}")
        elif w in MONTHS:
            out.append("MON")
        elif w in WEEKDAYS:
            out.append("WD")
        elif w in ("am", "pm"):
            out.append("MER")
        else:
            out.append("?")
    return " ".join(out)


def induce(texts: Iterable[str]) -> Tuple[Set[str], int, int]:
    """Return (template shapes, year_min, year_max) observed in `texts`."""
    shapes: Set[str] = set()
    years: List[int] = []
    for t in texts:
        sh = _shape(t)
        shapes.add(sh)
        w = t.split()
        if sh == "MON D2 D4":
            years.append(int(w[2]))
    ymin = min(years) if years else DEFAULT_YEAR_MIN
    ymax = max(years) if years else DEFAULT_YEAR_MAX
    return shapes, ymin, ymax


# -----------------------------------------------------------------------------
# Enumeration
# -----------------------------------------------------------------------------
def generate(shapes: Set[str], year_min: int, year_max: int) -> List[str]:
    """Every transcript the grammar admits, as normalised text."""
    out: List[str] = []
    if "MON D2 D4" in shapes:
        for m in MONTHS:
            for d in range(1, DAYS_IN_MONTH[m] + 1):
                for y in range(year_min, year_max + 1):
                    out.append(f"{m} {d:02d} {y}")
    if "WD MON D2" in shapes:
        for wd in WEEKDAYS:
            for m in MONTHS:
                for d in range(1, DAYS_IN_MONTH[m] + 1):
                    out.append(f"{wd} {m} {d:02d}")
    if "D4 MER" in shapes:
        for h in range(1, 13):
            for mi in range(0, 60):
                for mer in ("am", "pm"):
                    out.append(f"{h:02d}{mi:02d} {mer}")
    return out


def is_valid(text: str, year_min: int = DEFAULT_YEAR_MIN, year_max: int = DEFAULT_YEAR_MAX) -> bool:
    """Cheap membership test that does not require enumerating the grammar."""
    w = text.split()
    if len(w) == 2 and w[1] in ("am", "pm"):
        t = w[0]
        return bool(re.fullmatch(r"\d{4}", t)) and 1 <= int(t[:2]) <= 12 and 0 <= int(t[2:]) <= 59
    if len(w) == 3 and w[0] in MONTHS:
        d, y = w[1], w[2]
        return bool(
            re.fullmatch(r"\d{2}", d) and 1 <= int(d) <= DAYS_IN_MONTH[w[0]]
            and re.fullmatch(r"\d{4}", y) and year_min <= int(y) <= year_max
        )
    if len(w) == 3 and w[0] in WEEKDAYS and w[1] in MONTHS:
        d = w[2]
        return bool(re.fullmatch(r"\d{2}", d) and 1 <= int(d) <= DAYS_IN_MONTH[w[1]])
    return False


# -----------------------------------------------------------------------------
# I/O helpers
# -----------------------------------------------------------------------------
def read_split(tt, data_dir: str | Path) -> List[str]:
    """Normalised transcripts for a split, using the same clean() as training."""
    out = []
    for js in sorted(Path(data_dir).glob("*.json")):
        out.append(tt.clean(json.load(open(js))["text"]))
    return out


def verify_coverage(texts: Sequence[str], year_min: int, year_max: int) -> List[str]:
    """Transcripts the grammar cannot generate. Must be empty for every split."""
    return [t for t in texts if not is_valid(t, year_min, year_max)]
