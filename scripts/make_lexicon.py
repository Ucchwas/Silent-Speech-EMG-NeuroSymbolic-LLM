#!/usr/bin/env python3
"""
make_lexicon.py

Build a closed-vocabulary word lexicon from transcripts.
Used by NeuroSymbolic decoding (trie constraint + boundary scoring).

Expected layout (from split_data.py):
  data/train_emg/*.json  (+ optional paired *.npy)
  data/val_emg/*.json
  data/test_emg/*.json

Default behavior (paper-faithful):
  Build lexicon from TRAIN only: --src data/train_emg

Outputs:
  artifacts/lexicon.txt       (one word per line, sorted)
  artifacts/lexicon.meta.json (counts + summary)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

# Robust import (repo may have scripts/ or root-level layout)
try:
    from scripts.data_utils import TextTransform
except Exception:
    from data_utils import TextTransform


TEXT_KEYS = ("text", "transcript", "sentence", "label")


def _read_text(js_path: Path) -> str:
    """Read transcript text from a JSON sidecar with robust key fallback."""
    data: Dict[str, Any] = json.loads(js_path.read_text(encoding="utf-8"))

    for k in TEXT_KEYS:
        v = data.get(k, None)
        if isinstance(v, str) and v.strip():
            return v

    for v in data.values():
        if isinstance(v, str) and v.strip():
            return v

    raise ValueError(f"No transcript field found in {js_path}")


def _iter_jsons(src_dirs: Iterable[Path]) -> Iterable[Path]:
    """Yield JSON files from directories (deduplicated, deterministic)."""
    seen = set()
    for d in src_dirs:
        d = Path(d)
        if not d.exists():
            continue

        for js in sorted(d.glob("*.json")):
            rp = js.resolve()
            if rp not in seen:
                seen.add(rp)
                yield js

        # Also allow sidecar JSONs for NPY pairs
        for npy in sorted(d.glob("*.npy")):
            js = npy.with_suffix(".json")
            if js.exists():
                rp = js.resolve()
                if rp not in seen:
                    seen.add(rp)
                    yield js


def expand_numeral_classes(words: Iterable[str]) -> List[str]:
    """
    Close the *numeral classes* observed in training, instead of memorising the
    exact numeral strings that happened to occur.

    The benchmark is template-generated ("<time> am|pm", "<month> <day> <year>",
    "<weekday> <month> <day>"), so the task's closed vocabulary is weekdays,
    months, am/pm and three numeral classes. Observing "0112" and "0115" in
    training tells you the class is hh:mm; it does not make "0113" out of
    vocabulary.

    Emitted classes, inferred from what training actually contains:
      * 2-digit -> every value in the observed range (days 01-31)
      * 4-digit -> every clock time hhmm with hh in 01..12 and mm in 00..59,
                   plus every year in the observed year range

    Without this, a strictly observed-token lexicon leaves ~17% of held-out
    words outside the trie, which caps trie-constrained exact-match far below
    the values reported in the paper.
    """
    words = set(words)
    out = {w for w in words if not w.isdigit()}

    d2 = [w for w in words if w.isdigit() and len(w) == 2]
    d4 = [w for w in words if w.isdigit() and len(w) == 4]

    if d2:
        lo, hi = min(int(w) for w in d2), max(int(w) for w in d2)
        out |= {f"{i:02d}" for i in range(lo, hi + 1)}

    if d4:
        out |= {f"{h:02d}{m:02d}" for h in range(1, 13) for m in range(60)}
        years = [int(w) for w in d4 if not (1 <= int(w[:2]) <= 12 and int(w[2:]) <= 59)]
        if years:
            out |= {str(y) for y in range(min(years), max(years) + 1)}

    out |= words  # never drop an actually observed token
    return sorted(out)


def build_lexicon(
    src_dirs: List[Path], min_count: int = 1, expand_numerals: bool = False
) -> Tuple[List[str], Counter]:
    """
    Build a word lexicon using TextTransform.clean() (paper-faithful normalization).
    """
    tt = TextTransform()
    counts: Counter = Counter()

    for js_path in _iter_jsons(src_dirs):
        txt = _read_text(js_path)
        txt = tt.clean(txt)
        for w in txt.split():
            if w:
                counts[w] += 1

    vocab = sorted({w for w, c in counts.items() if c >= int(min_count)})
    if expand_numerals:
        vocab = expand_numeral_classes(vocab)
    return vocab, counts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--src",
        nargs="+",
        default=["data/train_emg"],
        help="One or more directories containing transcripts (*.json). For paper-faithful lexicon, pass TRAIN only.",
    )
    ap.add_argument(
        "--out",
        default="artifacts/lexicon.txt",
        help="Output lexicon path (one word per line).",
    )
    ap.add_argument(
        "--min_count",
        type=int,
        default=1,
        help="Keep words that appear at least this many times across --src.",
    )
    ap.add_argument(
        "--expand_numerals",
        action="store_true",
        help="Close the numeral classes seen in training (days, clock times, years) "
             "instead of keeping only the exact numeral strings observed. See "
             "expand_numeral_classes() for what this emits and why it matters.",
    )
    args = ap.parse_args()

    src_dirs = [Path(s) for s in args.src]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lexicon, counts = build_lexicon(
        src_dirs, min_count=args.min_count, expand_numerals=args.expand_numerals
    )

    out_path.write_text("\n".join(lexicon) + ("\n" if lexicon else ""), encoding="utf-8")

    meta = {
        "src_dirs": [str(p) for p in src_dirs],
        "min_count": int(args.min_count),
        "expand_numerals": bool(args.expand_numerals),
        "num_observed_words": len(counts),
        "num_unique_words": len(lexicon),
        "top_50_words": counts.most_common(50),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[make_lexicon] wrote: {out_path}  ({len(lexicon)} words)")
    print(f"[make_lexicon] wrote: {meta_path}")


if __name__ == "__main__":
    main()