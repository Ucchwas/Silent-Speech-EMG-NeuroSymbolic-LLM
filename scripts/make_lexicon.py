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


def build_lexicon(src_dirs: List[Path], min_count: int = 1) -> Tuple[List[str], Counter]:
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

    vocab = [w for w, c in counts.items() if c >= int(min_count)]
    vocab = sorted(set(vocab))  # deterministic
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
    args = ap.parse_args()

    src_dirs = [Path(s) for s in args.src]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lexicon, counts = build_lexicon(src_dirs, min_count=args.min_count)

    out_path.write_text("\n".join(lexicon) + ("\n" if lexicon else ""), encoding="utf-8")

    meta = {
        "src_dirs": [str(p) for p in src_dirs],
        "min_count": int(args.min_count),
        "num_unique_words": len(lexicon),
        "top_50_words": counts.most_common(50),
    }
    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"[make_lexicon] wrote: {out_path}  ({len(lexicon)} words)")
    print(f"[make_lexicon] wrote: {meta_path}")


if __name__ == "__main__":
    main()