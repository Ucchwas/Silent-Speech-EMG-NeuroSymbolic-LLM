#!/usr/bin/env python3
"""
scripts/analyze_pool.py

Two analyses over decoding outputs that the per-decoder tables cannot express.

1. **Pool oracle.** `ctc_grammar` keeps the top-N grammar-valid strings from the
   CTC prefix beam and reports the first. If the reference is often further down
   that list, a reranker has something to find and the gap is its ceiling; if the
   oracle sits on top of the top-1 WER, no reranker can help and that ceiling is
   itself the result to report.

2. **Channel complementarity.** Grammar-validity rate per decoder, pairwise
   agreement, strict-win counts, and the two-system oracle. This is the evidence
   that symbolic validity constraints are vacuous on the AR channel: the frozen
   LLM's beam is already 100% well-formed, so a validity constraint has nothing
   left to exclude there.

Usage
-----
python scripts/analyze_pool.py Results_reproduced/decoders/ctc_grammar.jsonl
python scripts/analyze_pool.py --channels Results_reproduced/decoders
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

try:
    from scripts.data_utils import TextTransform
    from scripts.grammar import is_valid
    from scripts.metrics import score_corpus
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_utils import TextTransform
    from grammar import is_valid
    from metrics import score_corpus


def load(path: Path) -> List[dict]:
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _wer1(ref: str, hyp: str, tt) -> float:
    return score_corpus([ref], [hyp], tt).wer


def pool_oracle(recs: Sequence[dict], tt) -> None:
    """Oracle over the grammar-valid candidate pool, and where the best one sits."""
    refs = [r["ref"] for r in recs]
    pools: List[List[str]] = []
    for r in recs:
        p = (r.get("ns_info", {}).get("ctc_grammar", {}) or {}).get("pool") or []
        pools.append([t for t, _ in p])

    sizes = [len(p) for p in pools]
    if not any(sizes):
        print("  no pool recorded -- rerun with --ctc_grammar_pool > 1")
        return

    top1 = [p[0] if p else "" for p in pools]
    best = [min(p, key=lambda t: _wer1(r, t, tt)) if p else ""
            for r, p in zip(refs, pools)]

    s1, so = score_corpus(refs, top1, tt), score_corpus(refs, best, tt)
    print(f"  pool size          mean {sum(sizes)/len(sizes):.2f}  max {max(sizes)}")
    print(f"  top-1              WER {s1.wer:.4f}  CER {s1.cer:.4f}  EM {s1.exact_match:.0f}%")
    print(f"  pool oracle        WER {so.wer:.4f}  CER {so.cer:.4f}  EM {so.exact_match:.0f}%")
    print(f"  headroom           {s1.wer - so.wer:+.4f} WER "
          f"({100 * (s1.wer - so.wer) / max(s1.wer, 1e-9):.1f}% relative)")

    # Rank of the first pool entry that beats the top-1: tells you how deep a
    # reranker would have to look, not just whether anything better exists.
    improves = [next((i for i, t in enumerate(p)
                      if _wer1(r, t, tt) < _wer1(r, p[0], tt)), None)
                for r, p in zip(refs, pools) if p]
    hit = [i for i in improves if i is not None]
    print(f"  utterances where the pool holds something better: {len(hit)}/{len(improves)}"
          + (f"  (median rank {sorted(hit)[len(hit)//2]})" if hit else ""))
    exact = sum(1 for r, p in zip(refs, pools) if r in p)
    print(f"  reference present anywhere in the pool: {exact}/{len(refs)}")


def channels(recs: Sequence[dict], cols: Sequence[str], tt,
             csv_out: Path | None = None) -> None:
    """Validity rates, agreement, strict wins, and the multi-system oracle."""
    refs = [r["ref"] for r in recs]
    have = [c for c in cols if any(c in r for r in recs)]
    if not have:
        print("  none of the requested decoder columns are present")
        return

    print(f"  {'decoder':<16} {'WER':>7} {'grammar-valid':>14}")
    hyps: Dict[str, List[str]] = {}
    table: List[dict] = []
    for c in have:
        h = [r.get(c, "") for r in recs]
        hyps[c] = h
        v = sum(is_valid(x) for x in h)
        s = score_corpus(refs, h, tt)
        table.append({"system": c, "n": len(h), "WER": s.wer,
                      "valid": v, "valid_pct": 100.0 * v / max(1, len(h))})
        print(f"  {c:<16} {s.wer:>7.4f} {v:>8}/{len(h):<5}")

    if csv_out is not None:
        import csv as _csv

        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
        print(f"\n  wrote {csv_out}")

    print()
    for i, a in enumerate(have):
        for b in have[i + 1:]:
            ha, hb = hyps[a], hyps[b]
            agree = sum(1 for x, y in zip(ha, hb) if x == y)
            wa = sum(1 for r, x, y in zip(refs, ha, hb)
                     if _wer1(r, x, tt) < _wer1(r, y, tt))
            wb = sum(1 for r, x, y in zip(refs, ha, hb)
                     if _wer1(r, y, tt) < _wer1(r, x, tt))
            orc = [x if _wer1(r, x, tt) <= _wer1(r, y, tt) else y
                   for r, x, y in zip(refs, ha, hb)]
            print(f"  {a} vs {b}: agree {agree}/{len(refs)}, "
                  f"strict wins {wa} vs {wb}, "
                  f"oracle WER {score_corpus(refs, orc, tt).wer:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="jsonl file(s), or a directory of them.")
    ap.add_argument("--csv", default="",
                    help="Write the validity table here (feeds Fig. 3a).")
    ap.add_argument("--channels", nargs="*", default=None,
                    help="Decoder columns to cross-compare. Defaults to the "
                         "AR/CTC pair plus the constrained variants.")
    args = ap.parse_args()

    tt = TextTransform()
    paths: List[Path] = []
    for s in args.inputs:
        p = Path(s)
        paths.extend(sorted(p.glob("*.jsonl")) if p.is_dir() else [p])

    default_cols = ["ar_beam", "ctc_greedy", "ctc_lexicon", "ctc_grammar", "ctc_grammar_ar"]
    cols = args.channels if args.channels is not None else default_cols
    csv_out = Path(args.csv) if args.csv else None

    if len(paths) == 1:
        recs = load(paths[0])
        print(f"== {paths[0].name}  (n = {len(recs)})")
        pool_oracle(recs, tt)
        print()
        channels(recs, cols, tt, csv_out)
        return

    # A directory of one-decoder-per-file outputs: stitch the columns together
    # by utterance id so the same cross-comparison still works.
    byutt: Dict[str, dict] = {}
    for p in paths:
        name = p.stem
        for r in load(p):
            row = byutt.setdefault(r["utt"], {"ref": r["ref"]})
            row[name] = r.get("final", "")
            if name == "ctc_grammar" and "ns_info" in r:
                row.setdefault("ns_info", r["ns_info"])
    recs = list(byutt.values())
    print(f"== {len(paths)} files, {len(recs)} utterances")
    pool_oracle(recs, tt)
    print()
    channels(recs, cols, tt, csv_out)


if __name__ == "__main__":
    main()
