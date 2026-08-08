#!/usr/bin/env python3
"""
scripts/evaluate.py

Turn decoding outputs (*.jsonl written by inference_emg_llm.py) into the
tables reported in the paper.

Examples
--------
# Table II: decoding pipeline on the primary backbone
python scripts/evaluate.py Results/"Silent Speech Test_Decoders"/*.jsonl

# Table V: per-field accuracy. Sub/Del/Ins is degenerate on this corpus (every
# reference is a 3-word date or a 2-word time, so Del = Ins = 0 and Sub = WER
# for every system); --fields is what actually separates them.
python scripts/evaluate.py --fields --decomp Results_reproduced/decomp/*.jsonl

# Table IV + Fig. 3: NS ablations with 95% bootstrap CIs
python scripts/evaluate.py --bootstrap Results/Ablation_NS_Decoding/*.jsonl

# Compare two systems with a paired bootstrap test
python scripts/evaluate.py --compare a.jsonl b.jsonl

All WER/CER numbers are corpus level (scripts/metrics.py).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    from scripts.data_utils import TextTransform
    from scripts.metrics import (FIELD_ORDER, bootstrap_ci, field_breakdown,
                                 paired_bootstrap_pvalue, score_corpus)
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_utils import TextTransform
    from metrics import (FIELD_ORDER, bootstrap_ci, field_breakdown,
                         paired_bootstrap_pvalue, score_corpus)


def load_jsonl(path: Path, field: str = "final") -> Tuple[List[str], List[str], List[str]]:
    """Return (utt_ids, refs, hyps) from a decoding output file."""
    utts: List[str] = []
    refs: List[str] = []
    hyps: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            utts.append(rec.get("utt", ""))
            refs.append(rec.get("ref", ""))
            hyps.append(rec.get(field, rec.get("final", "")))
    return utts, refs, hyps


def label_for(path: Path, refs_field: str) -> str:
    return path.stem


def main() -> None:
    ap = argparse.ArgumentParser(description="Score decoding outputs into paper tables.")
    ap.add_argument("inputs", nargs="+", help="One or more *.jsonl decoding outputs.")
    ap.add_argument("--field", default="final", help="JSON field holding the hypothesis.")
    ap.add_argument("--decomp", action="store_true",
                    help="Character-level Sub/Del/Ins, normalised so they sum to CER (Table V).")
    ap.add_argument("--decomp_words", action="store_true",
                    help="Word-level Sub/Del/Ins. Degenerate on this corpus (Del = Ins = 0 for "
                         "every system, because every reference is a 3-word date or 2-word time); "
                         "kept only to substantiate that claim.")
    ap.add_argument("--fields", action="store_true",
                    help="Per-slot accuracy and the numeric/alphabetic split (Table V). "
                         "Sub/Del/Ins is degenerate on this fixed-arity corpus; this is "
                         "the decomposition that discriminates between systems.")
    ap.add_argument("--bootstrap", action="store_true", help="Add 95%% bootstrap CIs for WER (Fig. 3).")
    ap.add_argument("--n_resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--csv", default="", help="Also write the table to this CSV path.")
    ap.add_argument("--compare", nargs=2, default=None,
                    help="Two jsonl files: paired bootstrap p-value for equal WER.")
    args = ap.parse_args()

    tt = TextTransform()
    paths = [Path(p) for p in args.inputs]
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(p)

    rows: List[Dict[str, object]] = []
    for p in sorted(paths):
        _, refs, hyps = load_jsonl(p, args.field)
        if not refs:
            print(f"  (skipping empty file {p})")
            continue
        s = score_corpus(refs, hyps, tt)
        row: Dict[str, object] = {
            "system": label_for(p, args.field),
            "n": s.n_utt,
            "WER": s.wer,
            "CER": s.cer,
            "EM%": s.exact_match,
        }
        if args.decomp:
            row.update({"cSub": s.csub, "cDel": s.cdel, "cIns": s.cins})
        if args.decomp_words:
            row.update({"Sub": s.sub, "Del": s.dele, "Ins": s.ins})
        if args.fields:
            fb = field_breakdown(refs, hyps, tt)
            if fb.n_off_shape_refs:
                print(f"  (note: {fb.n_off_shape_refs} reference(s) in {p.name} match "
                      f"neither the date nor the time template and are excluded)")
            fd = fb.as_dict()
            row.update({k: v for k, v in fd.items() if k in FIELD_ORDER})
            row.update({"num%": fd["numeric"], "alpha%": fd["alpha"], "shape%": fd["shape"]})
        if args.bootstrap:
            _, lo, hi = bootstrap_ci(refs, hyps, tt, n_resamples=args.n_resamples, seed=args.seed)
            row.update({"WER_lo": lo, "WER_hi": hi})
        # Table VI: fold in the timing sidecar inference wrote next to the jsonl,
        # so latency lands in the CSV instead of only in the SLURM log.
        side = p.with_suffix(".latency.json")
        if side.exists():
            lat = json.loads(side.read_text(encoding="utf-8"))
            row.update({"ms/utt": lat.get("ms_per_utt"),
                        "chars/s": lat.get("chars_per_s"),
                        "peakGB": lat.get("peak_gpu_gb")})
        rows.append(row)

    if not rows:
        print("Nothing to score.")
        return

    # ---- pretty print ----------------------------------------------------
    # Union the keys rather than trusting rows[0]: a system that never emits a
    # given slot yields a shorter row, which would otherwise silently drop the
    # column for everyone and make DictWriter raise on the extra keys.
    cols: List[str] = []
    for r in rows:
        cols.extend(k for k in r if k not in cols)
    for r in rows:
        for c in cols:
            r.setdefault(c, float("nan"))
    widths = {c: max(len(c), max(len(_fmt(r[c], c)) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    print(header)
    print("-" * len(header))
    for r in rows:
        print("  ".join(_fmt(r[c], c).ljust(widths[c]) for c in cols))

    if args.csv:
        out = Path(args.csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {out}")

    # ---- paired comparison ----------------------------------------------
    if args.compare:
        pa, pb = Path(args.compare[0]), Path(args.compare[1])
        _, refs_a, hyps_a = load_jsonl(pa, args.field)
        _, refs_b, hyps_b = load_jsonl(pb, args.field)
        if refs_a != refs_b:
            print("\nWARNING: the two files do not share identical references; "
                  "the paired test assumes the same utterances in the same order.")
        pval = paired_bootstrap_pvalue(
            refs_a, hyps_a, hyps_b, tt, n_resamples=args.n_resamples, seed=args.seed
        )
        wa = score_corpus(refs_a, hyps_a, tt).wer
        wb = score_corpus(refs_b, hyps_b, tt).wer
        print(f"\nPaired bootstrap ({args.n_resamples} resamples)")
        print(f"  {pa.stem}: WER {wa:.4f}")
        print(f"  {pb.stem}: WER {wb:.4f}")
        print(f"  delta = {wa - wb:+.4f}, p = {pval:.4f}")


_PCT_COLS = set(FIELD_ORDER) | {"num%", "alpha%", "shape%"}
# Only the latency suite writes these; every other table leaves them empty.
_LAT_COLS = {"ms/utt": 1, "chars/s": 0, "peakGB": 2}


def _fmt(v: object, col: str) -> str:
    if col in ("system",):
        return str(v)
    if col in ("n",):
        return str(v)
    if col in _LAT_COLS:
        if v is None:
            return "-"
        f = float(v)
        return "-" if f != f else f"{f:.{_LAT_COLS[col]}f}"
    if col == "EM%":
        return f"{float(v):.0f}"
    if col in _PCT_COLS:
        f = float(v)
        return "-" if f != f else f"{f:.0f}"  # NaN when the slot never occurs
    if isinstance(v, float):
        return f"{v:.4f}" if col.startswith("WER") or col == "CER" else f"{v:.3f}"
    return str(v)


if __name__ == "__main__":
    main()
