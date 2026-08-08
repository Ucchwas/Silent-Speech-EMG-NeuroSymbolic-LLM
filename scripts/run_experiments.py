#!/usr/bin/env python3
"""
scripts/run_experiments.py

Drives every experiment reported in the paper and writes the results into a
directory layout that scripts/evaluate.py can score directly.

Suites
------
decoders    Table II  : the seven decoding conditions on the primary backbone
supervision Table III : AR+CTC vs AR-only vs CTC-only, same inference recipe
ablation    Table IV  : NS component ablations (trie / 5-gram / boundary / EOS)
decomp      Table V   : WER decomposition inputs (AR beam, NS, NS+adaptive)
latency     Table VI  : decoding latency and peak GPU memory
backbones   Fig. 2    : the same recipe across frozen LLM backbones
sweep_beta  Fig. 4    : validation WER vs the char 5-gram fusion weight beta
sweep_lam   Fig. 5    : validation WER vs the fixed AR/CTC fusion weight lambda

Examples
--------
python scripts/run_experiments.py --suite decoders --ckpt artifacts/best_checkpoint.pt
python scripts/run_experiments.py --suite ablation --data_dir data/test_emg
python scripts/run_experiments.py --suite sweep_beta --data_dir data/val_emg
python scripts/run_experiments.py --suite backbones \
    --backbone_ckpt llama32_1b=artifacts/best_llama1b.pt \
    --backbone_ckpt llama32_3b=artifacts/best_checkpoint.pt

Hyperparameter suites deliberately default to `data/val_emg`: the paper tunes
on validation and freezes the settings for a single test-set evaluation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence

REPO = Path(__file__).resolve().parent.parent
INFER = REPO / "inference_emg_llm.py"

# Table I final values, used unless artifacts/ns_tuning.json overrides them.
#
# min_eos_len is the one that matters most here. Table I lists m = 8, but the
# shortest references in this corpus are the 7-character time expressions
# ("1045 am"), so m = 8 makes EOS unreachable for 18 of the 50 test utterances:
# the constrained beam cannot terminate and appends a spurious lexicon word
# ("0948 am am"). Measured on the test split, NS emitted an extra word in 18/18
# of those utterances and 0/32 of the longer ones. The value is therefore tuned
# on validation by scripts/tune_ns.py rather than taken from the table.
TUNED_PATH = REPO / "artifacts" / "ns_tuning.json"

_DEFAULTS = {
    "alpha": 0.6, "delta": 5.0, "beta": 0.55, "kappa": 0.40, "gamma": 0.45,
    "min_eos_len": 8, "lambda_fix": 0.25, "lambda0": 0.25, "a": 0.20, "b": 0.20,
    "lambda_min": 0.0, "lambda_max": 0.6, "ctc_select_lengthnorm": False,
    # Beam width is tuned, not fixed at the paper's K = 6. Under the utterance
    # grammar, K = 6 leaves a candidate pool of ~2.5 and the reference is
    # reachable only 40% of the time; widening the beam raises reachability to
    # 70% and takes AR+CTC reranking from 0.308 to 0.233 on validation.
    "beam": 6,
}


def _ns_params(path: Path | None = None) -> dict:
    """Table I values, overridden by whatever tune_ns.py selected on validation."""
    path = path or TUNED_PATH
    params = dict(_DEFAULTS)
    if path.exists():
        tuned = json.loads(path.read_text()).get("tuned", {})
        params.update({k: v for k, v in tuned.items() if k in params})
        print(f"[run_experiments] using tuned NS config from {path.name}: {tuned}")
    else:
        print(f"[run_experiments] WARNING: {path} not found; using Table I defaults "
              f"(min_eos_len=8 is known to break short utterances)")

    # The adaptive lambda is clipped to [lambda_min, lambda_max] at inference.
    # If that window excludes the tuned lambda0 / lambda_fix, ns_joint_adaptive
    # is silently pinned to the boundary and scores *worse* than the fixed
    # ns_joint_rerank it is supposed to generalise -- a ladder inversion that
    # looks like a modelling result and is really a config mismatch. Older
    # tuned files predate lambda_min/lambda_max being written out, so widen
    # rather than trust the default.
    lo = min(params["lambda_min"], params["lambda0"], params["lambda_fix"])
    hi = max(params["lambda_max"], params["lambda0"], params["lambda_fix"])
    if (lo, hi) != (params["lambda_min"], params["lambda_max"]):
        print(f"[run_experiments] widening lambda clip "
              f"[{params['lambda_min']}, {params['lambda_max']}] -> [{lo}, {hi}] "
              f"to contain lambda0={params['lambda0']} lambda_fix={params['lambda_fix']}")
        params["lambda_min"], params["lambda_max"] = lo, hi
    return params


def _base_flags(path: Path | None = None) -> List[str]:
    p = _ns_params(path)
    K = int(p["beam"])
    flags = [
        "--beam", str(K),
        "--max_len", "64",
        # The candidate pool and per-step expansion must scale with the beam,
        # otherwise a wider search is truncated back down before reranking.
        "--rerank_M", str(max(16, 4 * K)),
        "--per_step_top", str(max(50, 4 * K)),
        "--alpha", str(p["alpha"]),
        "--delta", str(p["delta"]),
        "--beta", str(p["beta"]),
        "--kappa", str(p["kappa"]),
        "--gamma", str(p["gamma"]),
        "--min_eos_len", str(int(p["min_eos_len"])),
        "--lambda_fix", str(p["lambda_fix"]),
        "--lambda0", str(p["lambda0"]),
        "--a", str(p["a"]),
        "--b", str(p["b"]),
        "--lambda_min", str(p["lambda_min"]),
        "--lambda_max", str(p["lambda_max"]),
    ]
    if p["ctc_select_lengthnorm"]:
        flags.append("--ctc_select_lengthnorm")
    return flags


BASE_FLAGS = _base_flags()

DECODERS = [
    "ctc_greedy",
    # The three constrained variants share ctc_greedy's posterior and differ
    # only in what is imposed on it: nothing, the flat word lexicon, the
    # utterance grammar, and the grammar plus AR rescoring. They sit next to
    # their unconstrained twin so Table II reads as a matched series.
    "ctc_lexicon",
    "ctc_grammar",
    "ctc_grammar_ar",
    "ar_greedy",
    "ar_beam",
    "ns",
    "ns_ctc_select",
    "ns_joint_rerank",
    "ns_joint_adaptive",
]

# name -> extra flags disabling one NS component (Table IV / Fig. 3)
ABLATIONS: Dict[str, List[str]] = {
    "ns_full": [],
    "wo_trie": ["--no-trie"],
    "wo_5gram": ["--beta", "0.0"],
    "wo_boundary": ["--kappa", "0.0", "--gamma", "0.0"],
    "wo_eos": ["--no-eos_gating"],
}


def run(cmd: Sequence[str], dry: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}")
    if dry:
        return
    res = subprocess.run([str(c) for c in cmd], cwd=str(REPO))
    if res.returncode != 0:
        raise SystemExit(f"command failed ({res.returncode}): {printable}")


def infer_cmd(
    ckpt: str,
    data_dir: str,
    out: Path,
    decoder: str,
    extra: Sequence[str] = (),
    common: Sequence[str] = (),
    base: Sequence[str] | None = None,
    dry: bool = False,
) -> List[str]:
    # Under --dry_run this must not touch the filesystem: an empty suite
    # directory left behind by a dry run makes decode.sbatch archive a
    # "previous" result tree that never existed.
    if not dry:
        out.parent.mkdir(parents=True, exist_ok=True)
    return [
        sys.executable, str(INFER),
        "--ckpt", ckpt,
        "--data_dir", data_dir,
        "--out", str(out),
        "--decoder", decoder,
        *(BASE_FLAGS if base is None else base),
        *common,
        *extra,
    ]


def parse_kv(items: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"expected name=path, got {it!r}")
        k, v = it.split("=", 1)
        out[k] = v
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the paper's experiment suites.")
    ap.add_argument("--suite", required=True,
                    choices=["decoders", "ablation", "decomp", "latency",
                             "backbones", "supervision", "sweep_beta",
                             "sweep_lam", "all"])
    ap.add_argument("--ckpt", default="artifacts/best_checkpoint.pt")
    ap.add_argument("--data_dir", default=None,
                    help="Defaults to data/test_emg for reporting suites, "
                         "data/val_emg for the tuning sweeps.")
    ap.add_argument("--out_root", default="Results_reproduced")
    ap.add_argument("--normalizer", default="artifacts/emg_norm.pkl")
    ap.add_argument("--lexicon", default="artifacts/lexicon.txt")
    ap.add_argument("--train_dir_for_lm", default="data/train_emg")
    ap.add_argument("--backbone_ckpt", action="append", default=[],
                    help="name=path/to/checkpoint.pt (repeatable, for --suite backbones).")
    ap.add_argument("--betas", nargs="*", type=float,
                    default=[0.0, 0.2, 0.4, 0.55, 0.7, 0.8])
    # Fig. 5 has to bracket the optimum, not stop at it. The previous grid ended
    # at 0.6 while validation tuning selected lambda_fix = 2.0, so the curve was
    # monotone decreasing to its right edge and never showed a minimum.
    ap.add_argument("--lambdas", nargs="*", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    common = [
        "--normalizer", args.normalizer,
        "--lexicon", args.lexicon,
        "--train_dir_for_lm", args.train_dir_for_lm,
    ]
    root = Path(args.out_root)
    tuning_suites = {"sweep_beta", "sweep_lam"}

    def data_for(suite: str) -> str:
        if args.data_dir:
            return args.data_dir
        return "data/val_emg" if suite in tuning_suites else "data/test_emg"

    suites = (
        ["decoders", "ablation", "decomp", "latency", "sweep_beta", "sweep_lam"]
        if args.suite == "all"
        else [args.suite]
    )

    for suite in suites:
        data_dir = data_for(suite)

        if suite == "decoders":
            for dec in DECODERS:
                run(infer_cmd(args.ckpt, data_dir, root / "decoders" / f"{dec}.jsonl",
                              dec, common=common, dry=args.dry_run), args.dry_run)

        elif suite == "ablation":
            for name, extra in ABLATIONS.items():
                run(infer_cmd(args.ckpt, data_dir, root / "ablation" / f"{name}.jsonl",
                              "ns", extra=extra, common=common, dry=args.dry_run), args.dry_run)

        elif suite == "decomp":
            # ctc_greedy/ctc_grammar are a matched pair over the same posterior,
            # so the per-field breakdown shows exactly which fields the grammar
            # repairs rather than only that the corpus WER moved.
            for dec in ["ar_beam", "ctc_greedy", "ctc_grammar", "ns", "ns_joint_adaptive"]:
                run(infer_cmd(args.ckpt, data_dir, root / "decomp" / f"{dec}.jsonl",
                              dec, common=common, dry=args.dry_run), args.dry_run)

        elif suite == "latency":
            # ctc_grammar searches a 54k-utterance trie per utterance, so Table VI
            # has to price it; ctc_greedy is the floor it should be compared to.
            for dec in ["ctc_greedy", "ctc_grammar", "ar_beam", "ns",
                        "ns_ctc_select", "ns_joint_adaptive"]:
                run(infer_cmd(args.ckpt, data_dir, root / "latency" / f"{dec}.jsonl",
                              dec, extra=["--measure_latency"], common=common, dry=args.dry_run), args.dry_run)

        elif suite == "backbones":
            mapping = parse_kv(args.backbone_ckpt)
            if not mapping:
                raise SystemExit(
                    "--suite backbones needs at least one --backbone_ckpt name=path\n"
                    "(each backbone must be trained separately; only the frozen LLM changes)."
                )
            # Each backbone uses its OWN validation-tuned decoding config when
            # one exists. The optimum is not shared: on Llama-3.2-3B the joint
            # score saturates at K=12, while Llama-3.2-1B keeps improving to
            # K=48 (0.2707 -> 0.2331 on validation). Applying the primary's K to
            # every backbone would understate the smaller ones by ~0.04 WER.
            # The protocol is identical across backbones, which is what keeps
            # Fig. 2 a controlled comparison.
            for name, ck in mapping.items():
                tuned = REPO / "artifacts" / f"ns_tuning_{name}.json"
                flags = _base_flags(tuned) if tuned.exists() else BASE_FLAGS
                if not tuned.exists():
                    print(f"[run_experiments] WARNING: no {tuned.name}; "
                          f"{name} falls back to the primary's config")
                run(infer_cmd(ck, data_dir, root / "backbones" / f"{name}.jsonl",
                              "ns_joint_adaptive", common=common, base=flags, dry=args.dry_run), args.dry_run)

        elif suite == "supervision":
            # Table III. All rows share the primary's inference recipe. A
            # CTC-only model has no trained AR head, so it is scored with CTC
            # greedy; the other two use the full NS stack.
            mapping = parse_kv(args.backbone_ckpt)
            if not mapping:
                raise SystemExit(
                    "--suite supervision needs --backbone_ckpt name=path entries "
                    "(ar_ctc / ar_only / ctc_only)."
                )
            for name, ck in mapping.items():
                dec = "ctc_greedy" if name == "ctc_only" else "ns_joint_adaptive"
                run(infer_cmd(ck, data_dir, root / "supervision" / f"{name}.jsonl",
                              dec, common=common, dry=args.dry_run), args.dry_run)

        elif suite == "sweep_beta":
            # Fig. 4(a) is a MULTI-backbone curve, so the sweep runs per
            # backbone when checkpoints are supplied and
            # falls back to the primary alone otherwise. Each backbone keeps its
            # own tuned config and varies only beta, so the curves stay
            # comparable without pinning them all to the primary's beam.
            mapping = parse_kv(args.backbone_ckpt) or {"": args.ckpt}
            for name, ck in mapping.items():
                tuned = REPO / "artifacts" / f"ns_tuning_{name}.json" if name else TUNED_PATH
                flags = _base_flags(tuned) if tuned.exists() else BASE_FLAGS
                prefix = f"{name}__" if name else ""
                for b in args.betas:
                    run(infer_cmd(ck, data_dir,
                                  root / "sweep_beta" / f"{prefix}beta_{b:g}.jsonl",
                                  "ns_joint_adaptive", extra=["--beta", str(b)],
                                  common=common, base=flags, dry=args.dry_run), args.dry_run)

        elif suite == "sweep_lam":
            for lam in args.lambdas:
                run(infer_cmd(args.ckpt, data_dir,
                              root / "sweep_lam" / f"lambda_{lam:g}.jsonl",
                              "ns_joint_rerank", extra=["--lambda_fix", str(lam)],
                              common=common, dry=args.dry_run), args.dry_run)

    print(f"\nDone. Score the outputs with:\n"
          f"  python scripts/evaluate.py --decomp --bootstrap {root}/*/*.jsonl")


if __name__ == "__main__":
    main()
