#!/usr/bin/env python3
"""
inference_emg_llm.py

Inference and evaluation for the Silent Speech EMG closed-vocabulary task.

Implements every decoding condition of Sec. V-C. `--decoder` picks which one
fills the `final` field (and therefore what the reported WER/CER refer to):

  ctc_greedy          greedy collapse from the CTC head
  ctc_lexicon         CTC prefix beam constrained by the flat word lexicon
  ctc_grammar         CTC prefix beam constrained by the utterance grammar
  ctc_grammar_ar      the grammar-valid CTC pool, reranked by the frozen LLM
  ar_greedy           greedy AR decoding (K = 1)
  ar_beam             AR beam search with length normalisation
  ns                  NS constrained beam (trie + 5-gram + boundary + EOS gating)
  ns_ctc_select       NS + CTC candidate selection            (Eq. 18)
  ns_joint_rerank     NS + joint AR+CTC reranking, fixed lambda_fix
  ns_joint_adaptive   NS + joint reranking + adaptive fusion  (Algorithm 3)

NS component ablations (Table IV) are switches on top of `--decoder ns`:
  --no-trie            disable the trie constraint
  --beta 0.0           disable char 5-gram fusion
  --kappa 0 --gamma 0  disable the boundary terms
  --no-eos_gating      disable EOS gating

Reported WER/CER are corpus level (see scripts/metrics.py).
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    from scripts.data_utils import TextTransform, load_normalizer
    from scripts.ar_decode import PrefixCachedAR, ctc_greedy_decode_ids
    from scripts.metrics import score_corpus
    from scripts.ns_utils import (
        NSConfig,
        build_trie_from_lexicon,
        build_grammar_trie,
        decode_ns,
        ctc_grammar_beam,
        rerank_with_ctc,
        select_with_ctc,
        train_char_5gram,
    )
except ImportError:
    from data_utils import TextTransform, load_normalizer
    from ar_decode import PrefixCachedAR, ctc_greedy_decode_ids
    from metrics import score_corpus
    from ns_utils import (
        NSConfig,
        build_trie_from_lexicon,
        build_grammar_trie,
        decode_ns,
        ctc_grammar_beam,
        rerank_with_ctc,
        select_with_ctc,
        train_char_5gram,
    )

from train_emg_llm import EMGToFrozenLLM, ModelConfig


DECODERS = [
    "ctc_greedy",
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

NS_DECODERS = {"ns", "ns_ctc_select", "ns_joint_rerank", "ns_joint_adaptive"}

# Decoders that need the symbolic resources built. The ctc_* constrained
# decoders are deliberately NOT in NS_DECODERS -- they must not trigger the
# AR-side NS beam -- but they do need a trie, and without this they would
# silently fall back to plain CTC.
CTC_SYMBOLIC = {"ctc_lexicon", "ctc_grammar", "ctc_grammar_ar"}
NEEDS_SYMBOLIC = NS_DECODERS | CTC_SYMBOLIC

# ctc_lexicon is the same prefix beam over the *flat word lexicon* instead of
# the utterance grammar. It is the missing cell of the constraint-placement
# 2x2 (channel x granularity), so it needs its own trie built alongside.
NEEDS_LEXICON_TRIE = {"ctc_lexicon"}

_space_re = re.compile(r"\s+")


def tidy(s: str) -> str:
    """Collapse whitespace and clamp pathological character runs."""
    s = _space_re.sub(" ", (s or "")).strip()
    return re.sub(r"(.)\1{4,}", r"\1\1\1", s)


# -----------------------------------------------------------------------------
# Data discovery
# -----------------------------------------------------------------------------
def _json_for_npy(npy_path: Path) -> Optional[Path]:
    j1 = npy_path.with_suffix(".json")
    if j1.exists():
        return j1
    j2 = npy_path.with_name(npy_path.stem.replace("_silent", "") + ".json")
    return j2 if j2.exists() else None


def scan_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    pairs = []
    for npy_path in sorted(data_dir.glob("*.npy")):
        j = _json_for_npy(npy_path)
        if j is not None:
            pairs.append((npy_path, j))
    return pairs


def read_text_from_json(json_path: Path) -> str:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    return obj.get("text", obj.get("transcript", obj.get("label", "")))


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decode and evaluate the EMG->LLM silent-speech model.")

    p.add_argument("--ckpt", type=str, default="artifacts/best_checkpoint.pt")
    p.add_argument("--data_dir", type=str, default="data/test_emg")
    p.add_argument("--normalizer", type=str, default="artifacts/emg_norm.pkl")
    p.add_argument("--out", type=str, default="artifacts/infer.jsonl")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0)

    p.add_argument("--decoder", choices=DECODERS, default="ns_joint_adaptive",
                   help="Decoding condition that produces the `final` transcript.")
    p.add_argument("--also", nargs="*", default=None,
                   help="Extra decoders to record alongside `final` (default: the cheap neural ones).")

    # --- Table I: NS constrained beam (final values) ---
    p.add_argument("--beam", type=int, default=6)            # K
    p.add_argument("--max_len", type=int, default=64)        # L_max (test)
    p.add_argument("--lmax", type=int, default=None)         # alias
    p.add_argument("--min_len", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--beta", type=float, default=0.55)
    p.add_argument("--kappa", type=float, default=0.40)
    p.add_argument("--gamma", type=float, default=0.45)
    p.add_argument("--min_eos_len", type=int, default=8)     # m
    p.add_argument("--per_step_top", type=int, default=50)

    # --- NS component ablations (Table IV) ---
    p.add_argument("--trie", action=argparse.BooleanOptionalAction, default=True,
                   help="Trie constraint on within-word expansions.")
    p.add_argument("--grammar", action=argparse.BooleanOptionalAction, default=True,
                   help="Use the utterance-level grammar (induced from train) instead of "
                        "the flat word lexicon. The flat lexicon constrains almost nothing "
                        "here: 889 of its 910 entries are numeral literals.")
    p.add_argument("--eos_gating", action=argparse.BooleanOptionalAction, default=True,
                   help="Require |s| >= m and a complete word before allowing EOS.")

    # --- Table I: CTC-assisted inference ---
    p.add_argument("--rerank_M", type=int, default=16)       # M
    p.add_argument("--lambda_fix", type=float, default=0.25)
    p.add_argument("--lambda0", type=float, default=0.25)
    p.add_argument("--a", type=float, default=0.20)
    p.add_argument("--b", type=float, default=0.20)
    p.add_argument("--lambda_min", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=0.6)

    # --- NS resources (TRAIN only) ---
    p.add_argument("--lexicon", type=str, default="artifacts/lexicon.txt")
    p.add_argument("--train_dir_for_lm", type=str, default="data/train_emg")
    p.add_argument("--ngram_alpha", type=float, default=0.1)

    p.add_argument("--ctc_grammar_beam", type=int, default=32,
                   help="Beam width for grammar-constrained CTC decoding.")
    p.add_argument("--ctc_grammar_pool", type=int, default=8,
                   help="Grammar-valid candidates kept from the CTC beam. The pool is "
                        "written to the output jsonl so its oracle can be measured, and "
                        "it is what ctc_grammar_ar reranks.")
    p.add_argument("--lambda_ar", type=float, default=1.0,
                   help="Weight on the AR string score when reranking the grammar-valid "
                        "CTC pool (ctc_grammar_ar). 0 reproduces ctc_grammar exactly.")
    p.add_argument("--ctc_select_lengthnorm", action="store_true",
                   help="Length-normalise the Eq. 18 comparison instead of comparing raw "
                        "sums of log probabilities (which favour the shorter candidate).")
    p.add_argument("--dump_candidates", action="store_true")
    p.add_argument("--measure_latency", action="store_true",
                   help="Report per-utterance decoding latency and peak GPU memory (Table VI).")

    args = p.parse_args()
    if args.lmax is not None:
        args.max_len = int(args.lmax)
    return args


def load_model_and_tt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = ckpt.get("model_cfg")
    if cfg_dict is None:
        raise ValueError(f"{ckpt_path} is missing 'model_cfg'.")
    model = EMGToFrozenLLM(ModelConfig(**cfg_dict)).to(device)
    model.load_trainable_state(ckpt["model"])
    model.eval()
    return model, TextTransform(), ckpt


# -----------------------------------------------------------------------------
# Per-utterance decoding
# -----------------------------------------------------------------------------
def decode_one(
    model,
    tt: TextTransform,
    emg: torch.Tensor,
    emg_mask: torch.Tensor,
    args,
    ns_cfg: Optional[NSConfig],
    trie_root,
    charlm,
    wanted: List[str],
    lex_trie_root=None,
) -> Tuple[Dict[str, str], dict]:
    """Run every requested decoding condition on one utterance."""
    out: Dict[str, str] = {}
    info: dict = {}

    prefix_embeds, prefix_mask = model.prepare_context(emg, emg_mask)
    ctc_logits = model.ctc_posteriors(emg, emg_mask)

    capacity = max(int(args.beam), 2)
    dec = PrefixCachedAR(model, tt, prefix_embeds, prefix_mask, capacity=capacity)

    # --- CTC greedy -------------------------------------------------------
    ctc_text = tidy(tt.ids_to_text(ctc_greedy_decode_ids(ctc_logits, blank_id=tt.PAD_IDX)))
    if "ctc_greedy" in wanted:
        out["ctc_greedy"] = ctc_text

    # --- grammar-constrained CTC -----------------------------------------
    # The symbolic constraint is applied HERE rather than to the AR beam: the
    # AR beam is already grammar-valid 50/50 on validation, whereas only
    # 22-31/50 of CTC greedy outputs are, so this is the channel where the
    # grammar has anything to exclude.
    if ("ctc_grammar" in wanted or "ctc_grammar_ar" in wanted) and trie_root is not None:
        pool = ctc_grammar_beam(
            ctc_logits, trie_root, tt,
            beam=max(int(args.ctc_grammar_beam), 2),
            charlm=charlm, beta=float(args.beta),
            return_pool=max(int(args.ctc_grammar_pool), 1),
        )
        # An empty pool means no grammar-valid path survived the beam; falling
        # back to the unconstrained string is strictly better than emitting
        # nothing, and is recorded so the rate can be reported.
        if "ctc_grammar" in wanted:
            out["ctc_grammar"] = tidy(pool[0][0]) if pool else ctc_text
        # The whole pool goes into the jsonl: its oracle is what decides
        # whether AR reranking can buy anything, and recomputing it later
        # would mean re-running the model.
        info["ctc_grammar"] = {
            "n_pool": len(pool),
            "logp": float(pool[0][1]) if pool else None,
            "fellback": not pool,
            "pool": [[tidy(t), float(lp)] for t, lp in pool],
        }

        # --- AR rescoring over the grammar-valid pool ---------------------
        # The frozen LLM never generates here; it only ranks strings the
        # grammar already admits. That is the one place on this corpus where
        # the language prior can help without being able to drift, since
        # every option it sees is structurally well formed.
        if "ctc_grammar_ar" in wanted:
            if pool:
                texts = [tidy(t) for t, _ in pool]
                ar = dec.score([tt.text_to_ids(t, add_bos_eos=True) for t in texts])
                norm = [(len(t) + ns_cfg.delta) ** ns_cfg.alpha if ns_cfg else 1.0
                        for t in texts]
                fused = [(lp + args.lambda_ar * a) / n
                         for (_, lp), a, n in zip(pool, ar, norm)]
                j = max(range(len(texts)), key=lambda i: fused[i])
                out["ctc_grammar_ar"] = texts[j]
                info["ctc_grammar_ar"] = {"picked": j, "n_pool": len(pool)}
            else:
                out["ctc_grammar_ar"] = ctc_text
                info["ctc_grammar_ar"] = {"picked": None, "n_pool": 0}

    # --- word-lexicon-constrained CTC (constraint-placement 2x2) ----------
    if "ctc_lexicon" in wanted and lex_trie_root is not None:
        lp = ctc_grammar_beam(
            ctc_logits, lex_trie_root, tt,
            beam=max(int(args.ctc_grammar_beam), 2),
            charlm=charlm, beta=float(args.beta),
            word_restart=True,
        )
        out["ctc_lexicon"] = tidy(lp[0][0]) if lp else ctc_text
        info["ctc_lexicon"] = {"n_pool": len(lp), "fellback": not lp}

    # --- neural-only AR ---------------------------------------------------
    if "ar_greedy" in wanted:
        out["ar_greedy"] = tidy(tt.ids_to_text(dec.greedy(max_len=args.max_len, min_len=args.min_len)))
    if "ar_beam" in wanted:
        ids = dec.beam(
            beam_size=args.beam, max_len=args.max_len,
            alpha=args.alpha, delta=args.delta, min_len=args.min_len,
        )
        out["ar_beam"] = tidy(tt.ids_to_text(ids))

    # --- NS family --------------------------------------------------------
    if any(w in NS_DECODERS for w in wanted) and ns_cfg is not None:
        ns_text, cands, _ = decode_ns(dec.step, tt, trie_root, charlm, ns_cfg)
        ns_text = tidy(ns_text)
        if "ns" in wanted:
            out["ns"] = ns_text

        if "ns_ctc_select" in wanted:
            picked, sel_info = select_with_ctc(
                ns_text, ctc_text, tt, dec.score,
                length_norm=bool(args.ctc_select_lengthnorm), cfg=ns_cfg,
            )
            out["ns_ctc_select"] = tidy(picked)
            info["ctc_select"] = sel_info

        if "ns_joint_rerank" in wanted:
            txt, rinfo = rerank_with_ctc(cands, ctc_logits, tt, ns_cfg, adaptive=False)
            out["ns_joint_rerank"] = tidy(txt)
            info["joint_fixed"] = {k: v for k, v in rinfo.items() if k != "candidates" or args.dump_candidates}

        if "ns_joint_adaptive" in wanted:
            txt, rinfo = rerank_with_ctc(cands, ctc_logits, tt, ns_cfg, adaptive=True)
            out["ns_joint_adaptive"] = tidy(txt)
            info["joint_adaptive"] = {k: v for k, v in rinfo.items() if k != "candidates" or args.dump_candidates}

        info["n_candidates"] = len(cands)

    return out, info


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, tt, _ckpt = load_model_and_tt(Path(args.ckpt), device)

    norm = None
    if Path(args.normalizer).exists():
        norm = load_normalizer(args.normalizer)
    else:
        print(f"WARNING: normalizer not found at {args.normalizer}; using raw features.")

    pairs = scan_pairs(Path(args.data_dir))
    if not pairs:
        raise FileNotFoundError(f"No (npy,json) pairs found in {args.data_dir}")
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    # Which conditions to record: the requested one plus cheap neural baselines.
    if args.also is None:
        extra = ["ar_greedy", "ar_beam", "ctc_greedy"]
    else:
        extra = list(args.also)
    wanted = list(dict.fromkeys([args.decoder] + extra))
    for w in wanted:
        if w not in DECODERS:
            raise ValueError(f"Unknown decoder '{w}'. Choose from {DECODERS}")

    ns_cfg = trie_root = charlm = lex_trie_root = None
    if any(w in NEEDS_SYMBOLIC for w in wanted):
        lex_path = Path(args.lexicon)
        if not lex_path.exists():
            raise FileNotFoundError(
                f"lexicon not found: {lex_path}\n"
                "Build it with: python scripts/make_lexicon.py --src data/train_emg "
                "--out artifacts/lexicon.txt"
            )
        if args.grammar:
            trie_root = build_grammar_trie(args.train_dir_for_lm, tt)
            ymin, ymax = trie_root.year_range
            print(f"NS constraint: utterance-level grammar (years {ymin}-{ymax})")
        else:
            trie_root = build_trie_from_lexicon(tt, lexicon_path=str(lex_path))
            print(f"NS constraint: flat word lexicon ({lex_path})")
        # ctc_lexicon needs the word trie regardless of which constraint the
        # rest of the run uses, so build it separately rather than flipping
        # --grammar and changing what every other decoder sees.
        if any(w in NEEDS_LEXICON_TRIE for w in wanted):
            lex_trie_root = (trie_root if not args.grammar
                             else build_trie_from_lexicon(tt, lexicon_path=str(lex_path)))
        charlm = train_char_5gram(tt, train_dir=args.train_dir_for_lm, alpha=args.ngram_alpha)
        ns_cfg = NSConfig(
            beam_size=args.beam, max_len=args.max_len, top_m=args.rerank_M,
            min_eos_len=args.min_eos_len, per_step_top=args.per_step_top,
            beta=args.beta, kappa=args.kappa, gamma=args.gamma,
            alpha=args.alpha, delta=args.delta,
            use_trie=bool(args.trie), use_eos_gating=bool(args.eos_gating),
            lambda_fix=args.lambda_fix, lambda0=args.lambda0, a=args.a, b=args.b,
            lambda_min=args.lambda_min, lambda_max=args.lambda_max,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.measure_latency and device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    refs: List[str] = []
    finals: List[str] = []
    per_decoder: Dict[str, List[str]] = {w: [] for w in wanted}
    latencies: List[float] = []
    total_chars = 0

    with out_path.open("w", encoding="utf-8") as f_out:
        for npy_path, json_path in pairs:
            x = np.load(npy_path).astype(np.float32)
            if norm is not None:
                x = norm.transform(x).astype(np.float32)

            ref = tt.clean(read_text_from_json(json_path))
            emg = torch.from_numpy(x).unsqueeze(0).to(device)
            emg_mask = torch.ones((1, x.shape[0]), dtype=torch.bool, device=device)

            if args.measure_latency and device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            hyps, info = decode_one(
                model, tt, emg, emg_mask, args, ns_cfg, trie_root, charlm, wanted,
                lex_trie_root=lex_trie_root,
            )

            if args.measure_latency and device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000.0)

            final_txt = hyps.get(args.decoder, "")
            total_chars += len(final_txt)

            refs.append(ref)
            finals.append(final_txt)
            for w in wanted:
                per_decoder[w].append(hyps.get(w, ""))

            rec = {
                "utt": npy_path.stem,
                "ref": ref,
                "final": final_txt,
                "final_source": args.decoder,
                **{w: hyps.get(w, "") for w in wanted},
                "paths": {"npy": str(npy_path), "json": str(json_path)},
            }
            if info:
                rec["ns_info"] = info
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---- report ----------------------------------------------------------
    s = score_corpus(refs, finals, tt)
    print(f"\nDecoder: {args.decoder}   ({s.n_utt} utterances from {args.data_dir})")
    print(f"  WER {s.wer:.4f} | CER {s.cer:.4f} | Exact-match {s.exact_match:.0f}%")
    print(f"  Sub {s.sub:.3f} | Del {s.dele:.3f} | Ins {s.ins:.3f}")

    if len(wanted) > 1:
        print("\n  All recorded conditions:")
        for w in wanted:
            sw = score_corpus(refs, per_decoder[w], tt)
            print(f"    {w:20s} WER {sw.wer:.4f}  CER {sw.cer:.4f}  EM {sw.exact_match:3.0f}%")

    if args.measure_latency:
        mean_ms = sum(latencies) / max(1, len(latencies))
        chars_per_s = total_chars / max(1e-9, sum(latencies) / 1000.0)
        line = f"\n  Latency {mean_ms:.1f} ms/utt | {chars_per_s:.0f} chars/s"
        peak_gb = None
        if device.type == "cuda":
            peak_gb = torch.cuda.max_memory_allocated(device) / 1e9
            line += f" | peak GPU mem {peak_gb:.2f} GB"
        print(line)

        # Table VI has to be rebuildable from the results tree, not scraped back
        # out of a SLURM log that the next run overwrites. Sidecar so the .jsonl
        # schema stays exactly as every other suite writes it.
        side = out_path.with_suffix(".latency.json")
        side.write_text(json.dumps({
            "decoder": args.decoder,
            "n_utt": len(latencies),
            "ms_per_utt": round(mean_ms, 1),
            "chars_per_s": round(chars_per_s, 1),
            "peak_gpu_gb": None if peak_gb is None else round(peak_gb, 2),
        }, indent=2), encoding="utf-8")
        print(f"  wrote {side}")

    print(f"\nWrote: {out_path}")


if __name__ == "__main__":
    main()
