#!/usr/bin/env python3
"""
inference_emg_llm.py

Paper-aligned inference + evaluation for Silent Speech EMG.

Modes:
  (A) Neural-only AR beam
  (B) NeuroSymbolic (NS) constrained decoding (trie + char 5-gram + boundary terms + EOS gating)
  (C) NS + joint AR+CTC reranking + confidence-adaptive fusion

Inputs:
  data_dir contains *.npy features and matching *.json sidecars.
  json must contain transcript under "text" or fallback keys.

Outputs:
  JSONL file with ref + hypotheses.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# -------------------------
# Robust imports
# -------------------------
try:
    from scripts.data_utils import TextTransform, load_normalizer
except Exception:
    from data_utils import TextTransform, load_normalizer

try:
    from train_emg_llm import ModelConfig, EMGToFrozenLLM
except Exception:
    from scripts.train_emg_llm import ModelConfig, EMGToFrozenLLM

try:
    from scripts.ns_utils import NSConfig, build_trie_from_lexicon, train_char_5gram, decode_ns_with_rerank
except Exception:
    from ns_utils import NSConfig, build_trie_from_lexicon, train_char_5gram, decode_ns_with_rerank


_space_re = re.compile(r"\s+")


def collapse_spaces_and_repeats(s: str) -> str:
    s = _space_re.sub(" ", (s or "")).strip()
    s = re.sub(r"(.)\1{4,}", r"\1\1\1", s)
    return s


def _edit_distance(a: List[str], b: List[str]) -> int:
    n, m = len(a), len(b)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[m]


def wer(tt: TextTransform, ref: str, hyp: str) -> float:
    r = tt.clean(ref).split()
    h = tt.clean(hyp).split()
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return _edit_distance(r, h) / float(len(r))


def cer(tt: TextTransform, ref: str, hyp: str) -> float:
    r = list(tt.clean(ref))
    h = list(tt.clean(hyp))
    if len(r) == 0:
        return 0.0 if len(h) == 0 else 1.0
    return _edit_distance(r, h) / float(len(r))


def _json_for_npy(npy_path: Path) -> Optional[Path]:
    j1 = npy_path.with_suffix(".json")
    if j1.exists():
        return j1
    stem2 = npy_path.stem.replace("_silent", "")
    j2 = npy_path.with_name(stem2 + ".json")
    if j2.exists():
        return j2
    return None


def scan_pairs(data_dir: Path) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for npy_path in sorted(data_dir.glob("*.npy")):
        j = _json_for_npy(npy_path)
        if j is not None:
            pairs.append((npy_path, j))
    return pairs


def read_text_from_json(json_path: Path) -> str:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    return obj.get("text", obj.get("transcript", obj.get("label", "")))


@torch.no_grad()
def ar_greedy_decode(
    model: EMGToFrozenLLM,
    prefix_embeds: torch.Tensor,
    prefix_mask: torch.Tensor,
    tt: TextTransform,
    max_len: int,
    min_len: int,
) -> List[int]:
    device = prefix_embeds.device
    llm_dtype = next(model.llm.parameters()).dtype
    prefix_embeds = prefix_embeds.to(dtype=llm_dtype)
    prefix_mask = prefix_mask.long()

    ids = torch.tensor([[tt.BOS_IDX]], dtype=torch.long, device=device)
    for step in range(max_len):
        logits = model.next_token_logits(prefix_embeds, prefix_mask, ids, pad_idx=tt.PAD_IDX)
        logp = F.log_softmax(logits[0], dim=-1)
        if step + 1 < min_len:
            logp[tt.EOS_IDX] = -1e9
        nxt = int(torch.argmax(logp).item())
        ids = torch.cat([ids, torch.tensor([[nxt]], device=device, dtype=torch.long)], dim=1)
        if nxt == tt.EOS_IDX:
            break
    return ids[0].tolist()


@torch.no_grad()
def ar_beam_decode(
    model: EMGToFrozenLLM,
    prefix_embeds: torch.Tensor,
    prefix_mask: torch.Tensor,
    tt: TextTransform,
    beam: int,
    alpha: float,
    delta: float,
    max_len: int,
    min_len: int,
) -> List[int]:
    device = prefix_embeds.device
    llm_dtype = next(model.llm.parameters()).dtype
    prefix_embeds = prefix_embeds.to(dtype=llm_dtype)
    prefix_mask = prefix_mask.long()

    hyps: List[Tuple[List[int], float]] = [([tt.BOS_IDX], 0.0)]
    finished: List[Tuple[List[int], float]] = []

    def norm_score(ids: List[int], score: float) -> float:
        L = max(1, len(ids) - 1)  # exclude BOS
        return score / ((L + float(delta)) ** float(alpha))

    for step in range(max_len):
        B = len(hyps)
        maxL = max(len(ids) for ids, _ in hyps)
        ids_pad = torch.full((B, maxL), tt.PAD_IDX, dtype=torch.long, device=device)
        for i, (ids, _) in enumerate(hyps):
            ids_pad[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)

        ctx_emb = prefix_embeds.expand(B, -1, -1)
        ctx_mask = prefix_mask.expand(B, -1)
        logits = model.next_token_logits(ctx_emb, ctx_mask, ids_pad, pad_idx=tt.PAD_IDX)
        logp = F.log_softmax(logits, dim=-1)

        cand: List[Tuple[List[int], float]] = []
        for i, (ids, sc) in enumerate(hyps):
            row = logp[i].clone()
            if step + 1 < min_len:
                row[tt.EOS_IDX] = -1e9
            topv, topi = torch.topk(row, k=min(beam, row.numel()))
            for lp, tok in zip(topv.tolist(), topi.tolist()):
                tok = int(tok)
                new_ids = ids + [tok]
                new_sc = sc + float(lp)
                if tok == tt.EOS_IDX:
                    finished.append((new_ids, new_sc))
                else:
                    cand.append((new_ids, new_sc))

        if not cand:
            break
        cand.sort(key=lambda x: norm_score(x[0], x[1]), reverse=True)
        hyps = cand[:beam]

    if finished:
        finished.sort(key=lambda x: norm_score(x[0], x[1]), reverse=True)
        return finished[0][0]
    hyps.sort(key=lambda x: norm_score(x[0], x[1]), reverse=True)
    return hyps[0][0]


def ctc_greedy_decode_ids(ctc_logits: torch.Tensor, blank_id: int) -> List[int]:
    ids = torch.argmax(ctc_logits, dim=-1).tolist()
    out: List[int] = []
    prev = None
    for i in ids:
        if i == blank_id:
            prev = i
            continue
        if prev is None or i != prev:
            out.append(int(i))
        prev = i
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--ckpt", type=str, default="artifacts/best_checkpoint.pt")
    p.add_argument("--data_dir", type=str, default="data/val_emg")
    p.add_argument("--normalizer", type=str, default="artifacts/emg_norm.pkl")
    p.add_argument("--out", type=str, default="artifacts/infer.jsonl")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=0)

    # Neural-only decoding
    p.add_argument("--beam", type=int, default=6)
    p.add_argument("--alpha", type=float, default=0.6)
    p.add_argument("--delta", type=float, default=5.0)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--lmax", type=int, default=None)  # alias
    p.add_argument("--min_len", type=int, default=1)

    # NS decoding
    p.add_argument("--ns", action="store_true")
    p.add_argument("--lexicon", type=str, default="artifacts/lexicon.txt")
    p.add_argument("--train_dir_for_lm", type=str, default="data/train_emg")
    p.add_argument("--train_texts", type=str, default="", help="Unused; kept for compatibility.")

    p.add_argument("--trie", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--beta", type=float, default=0.55)
    p.add_argument("--kappa", type=float, default=0.40)
    p.add_argument("--gamma", type=float, default=0.45)
    p.add_argument("--min_eos_len", type=int, default=8)
    p.add_argument("--per_step_top", type=int, default=50)

    # Rerank / fusion flags
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--joint", action="store_true")
    p.add_argument("--adaptive", action="store_true")

    p.add_argument("--rerank_M", type=int, default=16)
    p.add_argument("--lambda_fix", type=float, default=0.25)
    p.add_argument("--lambda_min", type=float, default=0.0)
    p.add_argument("--lambda_max", type=float, default=0.6)
    p.add_argument("--lambda0", type=float, default=0.25)
    p.add_argument("--a", type=float, default=0.20)
    p.add_argument("--b", type=float, default=0.20)

    p.add_argument("--dump_candidates", action="store_true", help="Store NS candidate list in output JSONL.")

    args = p.parse_args()
    if args.lmax is not None:
        args.max_len = int(args.lmax)
    return args


def load_model_and_tt(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("model_cfg", None)
    if cfg_dict is None:
        raise ValueError(f"{ckpt_path} missing 'model_cfg'.")
    cfg = ModelConfig(**cfg_dict)
    model = EMGToFrozenLLM(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    tt = TextTransform()
    return model, tt


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model, tt = load_model_and_tt(Path(args.ckpt), device)

    norm = None
    norm_path = Path(args.normalizer)
    if norm_path.exists():
        norm = load_normalizer(str(norm_path))

    pairs = scan_pairs(Path(args.data_dir))
    if not pairs:
        raise FileNotFoundError(f"No (npy,json) pairs found in {args.data_dir}")
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    trie_root = None
    charlm = None
    ns_cfg = None

    if args.ns:
        lex_path = Path(args.lexicon)
        if not lex_path.exists():
            raise FileNotFoundError(
                f"lexicon not found: {lex_path}. Run: python scripts/make_lexicon.py --src data/train_emg --out artifacts/lexicon.txt"
            )

        trie_root = build_trie_from_lexicon(tt, lexicon_path=str(lex_path), artifacts_dir="artifacts")
        charlm = train_char_5gram(tt, train_dir=args.train_dir_for_lm, alpha=0.1)

        ns_cfg = NSConfig(
            beam_size=int(args.beam),
            max_len=int(args.max_len),
            top_m=int(args.rerank_M),
            beta=float(args.beta),
            kappa=float(args.kappa),
            gamma=float(args.gamma),
            alpha=float(args.alpha),
            delta=float(args.delta),
            min_eos_len=int(args.min_eos_len),
            per_step_top=int(args.per_step_top),
            lambda_fix=float(args.lambda_fix),
            lambda0=float(args.lambda0),
            a=float(args.a),
            b=float(args.b),
            lambda_min=float(args.lambda_min),
            lambda_max=float(args.lambda_max),
        )

    sum_w = 0.0
    sum_c = 0.0

    with out_path.open("w", encoding="utf-8") as f_out:
        for npy_path, json_path in pairs:
            x = np.load(npy_path).astype(np.float32)
            raw_ref = read_text_from_json(json_path)
            ref = tt.clean(raw_ref)

            if norm is not None:
                x = norm.transform(x).astype(np.float32)

            T = x.shape[0]
            emg = torch.from_numpy(x).unsqueeze(0).to(device)
            emg_mask = torch.ones((1, T), dtype=torch.bool, device=device)

            with torch.no_grad():
                prefix_embeds, prefix_mask = model.prepare_context(emg, emg_mask)
                emg_embeds = model.adapter(emg)  # (1, T', H)
                ctc_logits = model.ctc_head(emg_embeds)[0]  # (T', V)

            ids_g = ar_greedy_decode(model, prefix_embeds, prefix_mask, tt, args.max_len, args.min_len)
            ids_b = ar_beam_decode(model, prefix_embeds, prefix_mask, tt, args.beam, args.alpha, args.delta, args.max_len, args.min_len)

            hyp_ar_greedy = collapse_spaces_and_repeats(tt.ids_to_text(ids_g))
            hyp_ar_beam = collapse_spaces_and_repeats(tt.ids_to_text(ids_b))

            blank_id = tt.PAD_IDX
            ctc_ids = ctc_greedy_decode_ids(ctc_logits, blank_id=blank_id)
            hyp_ctc = collapse_spaces_and_repeats(tt.ids_to_text(ctc_ids))

            final_txt = hyp_ar_beam
            final_src = "ar_beam"
            ns_best = ""
            ns_info = None

            if args.ns and trie_root is not None and charlm is not None and ns_cfg is not None:
                def step_fn(ids_list: List[int]) -> torch.Tensor:
                    ids_t = torch.tensor([ids_list], dtype=torch.long, device=device)
                    return model.next_token_logits(prefix_embeds, prefix_mask.long(), ids_t, pad_idx=tt.PAD_IDX)[0]

                # If user did not ask for rerank, we still decode NS but with joint=False/adaptive=False
                use_joint = bool(args.rerank and args.joint)
                use_adapt = bool(args.rerank and args.joint and args.adaptive)

                ns_text, info = decode_ns_with_rerank(
                    step_fn=step_fn,
                    ctc_logits=ctc_logits,
                    tt=tt,
                    trie_root=trie_root,
                    charlm=charlm,
                    cfg=ns_cfg,
                    adaptive=use_adapt,
                    joint=use_joint,
                )

                ns_best = collapse_spaces_and_repeats(ns_text)
                if info is not None and (not args.dump_candidates) and isinstance(info, dict) and "candidates" in info:
                    info = {k: v for k, v in info.items() if k != "candidates"}
                ns_info = info

                final_txt = ns_best
                if use_joint:
                    final_src = "ns_joint_adaptive" if use_adapt else "ns_joint_fixed"
                else:
                    final_src = "ns"

            w = wer(tt, ref, final_txt)
            c = cer(tt, ref, final_txt)
            sum_w += w
            sum_c += c

            rec = {
                "utt": npy_path.stem,
                "ref": ref,
                "final": final_txt,
                "final_source": final_src,
                "ar_greedy": hyp_ar_greedy,
                "ar_beam": hyp_ar_beam,
                "ctc_greedy": hyp_ctc,
                "ns_best": ns_best,
                "ns_info": ns_info,
                "paths": {"npy": str(npy_path), "json": str(json_path)},
            }
            f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n = len(pairs)
    print(f"Done. WER={sum_w/n:.4f} CER={sum_c/n:.4f} over {n} utterances")
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()