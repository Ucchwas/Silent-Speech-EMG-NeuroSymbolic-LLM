#!/usr/bin/env python3
"""
scripts/ns_utils.py

NeuroSymbolic decoding utilities for the EMG→LLM silent-speech pipeline.

Implements (inference-time):
  1) Trie-constrained candidate generation (AR + char 5-gram + boundary terms, λ=0 inside beam)
  2) Joint AR+CTC reranking over top-M completed candidates
  3) Confidence-adaptive fusion (utterance-level λ from AR uncertainty + CTC blank dominance)

Resources:
  - Lexicon must be built from TRAIN transcripts only (artifacts/lexicon.txt).
  - Char 5-gram LM is trained from TRAIN transcripts only (data/train_emg/*.json).

This file intentionally does NOT require train_texts.txt.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import math
import re
import json
import torch


# -----------------------------
# Trie
# -----------------------------
class TrieNode:
    __slots__ = ("children", "is_word")

    def __init__(self) -> None:
        self.children: Dict[str, "TrieNode"] = {}
        self.is_word: bool = False

    def add(self, word: str) -> None:
        node = self
        for ch in word:
            node = node.children.setdefault(ch, TrieNode())
        node.is_word = True


def _pick_lexicon_path(lexicon_path: Optional[str | Path], artifacts_dir: str | Path) -> Path:
    if lexicon_path is not None:
        p = Path(lexicon_path)
        if p.exists():
            return p
        raise FileNotFoundError(f"lexicon_path not found: {p}")

    artifacts_dir = Path(artifacts_dir)
    cand = artifacts_dir / "lexicon.txt"
    if cand.exists():
        return cand

    # fallback for older layouts
    cand2 = Path("data") / "lexicon.txt"
    if cand2.exists():
        return cand2

    raise FileNotFoundError("Lexicon not found. Expected artifacts/lexicon.txt (preferred).")


def build_trie_from_lexicon(tt, lexicon_path: Optional[str | Path] = None, artifacts_dir: str | Path = "artifacts") -> TrieNode:
    """
    Build a prefix trie from a train-only lexicon (one word per line).
    Words containing characters outside tt.BASE_CHARS (excluding space) are skipped.
    """
    lex_path = _pick_lexicon_path(lexicon_path, artifacts_dir)
    root = TrieNode()

    with lex_path.open("r", encoding="utf-8") as f:
        for line in f:
            w = (line.strip() or "").lower()
            if not w:
                continue
            if " " in w:
                continue
            ok = True
            for ch in w:
                # Require valid character in TextTransform base vocabulary
                if getattr(tt, "char_to_id", None) is not None:
                    if tt.char_to_id(ch) is None:
                        ok = False
                        break
                else:
                    if ch not in getattr(tt, "BASE_CHARS", ""):
                        ok = False
                        break
            if ok:
                root.add(w)

    return root


# -----------------------------
# Character n-gram LM (5-gram)
# -----------------------------
class CharNgramLM:
    """Add-α smoothed character n-gram LM."""

    def __init__(self, n: int = 5, alpha: float = 0.1, vocab: Optional[List[str]] = None):
        assert n >= 2
        self.n = int(n)
        self.alpha = float(alpha)
        self.vocab = vocab or []
        self._V = max(1, len(self.vocab))
        self.ctx_counts: Dict[str, int] = {}
        self.ctx_ch_counts: Dict[Tuple[str, str], int] = {}
        self._pad = "\u0002" * (self.n - 1)

    def fit(self, sequences: Iterable[str]) -> "CharNgramLM":
        for s in sequences:
            if s is None:
                continue
            s = (s or "").strip().lower()
            if not s:
                continue
            s = re.sub(r"\s+", " ", s)
            padded = self._pad + s
            for i, ch in enumerate(s):
                ctx = padded[i : i + self.n - 1]
                self.ctx_counts[ctx] = self.ctx_counts.get(ctx, 0) + 1
                key = (ctx, ch)
                self.ctx_ch_counts[key] = self.ctx_ch_counts.get(key, 0) + 1
        return self

    def init_ctx(self) -> str:
        return self._pad

    def update_ctx(self, ctx: str, ch: str) -> str:
        ctx2 = (ctx + ch)[-(self.n - 1) :]
        if len(ctx2) < self.n - 1:
            ctx2 = (self._pad + ctx2)[-(self.n - 1) :]
        return ctx2

    def logp_next(self, ctx: str, ch: str) -> float:
        denom = self.ctx_counts.get(ctx, 0) + self.alpha * self._V
        num = self.ctx_ch_counts.get((ctx, ch), 0) + self.alpha
        return math.log(num / denom)


def _char_vocab_from_tt(tt) -> List[str]:
    # Use BASE_CHARS so vocab includes space too.
    base = getattr(tt, "BASE_CHARS", "")
    return list(base) if isinstance(base, str) else list(base)


def _iter_jsons(train_dir: Path) -> Iterable[Path]:
    for js in sorted(train_dir.glob("*.json")):
        yield js
    for npy in sorted(train_dir.glob("*.npy")):
        js = npy.with_suffix(".json")
        if js.exists():
            yield js


def _read_text(js_path: Path) -> str:
    obj = json.loads(js_path.read_text(encoding="utf-8"))
    return obj.get("text", obj.get("transcript", obj.get("label", "")))


def train_char_5gram(
    tt,
    train_dir: str | Path = "data/train_emg",
    alpha: float = 0.1,
) -> CharNgramLM:
    """
    Train a char 5-gram LM from TRAIN transcripts only (data/train_emg).
    No train_texts.txt needed.
    """
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    vocab = _char_vocab_from_tt(tt)
    lm = CharNgramLM(n=5, alpha=float(alpha), vocab=vocab)

    lines: List[str] = []
    for js in _iter_jsons(train_dir):
        raw = _read_text(js)
        clean = tt.clean(raw) if hasattr(tt, "clean") else (raw or "").lower().strip()
        if clean:
            lines.append(clean)

    lm.fit(lines)
    return lm


# -----------------------------
# CTC forward scoring (log-space)
# -----------------------------
def ctc_forward_logprob(ctc_logits: torch.Tensor, labels: List[int], blank_id: int) -> float:
    """
    Compute log p_CTC(labels | logits) with the standard forward algorithm.
    ctc_logits: [T, V] (unnormalized)
    labels: target sequence over the same symbol ids (no BOS/EOS; include space if applicable)
    """
    if ctc_logits.ndim != 2:
        raise ValueError(f"ctc_logits must be [T,V], got {tuple(ctc_logits.shape)}")
    T, V = ctc_logits.shape
    if T == 0:
        return float("-inf")
    if blank_id < 0 or blank_id >= V:
        raise ValueError(f"blank_id out of range: {blank_id} vs V={V}")

    logp = torch.log_softmax(ctc_logits, dim=-1)

    # extended target with blanks
    ext: List[int] = [blank_id]
    for l in labels:
        if l == blank_id:
            continue
        ext.append(int(l))
        ext.append(blank_id)

    S = len(ext)
    neg_inf = torch.tensor(float("-inf"), device=logp.device, dtype=logp.dtype)
    alpha = torch.full((S,), neg_inf, device=logp.device, dtype=logp.dtype)

    alpha[0] = logp[0, blank_id]
    if S > 1:
        alpha[1] = logp[0, ext[1]]

    for t in range(1, T):
        new_alpha = torch.full((S,), neg_inf, device=logp.device, dtype=logp.dtype)
        for s in range(S):
            emit = logp[t, ext[s]]
            a = alpha[s]
            b = alpha[s - 1] if s - 1 >= 0 else neg_inf
            c = neg_inf
            if s - 2 >= 0 and ext[s] != blank_id and ext[s] != ext[s - 2]:
                c = alpha[s - 2]
            new_alpha[s] = torch.logsumexp(torch.stack([a, b, c]), dim=0) + emit
        alpha = new_alpha

    if S == 1:
        out = alpha[0]
    else:
        out = torch.logsumexp(torch.stack([alpha[S - 1], alpha[S - 2]]), dim=0)
    return float(out.item())


# -----------------------------
# NeuroSymbolic decoding
# -----------------------------
@dataclass
class NSConfig:
    beam_size: int = 6
    max_len: int = 64
    top_m: int = 16

    beta: float = 0.55
    kappa: float = 0.40
    gamma: float = 0.45

    alpha: float = 0.60
    delta: float = 5.0

    min_eos_len: int = 8
    per_step_top: int = 50  # IMPORTANT: avoid overly aggressive pruning

    # Fusion / adaptive λ
    lambda_fix: float = 0.25
    lambda0: float = 0.25
    a: float = 0.20
    b: float = 0.20
    lambda_min: float = 0.0
    lambda_max: float = 0.6


@dataclass
class Candidate:
    ids: List[int]          # includes BOS; may include EOS at end
    node: TrieNode          # trie node for current token prefix
    ctx: str                # n-gram context
    vis_len: int            # visible length (includes spaces)
    ar_logp: float
    lm_logp: float
    wb: int
    oov: int
    ent_sum: float
    steps: int
    ended: bool

    def score0(self, cfg: NSConfig) -> float:
        raw = self.ar_logp + cfg.beta * self.lm_logp + cfg.kappa * float(self.wb) - cfg.gamma * float(self.oov)
        L = max(1, self.vis_len)
        return raw / ((L + cfg.delta) ** cfg.alpha)

    def score_final(self, cfg: NSConfig, ctc_logp: float, lam: float) -> float:
        raw = self.ar_logp + lam * float(ctc_logp) + cfg.beta * self.lm_logp + cfg.kappa * float(self.wb) - cfg.gamma * float(self.oov)
        L = max(1, self.vis_len)
        return raw / ((L + cfg.delta) ** cfg.alpha)


def _entropy_from_logits(logits: torch.Tensor) -> float:
    p = torch.softmax(logits, dim=-1)
    return float((-(p * torch.log(p.clamp_min(1e-12))).sum()).item())


def _id_to_char(tt, tid: int) -> Optional[str]:
    if hasattr(tt, "id_to_char"):
        return tt.id_to_char(tid)
    base = getattr(tt, "BASE_CHARS", "")
    if 0 <= tid < len(base):
        return base[tid]
    return None


def _char_to_id(tt, ch: str) -> Optional[int]:
    if hasattr(tt, "char_to_id"):
        return tt.char_to_id(ch)
    base = getattr(tt, "BASE_CHARS", "")
    try:
        return base.index(ch)
    except ValueError:
        return None


def _expand_one(
    hyp: Candidate,
    tid: int,
    logp_tid: float,
    ent_parent: float,
    tt,
    trie_root: TrieNode,
    charlm: CharNgramLM,
) -> Candidate:
    ids2 = hyp.ids + [tid]
    ar_logp2 = hyp.ar_logp + float(logp_tid)
    ent_sum2 = hyp.ent_sum + float(ent_parent)
    steps2 = hyp.steps + 1

    wb2, oov2 = hyp.wb, hyp.oov
    node2 = hyp.node
    ctx2 = hyp.ctx
    lm_logp2 = hyp.lm_logp
    vis_len2 = hyp.vis_len
    ended2 = hyp.ended

    if tid == tt.EOS_IDX:
        wb2 += 1
        ended2 = True
        return Candidate(ids2, node2, ctx2, vis_len2, ar_logp2, lm_logp2, wb2, oov2, ent_sum2, steps2, ended2)

    ch = _id_to_char(tt, tid)
    if ch is None or len(ch) != 1:
        return Candidate(ids2, trie_root, ctx2, vis_len2, ar_logp2, lm_logp2, wb2, oov2 + 1, ent_sum2, steps2, ended2)

    # LM update on visible characters (including space)
    lm_logp2 += charlm.logp_next(ctx2, ch)
    ctx2 = charlm.update_ctx(ctx2, ch)
    vis_len2 += 1

    if ch == " ":
        # word boundary was already verified before allowing this action
        wb2 += 1
        node2 = trie_root
        return Candidate(ids2, node2, ctx2, vis_len2, ar_logp2, lm_logp2, wb2, oov2, ent_sum2, steps2, ended2)

    # advance in trie
    nxt = hyp.node.children.get(ch)
    if nxt is None:
        # should not happen if allowed set is correct
        oov2 += 1
        nxt = trie_root
    node2 = nxt
    return Candidate(ids2, node2, ctx2, vis_len2, ar_logp2, lm_logp2, wb2, oov2, ent_sum2, steps2, ended2)


def generate_candidates(
    step_fn: Callable[[List[int]], torch.Tensor],
    tt,
    trie_root: TrieNode,
    charlm: CharNgramLM,
    cfg: NSConfig,
) -> List[Candidate]:
    """
    Trie-constrained candidate generation (paper-style):
      - allow next letters only from trie children
      - allow space only if current trie node is a complete word (word boundary)
      - allow EOS only if node.is_word and vis_len >= min_eos_len
      - keep top-K by score0 each step
    Returns up to cfg.top_m completed candidates if available, else top_m from beam.
    """
    space_id = _char_to_id(tt, " ")
    if space_id is None:
        raise ValueError("TextTransform must include space in BASE_CHARS")

    init = Candidate(
        ids=[tt.BOS_IDX],
        node=trie_root,
        ctx=charlm.init_ctx(),
        vis_len=0,
        ar_logp=0.0,
        lm_logp=0.0,
        wb=0,
        oov=0,
        ent_sum=0.0,
        steps=0,
        ended=False,
    )

    beam: List[Candidate] = [init]
    finished: List[Candidate] = []

    for _ in range(int(cfg.max_len)):
        new_beam: List[Candidate] = []
        all_ended = True

        for hyp in beam:
            if hyp.ended:
                new_beam.append(hyp)
                continue

            all_ended = False
            logits = step_fn(hyp.ids)
            if logits.ndim != 1:
                logits = logits.view(-1)
            logp = torch.log_softmax(logits, dim=-1)
            ent = _entropy_from_logits(logits)

            allowed: List[int] = []

            # allowed letters from trie
            for ch in hyp.node.children.keys():
                tid = _char_to_id(tt, ch)
                if tid is not None:
                    allowed.append(tid)

            # allow space only if current prefix forms a complete word
            if hyp.node.is_word:
                allowed.append(space_id)

            # EOS gating: only at word boundary and after minimum visible length
            if hyp.vis_len >= int(cfg.min_eos_len) and hyp.node.is_word:
                allowed.append(tt.EOS_IDX)

            if not allowed:
                continue

            allowed = list(dict.fromkeys(allowed))
            allowed_t = torch.tensor(allowed, device=logp.device, dtype=torch.long)
            sel = logp.index_select(0, allowed_t)

            k = min(int(cfg.per_step_top), int(sel.numel()))
            vals, idxs = torch.topk(sel, k)

            for v, j in zip(vals.tolist(), idxs.tolist()):
                tid = int(allowed[int(j)])
                new_beam.append(_expand_one(hyp, tid, float(v), ent, tt, trie_root, charlm))

        if not new_beam:
            break

        new_beam.sort(key=lambda c: c.score0(cfg), reverse=True)
        beam = new_beam[: int(cfg.beam_size)]

        # collect finished candidates
        finished.extend([c for c in beam if c.ended])
        if all_ended:
            break

    if finished:
        finished.sort(key=lambda c: c.score0(cfg), reverse=True)
        return finished[: int(cfg.top_m)]

    beam.sort(key=lambda c: c.score0(cfg), reverse=True)
    return beam[: int(cfg.top_m)]


def compute_adaptive_lambda(best: Candidate, ctc_logits: torch.Tensor, tt, cfg: NSConfig) -> Tuple[float, float, float]:
    """
    λ = clip(λ0 + a*uAR − b*uCTC, [λmin, λmax])
    uAR: average entropy along decoding steps (approx.)
    uCTC: mean blank probability over time
    """
    u_ar = float(best.ent_sum / max(1, best.steps))

    blank_id = getattr(tt, "PAD_IDX", ctc_logits.shape[-1] - 1)
    probs = torch.softmax(ctc_logits, dim=-1)
    if blank_id < 0 or blank_id >= probs.shape[-1]:
        blank_id = probs.shape[-1] - 1
    u_ctc = float(probs[:, blank_id].mean().item())

    lam = float(cfg.lambda0) + float(cfg.a) * u_ar - float(cfg.b) * u_ctc
    lam = float(max(float(cfg.lambda_min), min(float(cfg.lambda_max), lam)))
    return lam, u_ar, u_ctc


def decode_ns_with_rerank(
    step_fn: Callable[[List[int]], torch.Tensor],
    ctc_logits: torch.Tensor,
    tt,
    trie_root: TrieNode,
    charlm: CharNgramLM,
    cfg: Optional[NSConfig] = None,
    adaptive: bool = True,
    joint: bool = True,
) -> Tuple[str, dict]:
    """
    Returns:
      best_text, info dict with lambda/u_ar/u_ctc and candidate scores
    """
    cfg = cfg or NSConfig()
    candidates = generate_candidates(step_fn, tt, trie_root, charlm, cfg)

    # best NS-only (λ=0)
    best0 = max(candidates, key=lambda c: c.score0(cfg))

    # fusion weight
    if adaptive:
        lam, u_ar, u_ctc = compute_adaptive_lambda(best0, ctc_logits, tt, cfg)
    else:
        lam, u_ar, u_ctc = float(cfg.lambda_fix), None, None

    blank_id = getattr(tt, "PAD_IDX", ctc_logits.shape[-1] - 1)

    best = None
    best_score = float("-inf")
    cand_info = []

    for cand in candidates:
        # CTC labels should NOT include BOS/EOS/PAD
        text = tt.ids_to_text(cand.ids)
        labels = tt.text_to_ids(text, add_bos_eos=False)
        ctc_lp = ctc_forward_logprob(ctc_logits, labels, blank_id=blank_id)

        if joint:
            score = cand.score_final(cfg, ctc_lp, lam)
        else:
            score = cand.score0(cfg)

        cand_info.append(
            {
                "text": text,
                "score0": cand.score0(cfg),
                "ctc_logp": ctc_lp,
                "final_score": score,
                "wb": cand.wb,
                "oov": cand.oov,
                "vis_len": cand.vis_len,
            }
        )

        if score > best_score:
            best_score = score
            best = cand

    assert best is not None
    best_text = tt.ids_to_text(best.ids)

    return best_text, {"lambda": lam, "u_ar": u_ar, "u_ctc": u_ctc, "candidates": cand_info}