#!/usr/bin/env python3
"""
scripts/ns_utils.py

NeuroSymbolic (NS) decoding utilities for the EMG->LLM silent-speech pipeline.
Everything here runs at inference time only; no parameters are trained.

Implements, following the paper:
  * Sec. IV-E1  trie state, boundary counts WB/OOV (Eq. 12-13), EOS gating
  * Sec. IV-E2  joint score (Eq. 16) and length normalisation (Eq. 17)
  * Algorithm 2 trie-aware constrained beam search (candidate generation,
                run with lambda_inf = 0)
  * Sec. IV-F-a CTC candidate selection (Eq. 18)
  * Sec. IV-F-b joint AR+CTC reranking with the CTC forward score (Eq. 9)
  * Algorithm 3 final selection with confidence-adaptive fusion (Eq. 19-21)

Symbolic resources are derived from TRAIN transcripts only:
  * lexicon L         -> artifacts/lexicon.txt (scripts/make_lexicon.py)
  * prefix trie T     -> built from L
  * char 5-gram p_ng  -> fitted on data/train_emg/*.json
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch


# =============================================================================
# Prefix trie over the closed vocabulary
# =============================================================================
class TrieNode:
    # `utterance_level` and `year_range` are only ever set on the root.
    __slots__ = ("children", "is_word", "utterance_level", "year_range")

    def __init__(self) -> None:
        self.children: Dict[str, "TrieNode"] = {}
        self.is_word: bool = False
        self.utterance_level: bool = False
        self.year_range = None

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

    cand = Path(artifacts_dir) / "lexicon.txt"
    if cand.exists():
        return cand
    cand2 = Path("data") / "lexicon.txt"
    if cand2.exists():
        return cand2
    raise FileNotFoundError(
        "Lexicon not found. Build it with:\n"
        "  python scripts/make_lexicon.py --src data/train_emg --out artifacts/lexicon.txt"
    )


def load_lexicon(
    tt, lexicon_path: Optional[str | Path] = None, artifacts_dir: str | Path = "artifacts"
) -> List[str]:
    """Read the train-only lexicon, keeping words spellable in the model alphabet."""
    lex_path = _pick_lexicon_path(lexicon_path, artifacts_dir)
    words: List[str] = []
    with lex_path.open("r", encoding="utf-8") as f:
        for line in f:
            w = (line.strip() or "").lower()
            if not w or " " in w:
                continue
            if all(tt.char_to_id(ch) is not None for ch in w):
                words.append(w)
    return sorted(set(words))


def build_trie_from_lexicon(
    tt, lexicon_path: Optional[str | Path] = None, artifacts_dir: str | Path = "artifacts"
) -> TrieNode:
    """Build the prefix trie T from the closed-vocabulary lexicon L."""
    root = TrieNode()
    for w in load_lexicon(tt, lexicon_path, artifacts_dir):
        root.add(w)
    root.utterance_level = False
    return root


def build_grammar_trie(train_dir: str | Path = "data/train_emg", tt=None) -> TrieNode:
    """
    Trie over complete valid utterances, induced from TRAIN transcripts.

    Unlike the word-level trie this one is *not* reset at a space: the space is
    an ordinary edge, so the structure of the whole transcript is enforced. That
    is what rules out "1045 am am" and "august 06 1925 august", which the flat
    lexicon accepts.
    """
    from scripts.grammar import generate, induce, read_split, verify_coverage

    texts = read_split(tt, train_dir)
    shapes, ymin, ymax = induce(texts)
    missing = verify_coverage(texts, ymin, ymax)
    if missing:
        raise ValueError(f"grammar cannot generate {len(missing)} TRAIN transcripts, e.g. {missing[:3]}")

    root = TrieNode()
    for u in generate(shapes, ymin, ymax):
        root.add(u)
    root.utterance_level = True
    root.year_range = (ymin, ymax)
    return root


# =============================================================================
# Character n-gram language model (n = 5)
# =============================================================================
class CharNgramLM:
    """Add-alpha smoothed character n-gram LM (Eq. 15)."""

    def __init__(self, n: int = 5, alpha: float = 0.1, vocab: Optional[List[str]] = None):
        assert n >= 2
        self.n = int(n)
        self.alpha = float(alpha)
        self.vocab = vocab or []
        self._V = max(1, len(self.vocab))
        self.ctx_counts: Dict[str, int] = {}
        self.ctx_ch_counts: Dict[Tuple[str, str], int] = {}
        self._pad = "" * (self.n - 1)

    def fit(self, sequences: Iterable[str]) -> "CharNgramLM":
        for s in sequences:
            if not s:
                continue
            s = re.sub(r"\s+", " ", str(s).strip().lower())
            if not s:
                continue
            padded = self._pad + s
            for i, ch in enumerate(s):
                ctx = padded[i : i + self.n - 1]
                self.ctx_counts[ctx] = self.ctx_counts.get(ctx, 0) + 1
                self.ctx_ch_counts[(ctx, ch)] = self.ctx_ch_counts.get((ctx, ch), 0) + 1
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


def _iter_jsons(train_dir: Path) -> Iterable[Path]:
    seen = set()
    for js in sorted(train_dir.glob("*.json")):
        rp = js.resolve()
        if rp not in seen:
            seen.add(rp)
            yield js
    for npy in sorted(train_dir.glob("*.npy")):
        js = npy.with_suffix(".json")
        if js.exists():
            rp = js.resolve()
            if rp not in seen:
                seen.add(rp)
                yield js


def _read_text(js_path: Path) -> str:
    obj = json.loads(js_path.read_text(encoding="utf-8"))
    return obj.get("text", obj.get("transcript", obj.get("label", "")))


def train_char_5gram(tt, train_dir: str | Path = "data/train_emg", alpha: float = 0.1) -> CharNgramLM:
    """Fit the character 5-gram LM on TRAIN transcripts only."""
    train_dir = Path(train_dir)
    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")

    lm = CharNgramLM(n=5, alpha=float(alpha), vocab=list(getattr(tt, "BASE_CHARS", "")))
    lines = [tt.clean(_read_text(js)) for js in _iter_jsons(train_dir)]
    lm.fit([s for s in lines if s])
    return lm


# =============================================================================
# CTC forward scoring (Eq. 9), log-space and vectorised over the label axis
# =============================================================================
def ctc_forward_logprob(ctc_logits: torch.Tensor, labels: Sequence[int], blank_id: int) -> float:
    """
    log p_CTC(labels | E) via the standard CTC forward recursion.

    ctc_logits : (T', V) unnormalised adapter-head outputs
    labels     : target ids, no BOS/EOS/PAD (spaces included)
    """
    if ctc_logits.ndim != 2:
        raise ValueError(f"ctc_logits must be (T,V), got {tuple(ctc_logits.shape)}")
    T, V = ctc_logits.shape
    if T == 0:
        return float("-inf")
    if not (0 <= blank_id < V):
        raise ValueError(f"blank_id out of range: {blank_id} vs V={V}")

    logp = torch.log_softmax(ctc_logits.float(), dim=-1)

    ext: List[int] = [blank_id]
    for l in labels:
        l = int(l)
        if l == blank_id:
            continue
        ext.append(l)
        ext.append(blank_id)
    S = len(ext)

    if T < (S + 1) // 2:  # too few frames to emit the label sequence
        return float("-inf")

    ext_t = torch.tensor(ext, device=logp.device, dtype=torch.long)
    emit = logp.index_select(1, ext_t)  # (T, S)

    neg_inf = torch.finfo(logp.dtype).min
    alpha = torch.full((S,), neg_inf, device=logp.device, dtype=logp.dtype)
    alpha[0] = emit[0, 0]
    if S > 1:
        alpha[1] = emit[0, 1]

    # skip[s] is True when the s-2 transition is legal
    skip = torch.zeros(S, dtype=torch.bool, device=logp.device)
    if S > 2:
        skip[2:] = (ext_t[2:] != blank_id) & (ext_t[2:] != ext_t[:-2])

    for t in range(1, T):
        a0 = alpha
        a1 = torch.cat([torch.full((1,), neg_inf, device=logp.device, dtype=logp.dtype), alpha[:-1]])
        a2 = torch.cat([torch.full((2,), neg_inf, device=logp.device, dtype=logp.dtype), alpha[:-2]])
        a2 = torch.where(skip, a2, torch.full_like(a2, neg_inf))
        alpha = torch.logsumexp(torch.stack([a0, a1, a2], dim=0), dim=0) + emit[t]

    out = alpha[S - 1] if S == 1 else torch.logsumexp(torch.stack([alpha[S - 1], alpha[S - 2]]), dim=0)
    return float(out.item())


# =============================================================================
# Configuration
# =============================================================================
@dataclass
class NSConfig:
    """Decoding hyperparameters. Defaults are the Table I final values."""

    # constrained beam (Algorithm 2)
    beam_size: int = 6            # K
    max_len: int = 64             # L_max
    top_m: int = 16               # M completed hypotheses kept for reranking
    min_eos_len: int = 8          # m
    per_step_top: int = 50        # per-hypothesis expansion cap

    # symbolic terms (Eq. 16)
    beta: float = 0.55            # char 5-gram fusion weight
    kappa: float = 0.40           # valid-boundary reward
    gamma: float = 0.45           # invalid-word (OOV) penalty

    # length normalisation (Eq. 17)
    alpha: float = 0.60
    delta: float = 5.0

    # ablation switches (Sec. VI-C2 / Table IV)
    use_trie: bool = True
    use_eos_gating: bool = True

    # CTC-assisted inference (Sec. IV-F, IV-G)
    lambda_fix: float = 0.25      # fixed lambda for the reranking baseline
    lambda0: float = 0.25
    a: float = 0.20
    b: float = 0.20
    lambda_min: float = 0.0
    lambda_max: float = 0.6


@dataclass
class Candidate:
    """One partial or completed hypothesis in the constrained beam."""

    ids: List[int]        # includes BOS; may end with EOS
    node: TrieNode        # trie state for the token since the last space
    ctx: str              # char n-gram context
    token: str            # current token text since the last space
    text: str             # visible string so far
    ar_logp: float
    lm_logp: float
    wb: int               # WB(s), Eq. 12
    oov: int              # OOV(s), Eq. 13
    ent_sum: float        # running sum of AR entropies, for Eq. 19
    steps: int
    ended: bool

    @property
    def vis_len(self) -> int:
        return len(self.text)

    def raw_score(self, cfg: NSConfig, ctc_logp: float = 0.0, lam: float = 0.0) -> float:
        """RawScore(s), Eq. 16."""
        return (
            self.ar_logp
            + lam * float(ctc_logp)
            + cfg.beta * self.lm_logp
            + cfg.kappa * float(self.wb)
            - cfg.gamma * float(self.oov)
        )

    def score(self, cfg: NSConfig, ctc_logp: float = 0.0, lam: float = 0.0) -> float:
        """Score(s), Eq. 17 (length-normalised)."""
        L = max(1, self.vis_len)
        return self.raw_score(cfg, ctc_logp, lam) / ((L + cfg.delta) ** cfg.alpha)

    def score0(self, cfg: NSConfig) -> float:
        """Candidate-generation score: Eq. 17 with lambda_inf = 0."""
        return self.score(cfg, 0.0, 0.0)


def _entropy(logits: torch.Tensor) -> float:
    p = torch.softmax(logits.float(), dim=-1)
    return float((-(p * torch.log(p.clamp_min(1e-12))).sum()).item())


# =============================================================================
# Algorithm 2: trie-aware constrained beam search
# =============================================================================
def generate_candidates(
    step_fn: Callable[[List[List[int]]], torch.Tensor],
    tt,
    trie_root: TrieNode,
    charlm: CharNgramLM,
    cfg: NSConfig,
) -> List[Candidate]:
    """
    Trie-aware NS beam search (Algorithm 2), run with lambda_inf = 0.

    `step_fn` takes a *list* of id sequences and returns (B, V) logits, so a
    whole beam is scored in one LLM call.

    Constraint semantics follow Sec. IV-E1:
      * a non-space character c is allowed only if delta(q, c) exists in the trie
        (when `cfg.use_trie` is off, every alphabet character is allowed);
      * a space is *always* allowed, and triggers the boundary decision --
        the completed token either counts towards WB or towards OOV;
      * EOS is allowed only when |s| >= m and the current token is a complete
        word in L (when `cfg.use_eos_gating` is off, EOS is always allowed and
        the final boundary is still scored as valid/invalid).
    """
    space_id = tt.char_to_id(" ")
    if space_id is None:
        raise ValueError("TextTransform must include space in BASE_CHARS")

    alphabet = [c for c in getattr(tt, "BASE_CHARS", "") if c != " "]
    # An utterance-level trie encodes the whole transcript, so trie state is not
    # reset at word boundaries and `is_word` marks a complete transcript.
    utt_level = bool(cfg.use_trie and getattr(trie_root, "utterance_level", False))

    init = Candidate(
        ids=[tt.BOS_IDX],
        node=trie_root,
        ctx=charlm.init_ctx(),
        token="",
        text="",
        ar_logp=0.0,
        lm_logp=0.0,
        wb=0,
        oov=0,
        ent_sum=0.0,
        steps=0,
        ended=False,
    )

    beam: List[Candidate] = [init]
    completed: List[Candidate] = []

    for _ in range(int(cfg.max_len)):
        active = [h for h in beam if not h.ended]
        done = [h for h in beam if h.ended]
        if not active:
            break

        logits = step_fn([h.ids for h in active])
        logp_all = torch.log_softmax(logits.float(), dim=-1)

        # Terminated hypotheses stay in the beam so they can still win the prune.
        new_beam: List[Candidate] = list(done)

        for i, hyp in enumerate(active):
            logp = logp_all[i]
            ent = _entropy(logits[i])

            # ---- build the allowed action set -----------------------------
            allowed: List[Tuple[int, str]] = []  # (token id, action kind)

            if cfg.use_trie:
                for ch in hyp.node.children.keys():
                    tid = tt.char_to_id(ch)
                    if tid is not None:
                        allowed.append((tid, "char"))
            else:
                for ch in alphabet:
                    tid = tt.char_to_id(ch)
                    if tid is not None:
                        allowed.append((tid, "char"))

            # A space is always allowed (Algorithm 2, line 14) -- except under an
            # utterance-level grammar, where the space is an ordinary trie edge
            # and is legal only where the grammar puts a word boundary.
            if hyp.token:
                if utt_level:
                    if " " in hyp.node.children:
                        allowed.append((space_id, "space"))
                else:
                    allowed.append((space_id, "space"))

            # EOS gating (Algorithm 2, line 18).
            if cfg.use_eos_gating:
                if hyp.vis_len >= int(cfg.min_eos_len) and hyp.token and hyp.node.is_word:
                    allowed.append((tt.EOS_IDX, "eos"))
            else:
                if hyp.token:
                    allowed.append((tt.EOS_IDX, "eos"))

            if not allowed:
                continue

            ids_t = torch.tensor([a[0] for a in allowed], device=logp.device, dtype=torch.long)
            sel = logp.index_select(0, ids_t)
            k = min(int(cfg.per_step_top), int(sel.numel()))
            vals, idxs = torch.topk(sel, k)

            for v, j in zip(vals.tolist(), idxs.tolist()):
                tid, kind = allowed[int(j)]
                child = _expand(hyp, tid, kind, float(v), ent, tt, trie_root, charlm)
                new_beam.append(child)
                # Algorithm 2 lines 19-20: a terminated hypothesis joins both
                # B_next and the completed pool C_all, so a completion is kept
                # even if this step's prune drops it from the beam.
                if kind == "eos":
                    completed.append(child)

        if len(new_beam) == len(done):
            break

        new_beam.sort(key=lambda c: c.score0(cfg), reverse=True)
        beam = new_beam[: int(cfg.beam_size)]

        if all(c.ended for c in beam):
            break

    pool = completed if completed else list(beam)
    pool = _dedup_by_text(pool, cfg)
    pool.sort(key=lambda c: c.score0(cfg), reverse=True)
    return pool[: int(cfg.top_m)]


def _dedup_by_text(cands: List[Candidate], cfg: NSConfig) -> List[Candidate]:
    """Keep the best-scoring Candidate per distinct transcript."""
    best: Dict[str, Candidate] = {}
    for c in cands:
        t = c.text.strip()
        cur = best.get(t)
        if cur is None or c.score0(cfg) > cur.score0(cfg):
            best[t] = c
    return list(best.values())


def _expand(
    hyp: Candidate,
    tid: int,
    kind: str,
    logp_tid: float,
    ent_parent: float,
    tt,
    trie_root: TrieNode,
    charlm: CharNgramLM,
) -> Candidate:
    """Apply one action to a hypothesis, updating trie state and WB/OOV."""
    ids2 = hyp.ids + [tid]
    ar2 = hyp.ar_logp + float(logp_tid)
    ent2 = hyp.ent_sum + float(ent_parent)
    steps2 = hyp.steps + 1

    wb2, oov2 = hyp.wb, hyp.oov
    lm2 = hyp.lm_logp
    ctx2 = hyp.ctx

    if kind == "eos":
        # Final boundary: the trailing token is scored like any other boundary
        # (Eq. 12-13 count EOS among the boundary positions B(s)).
        if hyp.node.is_word and hyp.token:
            wb2 += 1
        else:
            oov2 += 1
        return Candidate(
            ids2, hyp.node, ctx2, hyp.token, hyp.text, ar2, lm2, wb2, oov2, ent2, steps2, True
        )

    ch = tt.id_to_char(tid)
    if ch is None:
        return Candidate(
            ids2, trie_root, ctx2, "", hyp.text, ar2, lm2, wb2, oov2 + 1, ent2, steps2, hyp.ended
        )

    # The n-gram LM scores every visible character, spaces included (Eq. 15).
    lm2 += charlm.logp_next(ctx2, ch)
    ctx2 = charlm.update_ctx(ctx2, ch)
    text2 = hyp.text + ch

    if kind == "space":
        # Under an utterance-level grammar the space is a real trie edge, so the
        # boundary is valid by construction and the state advances instead of
        # returning to the root; OOV is then unreachable while the trie is on.
        if getattr(trie_root, "utterance_level", False):
            nxt = hyp.node.children.get(" ")
            wb2 += 1
            return Candidate(
                ids2, nxt if nxt is not None else hyp.node, ctx2, "", text2,
                ar2, lm2, wb2, oov2, ent2, steps2, hyp.ended,
            )
        if hyp.node.is_word and hyp.token:
            wb2 += 1
        else:
            oov2 += 1
        return Candidate(
            ids2, trie_root, ctx2, "", text2, ar2, lm2, wb2, oov2, ent2, steps2, hyp.ended
        )

    # Ordinary character: advance the trie state.
    nxt = hyp.node.children.get(ch)
    if nxt is None:
        # Only reachable with the trie constraint disabled.
        nxt = TrieNode()
    return Candidate(
        ids2, nxt, ctx2, hyp.token + ch, text2, ar2, lm2, wb2, oov2, ent2, steps2, hyp.ended
    )


# =============================================================================
# Confidence-adaptive fusion (Eq. 19-21)
# =============================================================================
def compute_adaptive_lambda(
    ref: Candidate, ctc_logits: torch.Tensor, tt, cfg: NSConfig
) -> Tuple[float, float, float]:
    """
    lambda_inf = clip(lambda0 + a*u_AR - b*u_CTC, [lambda_min, lambda_max])

    u_AR  : mean AR entropy along the reference hypothesis  (Eq. 19)
    u_CTC : mean blank posterior over adapter steps         (Eq. 20)
    """
    u_ar = float(ref.ent_sum / max(1, ref.steps))

    blank_id = int(getattr(tt, "PAD_IDX", ctc_logits.shape[-1] - 1))
    probs = torch.softmax(ctc_logits.float(), dim=-1)
    if not (0 <= blank_id < probs.shape[-1]):
        blank_id = probs.shape[-1] - 1
    u_ctc = float(probs[:, blank_id].mean().item())

    lam = float(cfg.lambda0) + float(cfg.a) * u_ar - float(cfg.b) * u_ctc
    lam = max(float(cfg.lambda_min), min(float(cfg.lambda_max), lam))
    return lam, u_ar, u_ctc


# =============================================================================
# Decoding entry points (the seven conditions of Sec. V-C)
# =============================================================================
def decode_ns(
    step_fn: Callable[[List[List[int]]], torch.Tensor],
    tt,
    trie_root: TrieNode,
    charlm: CharNgramLM,
    cfg: NSConfig,
) -> Tuple[str, List[Candidate], dict]:
    """NS constrained beam only (no CTC evidence)."""
    cands = generate_candidates(step_fn, tt, trie_root, charlm, cfg)
    if not cands:
        return "", [], {"n_candidates": 0}
    best = max(cands, key=lambda c: c.score0(cfg))
    return best.text.strip(), cands, {"n_candidates": len(cands)}


def select_with_ctc(
    ns_text: str,
    ctc_text: str,
    tt,
    ar_score_fn: Callable[[List[List[int]]], List[float]],
    length_norm: bool = False,
    cfg: Optional[NSConfig] = None,
) -> Tuple[str, dict]:
    """
    NS + CTC candidate selection (Eq. 18).

    A restricted two-candidate comparison: the alignment-driven greedy-CTC
    transcript versus the lexicon-constrained NS transcript, chosen by AR
    teacher-forced string score under the same EMG evidence. No new search.

    Two safeguards on top of the literal equation:

    * An empty transcript is never selected over a non-empty one. Eq. 18
      compares unnormalised sums of log probabilities, which always favours
      the shorter string -- so whenever the CTC head emits all blanks the
      empty string wins every comparison outright and the step silently
      deletes the utterance. That is a degenerate case, not a modelling
      decision, so it is excluded.

    * `length_norm` optionally applies the same length normalisation used
      everywhere else (Eq. 17) before comparing. Default False, i.e. the
      paper's literal Eq. 18; the two candidates are usually of similar
      length once the model is trained, so this rarely changes the outcome.
    """
    options = [ns_text.strip(), ctc_text.strip()]
    seqs = [tt.text_to_ids(t, add_bos_eos=True) for t in options]
    raw = ar_score_fn(seqs)

    cfg = cfg or NSConfig()
    if length_norm:
        cmp = [r / ((max(1, len(t)) + cfg.delta) ** cfg.alpha) for r, t in zip(raw, options)]
    else:
        cmp = list(raw)

    # Never let an empty transcript beat a non-empty one.
    viable = [i for i, t in enumerate(options) if t]
    if not viable:
        viable = list(range(len(options)))

    j = max(viable, key=lambda i: cmp[i])
    return options[j], {
        "ar_score_ns": raw[0],
        "ar_score_ctc": raw[1],
        "picked": "ns" if j == 0 else "ctc",
        "length_norm": bool(length_norm),
    }


def rerank_with_ctc(
    candidates: List[Candidate],
    ctc_logits: torch.Tensor,
    tt,
    cfg: NSConfig,
    adaptive: bool = True,
) -> Tuple[str, dict]:
    """
    Algorithm 3: joint AR+CTC reranking over the completed candidate set, with
    either the fixed fusion weight lambda_fix or the per-utterance adaptive one.
    """
    if not candidates:
        return "", {"lambda": None, "u_ar": None, "u_ctc": None, "candidates": []}

    ref = max(candidates, key=lambda c: c.score0(cfg))
    if adaptive:
        lam, u_ar, u_ctc = compute_adaptive_lambda(ref, ctc_logits, tt, cfg)
    else:
        lam, u_ar, u_ctc = float(cfg.lambda_fix), None, None

    blank_id = int(getattr(tt, "PAD_IDX", ctc_logits.shape[-1] - 1))

    best: Optional[Candidate] = None
    best_score = float("-inf")
    info: List[dict] = []

    for cand in candidates:
        text = cand.text.strip()
        labels = tt.text_to_ids(text, add_bos_eos=False)
        ctc_lp = ctc_forward_logprob(ctc_logits, labels, blank_id=blank_id)
        score = cand.score(cfg, ctc_lp, lam)
        info.append(
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
    return best.text.strip(), {
        "lambda": lam,
        "u_ar": u_ar,
        "u_ctc": u_ctc,
        "candidates": info,
    }


# -----------------------------------------------------------------------------
# Backwards-compatible wrapper (older call sites used this single entry point)
# -----------------------------------------------------------------------------
def decode_ns_with_rerank(
    step_fn: Callable[[List[List[int]]], torch.Tensor],
    ctc_logits: torch.Tensor,
    tt,
    trie_root: TrieNode,
    charlm: CharNgramLM,
    cfg: Optional[NSConfig] = None,
    adaptive: bool = True,
    joint: bool = True,
) -> Tuple[str, dict]:
    cfg = cfg or NSConfig()
    ns_text, cands, _ = decode_ns(step_fn, tt, trie_root, charlm, cfg)
    if not joint or not cands:
        return ns_text, {"lambda": 0.0, "u_ar": None, "u_ctc": None, "candidates": []}
    return rerank_with_ctc(cands, ctc_logits, tt, cfg, adaptive=adaptive)


# -----------------------------------------------------------------------------
# Grammar-constrained CTC decoding
# -----------------------------------------------------------------------------
# Applying the grammar to the AR beam is a no-op: measured on validation, the AR
# beam already emits a grammar-valid utterance 50/50 for every backbone, so
# there is nothing for the constraint to exclude. The CTC channel is the
# opposite -- only 22-31 of 50 of its greedy outputs are valid, the rest being
# malformed strings the grammar rejects outright ("december 24 19841",
# "thursday fvebruer 02", "saturday mach 17").
#
# That is where the symbolic constraint has purchase, so this decodes the CTC
# posteriors *through* the trie rather than repairing the argmax afterwards:
# standard CTC prefix beam search in which the set of legal next characters at
# every prefix is exactly the set of edges leaving that prefix's trie node, and
# only prefixes resting on a terminal node are admissible answers. Because the
# constraint is applied during search it can recover mass from paths that greedy
# collapse discards, which nearest-valid-string projection cannot.
_NEG_INF = float("-inf")


def _logaddexp(a: float, b: float) -> float:
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    m = a if a > b else b
    return m + math.log(math.exp(a - m) + math.exp(b - m))


def ctc_grammar_beam(
    ctc_logits: torch.Tensor,
    trie_root: "TrieNode",
    tt,
    beam: int = 32,
    charlm: Optional["CharNgramLM"] = None,
    beta: float = 0.0,
    blank_id: Optional[int] = None,
    return_pool: int = 1,
    word_restart: bool = False,
) -> List[Tuple[str, float]]:
    """Grammar-constrained CTC prefix beam search.

    Returns up to `return_pool` (text, logprob) pairs, best first. The list is
    empty only if no grammar-valid prefix survives the beam, which the caller
    must treat as "fall back to the unconstrained output" rather than as an
    empty transcript.

    `word_restart` is what makes a *word-level* trie usable here at all. In the
    utterance trie the space is an ordinary edge, so a whole transcript is one
    path from root to a terminal node. A word trie has no such edge: its
    terminal nodes are single words, so without this flag the beam completes
    after one word and the decoder emits "december" for "december 12 1904".
    With it, a space at a word-terminal node returns to the root and the
    lexicon can spell multi-word transcripts -- the fair word-level baseline
    for the constraint-placement comparison, rather than a straw man.
    """
    blank = tt.PAD_IDX if blank_id is None else blank_id
    logp = torch.log_softmax(ctc_logits.float(), dim=-1).cpu().numpy()
    T = logp.shape[0]

    cid = {ch: tt.char_to_id(ch) for ch in tt.BASE_CHARS}
    use_lm = charlm is not None and beta != 0.0

    # prefix -> [log p(ends in blank), log p(ends in non-blank), trie node, lm ctx]
    init_ctx = charlm.init_ctx() if use_lm else None
    beams: Dict[str, list] = {"": [0.0, _NEG_INF, trie_root, init_ctx]}

    for t in range(T):
        nxt: Dict[str, list] = {}

        def bump(key, node, ctx, idx, val):
            e = nxt.get(key)
            if e is None:
                e = [_NEG_INF, _NEG_INF, node, ctx]
                nxt[key] = e
            e[idx] = _logaddexp(e[idx], val)

        for pref, (lb, lnb, node, ctx) in beams.items():
            ptot = _logaddexp(lb, lnb)

            # (a) emit blank -- prefix unchanged, now ends in blank
            bump(pref, node, ctx, 0, ptot + logp[t, blank])

            # (b) repeat the last character -- collapses, prefix unchanged
            if pref:
                k = cid.get(pref[-1])
                if k is not None:
                    bump(pref, node, ctx, 1, lnb + logp[t, k])

            # (c) extend, but only along edges the GRAMMAR allows
            for ch, child in node.children.items():
                k = cid.get(ch)
                if k is None:
                    continue
                s = logp[t, k]
                if use_lm:
                    s += beta * charlm.logp_next(ctx, ch)
                # A repeat of the previous character needs a blank between the
                # two frames, so it may only extend the blank-ending mass.
                src = lb if (pref and ch == pref[-1]) else ptot
                if src == _NEG_INF:
                    continue
                nctx = charlm.update_ctx(ctx, ch) if use_lm else None
                bump(pref + ch, child, nctx, 1, src + s)

            # (d) word boundary: only for a word-level trie, and only from a
            # node that completes a lexicon entry.
            if word_restart and getattr(node, "is_word", False) and pref:
                k = cid.get(" ")
                if k is not None:
                    s = logp[t, k]
                    if use_lm:
                        s += beta * charlm.logp_next(ctx, " ")
                    src = lb if pref[-1] == " " else ptot
                    if src != _NEG_INF:
                        nctx = charlm.update_ctx(ctx, " ") if use_lm else None
                        bump(pref + " ", trie_root, nctx, 1, src + s)

        if not nxt:
            break
        beams = dict(
            sorted(nxt.items(), key=lambda kv: -_logaddexp(kv[1][0], kv[1][1]))[:beam]
        )

    # Only prefixes resting on a terminal node are complete utterances.
    # float(), not the raw numpy scalar: these scores get written into the
    # output jsonl and numpy float32 is not JSON-serialisable.
    done = [
        (p, float(_logaddexp(v[0], v[1])))
        for p, v in beams.items()
        if p and getattr(v[2], "is_word", False)
    ]
    done.sort(key=lambda kv: -kv[1])
    return done[:max(1, return_pool)]
