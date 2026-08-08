#!/usr/bin/env python3
"""
scripts/ar_decode.py

Autoregressive decoding utilities shared by training-time validation and
inference-time evaluation, so both use exactly one implementation.

The paper fixes the EMG prefix at inference time (Sec. IV-B2: "At inference,
the EMG prefix is fixed and generation starts from BOS"). We exploit that by
running the frozen LLM over the prefix

    U_prefix = [U_soft ; U_inst ; H_emg]

exactly once per utterance, keeping its KV cache, and re-running only the
short character suffix at every decoding step. This is numerically identical
to recomputing the whole sequence (verified to ~1e-7) but far cheaper: the
prefix is ~90 positions while the transcripts are ~20 characters.

Exposed interface (matches Algorithm 1's STEP function):
  PrefixCachedAR.step(list_of_id_lists) -> logits [B, V]
  PrefixCachedAR.score(list_of_id_lists) -> teacher-forced log p_AR(s | E)

plus greedy / beam decoding built on top of `step`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


class PrefixCachedAR:
    """
    Cached AR stepper for a single utterance (the EMG prefix is fixed).

    Parameters
    ----------
    model : EMGToFrozenLLM
    tt : TextTransform
    prefix_embeds : (1, P, H)
    prefix_mask : (1, P)
    capacity : batch width used for the cached prefix. Requests larger than
        `capacity` are processed in chunks; smaller ones pad up to it.
    use_cache : set False to fall back to full recomputation (debugging).
    """

    def __init__(
        self,
        model,
        tt,
        prefix_embeds: torch.Tensor,
        prefix_mask: torch.Tensor,
        capacity: int = 8,
        use_cache: bool = True,
    ) -> None:
        self.model = model
        self.tt = tt
        self.device = prefix_embeds.device
        self.llm_dtype = next(model.llm.parameters()).dtype

        self.capacity = max(1, int(capacity))
        self.prefix_embeds = prefix_embeds.to(dtype=self.llm_dtype)
        self.prefix_mask = prefix_mask.long()
        self.P = int(self.prefix_embeds.size(1))

        self._cache = None
        if use_cache:
            self._cache = self._build_cache(self.capacity)

    # ------------------------------------------------------------------
    # cache
    # ------------------------------------------------------------------
    def _build_cache(self, batch: int):
        """Run the frozen LLM over the fixed prefix once, at batch width `batch`."""
        emb = self.prefix_embeds.expand(batch, -1, -1).contiguous()
        msk = self.prefix_mask.expand(batch, -1).contiguous()
        try:
            with torch.no_grad():
                out = self.model.llm(inputs_embeds=emb, attention_mask=msk, use_cache=True)
            pkv = getattr(out, "past_key_values", None)
            if pkv is None or not hasattr(pkv, "crop"):
                return None
            return pkv
        except Exception:
            # Any backbone that does not support incremental caching still works,
            # just without the speedup.
            return None

    # ------------------------------------------------------------------
    # core forward over a batch of character prefixes
    # ------------------------------------------------------------------
    def _forward_chars(self, ids_batch: List[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run the LLM over `ids_batch` (right-padded) conditioned on the cached prefix.

        Returns
        -------
        hidden : (B, L_max, H) hidden states at the character positions
        lengths : (B,) true length of each row
        """
        pad = self.tt.PAD_IDX
        b = len(ids_batch)
        lengths = torch.tensor([len(x) for x in ids_batch], device=self.device, dtype=torch.long)
        L = int(lengths.max().item())

        ids = torch.full((b, L), pad, dtype=torch.long, device=self.device)
        for i, seq in enumerate(ids_batch):
            if len(seq):
                ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=self.device)

        char_embeds = self.model.char_embed(ids).to(dtype=self.llm_dtype)
        char_mask = (ids != pad).long()

        if self._cache is not None and b <= self.capacity:
            # Pad the batch up to the cached width so the cache is never resliced.
            if b < self.capacity:
                repeat = self.capacity - b
                char_embeds = torch.cat([char_embeds, char_embeds[:1].expand(repeat, -1, -1)], dim=0)
                char_mask = torch.cat([char_mask, char_mask[:1].expand(repeat, -1)], dim=0)

            attn = torch.cat(
                [self.prefix_mask.expand(self.capacity, -1), char_mask], dim=1
            )
            with torch.no_grad():
                out = self.model.llm(
                    inputs_embeds=char_embeds,
                    attention_mask=attn,
                    past_key_values=self._cache,
                    use_cache=True,
                )
            # Restore the cache to the pure-prefix state for the next call.
            self._cache.crop(self.P)
            hidden = out.last_hidden_state[:b]
        else:
            # Fallback / oversized batch: recompute prefix + chars from scratch.
            emb = torch.cat([self.prefix_embeds.expand(b, -1, -1), char_embeds], dim=1)
            attn = torch.cat([self.prefix_mask.expand(b, -1), char_mask], dim=1)
            with torch.no_grad():
                out = self.model.llm(inputs_embeds=emb, attention_mask=attn)
            hidden = out.last_hidden_state[:, self.P :, :]

        return hidden, lengths

    # ------------------------------------------------------------------
    # Algorithm 1: STEP(s)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(self, ids_batch: List[Sequence[int]]) -> torch.Tensor:
        """Next-character logits for each partial string. Returns (B, V)."""
        if not ids_batch:
            return torch.empty(0, self.tt.VOCAB_SIZE, device=self.device)

        outs: List[torch.Tensor] = []
        chunk = self.capacity if self._cache is not None else max(1, self.capacity)
        for i in range(0, len(ids_batch), chunk):
            part = ids_batch[i : i + chunk]
            hidden, lengths = self._forward_chars(part)
            idx = (lengths - 1).clamp(min=0)
            last = hidden[torch.arange(hidden.size(0), device=self.device), idx]
            outs.append(self.model.lm_head(last.to(self.model.lm_head.weight.dtype)))
        return torch.cat(outs, dim=0)

    # ------------------------------------------------------------------
    # Teacher-forced string scoring (Eq. 14 / Eq. 18)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def score(self, ids_batch: List[Sequence[int]]) -> List[float]:
        """
        Teacher-forced log p_AR(s | E) for full id sequences.

        Each element must be a complete sequence [BOS, c_1, ..., c_L, EOS];
        the score sums log p over every predicted position (characters + EOS).
        """
        if not ids_batch:
            return []

        scores: List[float] = []
        chunk = self.capacity if self._cache is not None else max(1, self.capacity)
        for i in range(0, len(ids_batch), chunk):
            part = [list(s) for s in ids_batch[i : i + chunk]]
            inp = [s[:-1] if len(s) > 1 else s for s in part]
            hidden, lengths = self._forward_chars(inp)
            logits = self.model.lm_head(hidden.to(self.model.lm_head.weight.dtype))
            logp = F.log_softmax(logits.float(), dim=-1)

            for r, seq in enumerate(part):
                if len(seq) < 2:
                    scores.append(0.0)
                    continue
                tgt = torch.tensor(seq[1:], dtype=torch.long, device=self.device)
                n = int(tgt.numel())
                got = logp[r, :n].gather(1, tgt.unsqueeze(1)).sum()
                scores.append(float(got.item()))
        return scores

    # ------------------------------------------------------------------
    # Decoders
    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy(self, max_len: int, min_len: int = 1) -> List[int]:
        """AR greedy decoding (K = 1)."""
        tt = self.tt
        ids = [tt.BOS_IDX]
        for t in range(int(max_len)):
            logits = self.step([ids])[0]
            logp = F.log_softmax(logits.float(), dim=-1)
            if t + 1 < int(min_len):
                logp[tt.EOS_IDX] = -1e9
            nxt = int(torch.argmax(logp).item())
            ids.append(nxt)
            if nxt == tt.EOS_IDX:
                break
        return ids

    @torch.no_grad()
    def beam(
        self,
        beam_size: int,
        max_len: int,
        alpha: float,
        delta: float,
        min_len: int = 1,
    ) -> List[int]:
        """
        Standard AR beam search with length normalisation (Eq. 17 with the
        symbolic terms and the CTC term switched off).
        """
        tt = self.tt
        K = int(beam_size)

        def norm(ids: List[int], sc: float) -> float:
            L = max(1, len(ids) - 1)  # exclude BOS
            return sc / ((L + float(delta)) ** float(alpha))

        hyps: List[Tuple[List[int], float]] = [([tt.BOS_IDX], 0.0)]
        finished: List[Tuple[List[int], float]] = []

        for t in range(int(max_len)):
            if not hyps:
                break
            logits = self.step([h[0] for h in hyps])
            logp = F.log_softmax(logits.float(), dim=-1)

            cand: List[Tuple[List[int], float]] = []
            for i, (ids, sc) in enumerate(hyps):
                row = logp[i]
                if t + 1 < int(min_len):
                    row = row.clone()
                    row[tt.EOS_IDX] = -1e9
                topv, topi = torch.topk(row, k=min(K, row.numel()))
                for lp, tok in zip(topv.tolist(), topi.tolist()):
                    tok = int(tok)
                    nxt = (ids + [tok], sc + float(lp))
                    (finished if tok == tt.EOS_IDX else cand).append(nxt)

            if not cand:
                break
            cand.sort(key=lambda x: norm(x[0], x[1]), reverse=True)
            hyps = cand[:K]

        if finished:
            finished.sort(key=lambda x: norm(x[0], x[1]), reverse=True)
            return finished[0][0]
        hyps.sort(key=lambda x: norm(x[0], x[1]), reverse=True)
        return hyps[0][0]


def ctc_greedy_decode_ids(ctc_logits: torch.Tensor, blank_id: int) -> List[int]:
    """Greedy CTC collapse: argmax per frame, drop repeats, drop blanks."""
    ids = torch.argmax(ctc_logits, dim=-1).tolist()
    out: List[int] = []
    prev: Optional[int] = None
    for i in ids:
        i = int(i)
        if i == blank_id:
            prev = i
            continue
        if prev is None or i != prev:
            out.append(i)
        prev = i
    return out
