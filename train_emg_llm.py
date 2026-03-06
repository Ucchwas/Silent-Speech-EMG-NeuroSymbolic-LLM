#!/usr/bin/env python3
"""
train_emg_llm.py

Train the EMG-to-frozen-LLM model (character-level AR head + auxiliary CTC head).

My implementation details:
- Frozen decoder-only LLM backbone (θ frozen)
- Trainable EMG adapter + soft/instruction prompts + lightweight heads
- Dual supervision: AR cross-entropy + λ_ctc * CTC loss
- Normalization fitted on TRAIN split only and reused for val/test
- Checkpoint selection by neural-only AR beam validation decoding with
  K=6, Lmax=128, α=0.6, δ=5.0 (defaults)

Expected data layout (from split_data.py):
  data/train_emg/*.npy + *.json
  data/val_emg/*.npy + *.json

Each JSON should contain a transcript under a common key (e.g., "text").
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from transformers import AutoConfig, AutoModel


# -----------------------------------------------------------------------------
# Flexible imports (repo may use scripts/ and models/)
# -----------------------------------------------------------------------------
try:
    from models.emg_adapter import EMGAdapterV2
    from scripts.dataset_emg import make_loader
    from scripts.data_utils import FeatureNormalizer, TextTransform, load_normalizer, save_normalizer, NormState
except Exception:
    from emg_adapter import EMGAdapterV2
    from dataset_emg import make_loader
    from data_utils import FeatureNormalizer, TextTransform, load_normalizer, save_normalizer, NormState


# -----------------------------------------------------------------------------
# Repro
# -----------------------------------------------------------------------------
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ceil_div(a: torch.Tensor, b: int) -> torch.Tensor:
    return (a + (b - 1)) // b


# -----------------------------------------------------------------------------
# Normalizer (fit on train only)
# -----------------------------------------------------------------------------
def fit_normalizer_from_dir(
    features_dir: Path, pattern: str = "*.npy", eps: float = 1e-6
) -> FeatureNormalizer:
    files = sorted(features_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No feature files found in {features_dir} with pattern {pattern}")

    sum_x = None
    sum_x2 = None
    count = 0

    for fp in tqdm(files, desc=f"Fitting normalizer from {features_dir.name}", leave=False):
        x = np.load(fp).astype(np.float64)  # (T, D)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got {x.shape} in {fp}")
        if sum_x is None:
            d = x.shape[1]
            sum_x = np.zeros((d,), dtype=np.float64)
            sum_x2 = np.zeros((d,), dtype=np.float64)
        sum_x += x.sum(axis=0)
        sum_x2 += (x * x).sum(axis=0)
        count += x.shape[0]

    mean = sum_x / max(count, 1)
    var = sum_x2 / max(count, 1) - mean * mean
    var = np.maximum(var, eps)
    std = np.sqrt(var)

    norm = FeatureNormalizer(eps=eps)
    norm.state = NormState(mean=mean.astype(np.float32), std=std.astype(np.float32))
    return norm


# -----------------------------------------------------------------------------
# Metrics (no external deps)
# -----------------------------------------------------------------------------
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


def compute_wer(refs: List[str], hyps: List[str], tt: TextTransform) -> float:
    num = 0.0
    den = 0.0
    for r, h in zip(refs, hyps):
        r = tt.clean(r)
        h = tt.clean(h)
        r_w = r.split()
        h_w = h.split()
        if len(r_w) == 0:
            den += 1.0
            num += 0.0 if len(h_w) == 0 else 1.0
            continue
        num += _edit_distance(r_w, h_w)
        den += float(len(r_w))
    return float(num / max(den, 1.0))


def compute_cer(refs: List[str], hyps: List[str], tt: TextTransform) -> float:
    num = 0.0
    den = 0.0
    for r, h in zip(refs, hyps):
        r = tt.clean(r)
        h = tt.clean(h)
        r_c = list(r)
        h_c = list(h)
        if len(r_c) == 0:
            den += 1.0
            num += 0.0 if len(h_c) == 0 else 1.0
            continue
        num += _edit_distance(r_c, h_c)
        den += float(len(r_c))
    return float(num / max(den, 1.0))


# -----------------------------------------------------------------------------
# CTC targets (drop EOS, no padding counted via target_lens)
# -----------------------------------------------------------------------------
def build_ctc_targets(
    out_ids: torch.Tensor,
    out_lens: torch.Tensor,
    eos_idx: int,
    pad_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    out_ids: (B, L) targets (shifted) including EOS before padding
    out_lens: (B,) true lengths including EOS
    returns:
      targets_padded: (B, L_ctc_max) without EOS
      target_lens: (B,)
    """
    bsz, _ = out_ids.shape
    targets: List[torch.Tensor] = []
    target_lens: List[int] = []

    for i in range(bsz):
        L = int(out_lens[i].item())
        seq = out_ids[i, :L]

        # drop final EOS if present
        if L > 0 and int(seq[-1].item()) == eos_idx:
            seq = seq[:-1]
        else:
            eos_pos = (seq == eos_idx).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                j = int(eos_pos[0, 0].item())
                seq = torch.cat([seq[:j], seq[j + 1 :]], dim=0)

        targets.append(seq)
        target_lens.append(int(seq.numel()))

    max_len = max(target_lens) if target_lens else 0
    padded = out_ids.new_full((bsz, max_len), fill_value=pad_idx)
    for i, seq in enumerate(targets):
        if seq.numel() > 0:
            padded[i, : seq.numel()] = seq
    return padded, torch.tensor(target_lens, device=out_ids.device, dtype=torch.long)


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    base_model: str
    vocab_size: int
    in_dim: int
    subsample_factor: int = 4
    soft_prompt_len: int = 8
    inst_prompt_len: int = 8


class EMGToFrozenLLM(nn.Module):
    """Frozen decoder-only LLM conditioned on EMG prefix, character-level decoding."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        hf_cfg = AutoConfig.from_pretrained(cfg.base_model)
        hidden_size = getattr(hf_cfg, "hidden_size", None) or getattr(hf_cfg, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from HF config")
        self.hidden_size = int(hidden_size)

        # Prefer newer 'dtype' arg; fall back to older 'torch_dtype'
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        try:
            self.llm = AutoModel.from_pretrained(cfg.base_model, dtype=dtype)
        except TypeError:
            self.llm = AutoModel.from_pretrained(cfg.base_model, torch_dtype=dtype)

        self.llm.config.use_cache = False
        for p in self.llm.parameters():
            p.requires_grad_(False)

        self.adapter = EMGAdapterV2(in_dim=cfg.in_dim, hidden_size=self.hidden_size)
        self.char_embed = nn.Embedding(cfg.vocab_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, cfg.vocab_size, bias=False)
        self.ctc_head = nn.Linear(self.hidden_size, cfg.vocab_size)

        self.soft_prompt = nn.Parameter(torch.randn(1, cfg.soft_prompt_len, self.hidden_size) * 0.02)
        self.inst_prompt = nn.Parameter(torch.randn(1, cfg.inst_prompt_len, self.hidden_size) * 0.02)

    @property
    def subsample_factor(self) -> int:
        return int(self.cfg.subsample_factor)

    def forward(
        self,
        emg_feats: torch.Tensor,  # (B, T, D)
        emg_mask: torch.Tensor,   # (B, T) bool
        in_ids: torch.Tensor,     # (B, L)
        pad_idx: int,
    ) -> Dict[str, torch.Tensor]:
        B = emg_feats.size(0)

        # EMG -> embeddings (time subsampled inside adapter)
        emg_embeds = self.adapter(emg_feats)  # (B, T', H)
        Tprime = emg_embeds.size(1)

        # Subsampled EMG mask
        emg_lens = emg_mask.long().sum(dim=1)  # (B,)
        emg_lens_sub = ceil_div(emg_lens, self.subsample_factor).clamp(min=1)
        t = torch.arange(Tprime, device=emg_feats.device).unsqueeze(0)
        emg_mask_sub = (t < emg_lens_sub.unsqueeze(1)).long()  # (B, T')
        emg_embeds = emg_embeds * emg_mask_sub.unsqueeze(-1).to(emg_embeds.dtype)

        # CTC head on adapter stream
        ctc_logits = self.ctc_head(emg_embeds)  # (B, T', V)

        # Character embeddings (teacher forcing)
        char_embeds = self.char_embed(in_ids)  # (B, L, H)
        char_mask = (in_ids != pad_idx).long()

        # Prefix: [soft ; inst ; emg]
        soft = self.soft_prompt.expand(B, -1, -1)
        inst = self.inst_prompt.expand(B, -1, -1)
        prefix = torch.cat([soft, inst, emg_embeds], dim=1)
        inputs = torch.cat([prefix, char_embeds], dim=1)

        prefix_mask = torch.ones(
            B, soft.size(1) + inst.size(1), device=inputs.device, dtype=torch.long
        )
        attn_mask = torch.cat([prefix_mask, emg_mask_sub, char_mask], dim=1)

        llm_dtype = next(self.llm.parameters()).dtype
        inputs = inputs.to(dtype=llm_dtype)

        out = self.llm(inputs_embeds=inputs, attention_mask=attn_mask)
        hidden = out.last_hidden_state  # (B, total_len, H)

        # AR logits over character positions (aligned to in_ids)
        prefix_len = prefix.size(1)
        ar_hidden = hidden[:, prefix_len:, :].to(self.lm_head.weight.dtype)
        ar_logits = self.lm_head(ar_hidden)  # (B, L, V)

        return {
            "ar_logits": ar_logits,
            "ctc_logits": ctc_logits,
            "emg_lens_sub": emg_lens_sub,
        }

    @torch.no_grad()
    def prepare_context(self, emg_feats: torch.Tensor, emg_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare prefix context for decoding (batch=1)."""
        assert emg_feats.size(0) == 1

        emg_embeds = self.adapter(emg_feats)  # (1, T', H)
        Tprime = emg_embeds.size(1)

        emg_len = emg_mask.long().sum(dim=1)  # (1,)
        emg_len_sub = ceil_div(emg_len, self.subsample_factor).clamp(min=1)
        t = torch.arange(Tprime, device=emg_feats.device).unsqueeze(0)
        emg_mask_sub = (t < emg_len_sub.unsqueeze(1)).long()
        emg_embeds = emg_embeds * emg_mask_sub.unsqueeze(-1).to(emg_embeds.dtype)

        soft = self.soft_prompt
        inst = self.inst_prompt
        prefix_embeds = torch.cat([soft, inst, emg_embeds], dim=1)

        llm_dtype = next(self.llm.parameters()).dtype
        prefix_embeds = prefix_embeds.to(dtype=llm_dtype)
        prefix_mask = torch.cat(
            [
                torch.ones(1, soft.size(1) + inst.size(1), device=emg_feats.device, dtype=torch.long),
                emg_mask_sub,
            ],
            dim=1,
        )
        return prefix_embeds, prefix_mask

    @torch.no_grad()
    def next_token_logits(
        self,
        prefix_embeds: torch.Tensor,
        prefix_mask: torch.Tensor,
        prefix_ids: torch.Tensor,  # (1, L)
        pad_idx: int,
    ) -> torch.Tensor:
        """Next-token logits given current prefix token ids (batch=1)."""
        char_embeds = self.char_embed(prefix_ids)  # (1, L, H)
        char_mask = (prefix_ids != pad_idx).long()
        inputs = torch.cat([prefix_embeds, char_embeds], dim=1)
        attn_mask = torch.cat([prefix_mask, char_mask], dim=1)

        llm_dtype = next(self.llm.parameters()).dtype
        inputs = inputs.to(dtype=llm_dtype)

        out = self.llm(inputs_embeds=inputs, attention_mask=attn_mask)
        hidden = out.last_hidden_state
        last_h = hidden[:, -1, :].to(self.lm_head.weight.dtype)
        return self.lm_head(last_h)  # (1, V)


# -----------------------------------------------------------------------------
# AR beam decoding for validation (neural-only, used for checkpoint selection)
# -----------------------------------------------------------------------------
@torch.no_grad()
def ar_beam_decode_one(
    model: EMGToFrozenLLM,
    tt: TextTransform,
    emg_feats: torch.Tensor,  # (1, T, D)
    emg_mask: torch.Tensor,   # (1, T)
    beam_size: int = 6,
    max_len: int = 128,
    len_alpha: float = 0.6,
    len_delta: float = 5.0,
) -> List[int]:
    pad = tt.PAD_IDX
    bos = tt.BOS_IDX
    eos = tt.EOS_IDX

    prefix_embeds, prefix_mask = model.prepare_context(emg_feats, emg_mask)

    # hypothesis: (tokens, logp, ended)
    hyps: List[Tuple[List[int], float, bool]] = [([bos], 0.0, False)]

    def norm_score(tokens: List[int], logp: float) -> float:
        L = max(len(tokens) - 1, 1)  # exclude BOS
        return logp / ((L + float(len_delta)) ** float(len_alpha))

    for _ in range(int(max_len)):
        if all(h[2] for h in hyps):
            break

        candidates: List[Tuple[List[int], float, bool]] = []
        for tokens, logp, ended in hyps:
            if ended:
                candidates.append((tokens, logp, ended))
                continue

            prefix_ids = torch.tensor(tokens, device=emg_feats.device, dtype=torch.long).unsqueeze(0)
            logits = model.next_token_logits(prefix_embeds, prefix_mask, prefix_ids, pad_idx=pad)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)
            topk = torch.topk(log_probs, k=min(int(beam_size), log_probs.numel()))

            for lp, idx in zip(topk.values.tolist(), topk.indices.tolist()):
                idx = int(idx)
                new_tokens = tokens + [idx]
                new_logp = float(logp + float(lp))
                new_ended = (idx == eos)
                candidates.append((new_tokens, new_logp, new_ended))

        candidates.sort(key=lambda x: norm_score(x[0], x[1]), reverse=True)
        hyps = candidates[: int(beam_size)]

    finished = [h for h in hyps if h[2]]
    best = max(finished, key=lambda x: norm_score(x[0], x[1])) if finished else hyps[0]
    return best[0]


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--train_dir", type=str, default="data/train_emg")
    ap.add_argument("--val_dir", type=str, default="data/val_emg")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")

    default_base = "models/llama3.2-3B" if Path("models/llama3.2-3B").exists() else "meta-llama/Llama-3.2-3B"
    ap.add_argument("--base_model", type=str, default=default_base)

    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)

    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_ctc", type=float, default=0.2)

    ap.add_argument("--soft_prompt_len", type=int, default=8)
    ap.add_argument("--inst_prompt_len", type=int, default=8)
    ap.add_argument("--subsample_factor", type=int, default=4)

    # Validation decoding (paper defaults)
    ap.add_argument("--beam_size", type=int, default=6)
    ap.add_argument("--max_decode_len", type=int, default=128)
    ap.add_argument("--ar_len_alpha", type=float, default=0.6)
    ap.add_argument("--ar_len_delta", type=float, default=5.0)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--recompute_norm", action="store_true")
    ap.add_argument("--resume", type=str, default="", help="Path to latest_checkpoint.pt")

    return ap.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    seed_all(args.seed)

    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    art_dir = Path(args.artifacts_dir)
    art_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        raise FileNotFoundError(f"train_dir not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"val_dir not found: {val_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)

    # Normalizer (TRAIN only)
    norm_path = art_dir / "emg_norm.pkl"
    if norm_path.exists() and not args.recompute_norm:
        normalizer = load_normalizer(str(norm_path))
    else:
        normalizer = fit_normalizer_from_dir(train_dir, pattern="*.npy")
        save_normalizer(normalizer, str(norm_path))

    # Data loaders
    train_loader, train_ds, tt = make_loader(
        data_dir=train_dir,
        normalizer_path=str(norm_path),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader, _, _ = make_loader(
        data_dir=val_dir,
        normalizer_path=str(norm_path),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    
    # Infer feature dim safely (dataset returns: feats, text, utt_id)
    sample_feats, _sample_text, _sample_utt = train_ds[0]
    in_dim = int(sample_feats.shape[1])

    # Model
    mcfg = ModelConfig(
        base_model=args.base_model,
        vocab_size=tt.VOCAB_SIZE,
        in_dim=in_dim,
        subsample_factor=args.subsample_factor,
        soft_prompt_len=args.soft_prompt_len,
        inst_prompt_len=args.inst_prompt_len,
    )
    model = EMGToFrozenLLM(mcfg).to(device)

    # Losses
    ce_loss_fn = nn.CrossEntropyLoss(ignore_index=tt.PAD_IDX)
    ctc_loss_fn = nn.CTCLoss(blank=tt.PAD_IDX, zero_infinity=True)

    # Optimizer
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_val_wer = float("inf")
    best_path = art_dir / "best_checkpoint.pt"
    latest_path = art_dir / "latest_checkpoint.pt"

    # Resume
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_wer = float(ckpt.get("best_val_wer", best_val_wer))
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_val_wer={best_val_wer:.4f})")

    # Save run config
    (art_dir / "train_config.json").write_text(
        json.dumps(
            {
                **vars(args),
                "model_cfg": asdict(mcfg),
                "vocab": tt.VOCAB,
                "special": {"PAD_IDX": tt.PAD_IDX, "BOS_IDX": tt.BOS_IDX, "EOS_IDX": tt.EOS_IDX},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "ar": 0.0, "ctc": 0.0}
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for batch in pbar:
            emg_feats, emg_mask, in_ids, out_ids, char_mask, _texts = batch
            emg_feats = emg_feats.to(device)
            emg_mask = emg_mask.to(device)
            in_ids = in_ids.to(device)
            out_ids = out_ids.to(device)
            char_mask = char_mask.to(device)

            out_lens = char_mask.long().sum(dim=1)  # includes EOS (targets)
            ctc_targets, ctc_target_lens = build_ctc_targets(out_ids, out_lens, tt.EOS_IDX, tt.PAD_IDX)

            emg_lens = emg_mask.long().sum(dim=1)
            ctc_input_lens = ceil_div(emg_lens, model.subsample_factor).clamp(min=1)

            opt.zero_grad(set_to_none=True)
            with autocast(enabled=use_amp):
                out = model(emg_feats, emg_mask, in_ids, pad_idx=tt.PAD_IDX)
                ar_logits = out["ar_logits"]      # (B, L, V)
                ctc_logits = out["ctc_logits"]    # (B, T', V)

                ar_loss = ce_loss_fn(ar_logits.reshape(-1, ar_logits.size(-1)), out_ids.reshape(-1))

                log_probs = F.log_softmax(ctc_logits, dim=-1).transpose(0, 1)  # (T', B, V)
                ctc_loss = ctc_loss_fn(log_probs, ctc_targets, ctc_input_lens, ctc_target_lens)

                loss = ar_loss + float(args.lambda_ctc) * ctc_loss

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], float(args.grad_clip)
                )
            scaler.step(opt)
            scaler.update()

            running["loss"] += float(loss.item())
            running["ar"] += float(ar_loss.item())
            running["ctc"] += float(ctc_loss.item())
            n_batches += 1

            pbar.set_postfix(
                loss=running["loss"] / max(n_batches, 1),
                ar=running["ar"] / max(n_batches, 1),
                ctc=running["ctc"] / max(n_batches, 1),
            )

        # Validation (AR-beam only; checkpoint selection)
        if epoch % args.eval_every == 0:
            model.eval()
            refs: List[str] = []
            hyps: List[str] = []

            for batch in tqdm(val_loader, desc="Validating", leave=False):
                emg_feats, emg_mask, _in_ids, _out_ids, _char_mask, texts = batch
                emg_feats = emg_feats.to(device)
                emg_mask = emg_mask.to(device)

                pred_ids = ar_beam_decode_one(
                    model=model,
                    tt=tt,
                    emg_feats=emg_feats,
                    emg_mask=emg_mask,
                    beam_size=args.beam_size,
                    max_len=args.max_decode_len,
                    len_alpha=args.ar_len_alpha,
                    len_delta=args.ar_len_delta,
                )
                hyp = tt.ids_to_text(pred_ids)
                ref = texts[0]  # already cleaned in dataset
                refs.append(ref)
                hyps.append(hyp)

            val_wer = compute_wer(refs, hyps, tt)
            val_cer = compute_cer(refs, hyps, tt)
            print(
                f"Epoch {epoch}: val WER={val_wer:.4f} CER={val_cer:.4f} "
                f"(AR beam K={args.beam_size}, Lmax={args.max_decode_len}, α={args.ar_len_alpha}, δ={args.ar_len_delta})"
            )

            # Save latest
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch": epoch,
                    "best_val_wer": best_val_wer,
                    "model_cfg": asdict(mcfg),
                    "vocab": tt.VOCAB,
                },
                latest_path,
            )

            # Save best
            if val_wer < best_val_wer:
                best_val_wer = float(val_wer)
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": opt.state_dict(),
                        "epoch": epoch,
                        "best_val_wer": best_val_wer,
                        "model_cfg": asdict(mcfg),
                        "vocab": tt.VOCAB,
                    },
                    best_path,
                )
                print(f"  -> new best saved to {best_path} (best_val_wer={best_val_wer:.4f})")

    print(f"Done. Best val WER = {best_val_wer:.4f}")


if __name__ == "__main__":
    main()