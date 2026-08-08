#!/usr/bin/env python3
"""
train_emg_llm.py

Train the EMG-to-frozen-LLM interface (character-level AR head + auxiliary
CTC head), following Sec. IV-D and Table I of the paper.

  * The decoder-only LLM backbone stays frozen (theta frozen).
  * Trainable: EMG adapter, soft prompt, instruction prompt, character
    embedding table, LM head, CTC head.
  * Dual supervision:  L = L_AR + lambda_ctc * L_CTC   (Eq. 10-11)
  * Feature normalisation is fitted on TRAIN only and reused for val/test.
  * Checkpoint selection uses neural-only AR beam validation decoding
    (K=6, L_max=128, alpha=0.6, delta=5.0), deliberately without any NS
    constraint so selection stays independent of inference-time symbolic
    structure.

Table I defaults: 400 epochs, batch size 8, AdamW lr 1e-4, weight decay 0.01,
grad clip 1.0, lambda_ctc 0.2, soft prompt 8, instruction prompt 8, r = 4.

Supervision variants for the Table III ablation:
  --supervision ar_ctc   (default, dual supervision)
  --supervision ar_only  (lambda_ctc = 0)
  --supervision ctc_only (AR objective disabled)

Expected data layout (from split_data.py):
  data/train_emg/*.npy + *.json
  data/val_emg/*.npy   + *.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoConfig, AutoModel

try:
    from models.emg_adapter import EMGAdapterV2
    from scripts.dataset_emg import make_loader
    from scripts.data_utils import FeatureNormalizer, NormState, TextTransform, load_normalizer, save_normalizer
    from scripts.ar_decode import PrefixCachedAR, ctc_greedy_decode_ids
    from scripts.augment import AugmentConfig
    from scripts.metrics import score_corpus
except ImportError:  # flat layout fallback
    from emg_adapter import EMGAdapterV2
    from dataset_emg import make_loader
    from data_utils import FeatureNormalizer, NormState, TextTransform, load_normalizer, save_normalizer
    from ar_decode import PrefixCachedAR, ctc_greedy_decode_ids
    from augment import AugmentConfig
    from metrics import score_corpus


# -----------------------------------------------------------------------------
# Reproducibility
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


def pick_dtype() -> torch.dtype:
    """BF16 where supported (Sec. VI-D2 reports BF16 inference), else FP16/FP32."""
    if not torch.cuda.is_available():
        return torch.float32
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


# -----------------------------------------------------------------------------
# Feature normalisation, fitted on TRAIN only (Sec. III-C)
# -----------------------------------------------------------------------------
def fit_normalizer_from_dir(
    features_dir: Path, pattern: str = "*.npy", eps: float = 1e-8
) -> FeatureNormalizer:
    files = sorted(features_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No feature files found in {features_dir} with pattern {pattern}")

    sum_x = sum_x2 = None
    count = 0
    for fp in tqdm(files, desc=f"Fitting normalizer from {features_dir.name}", leave=False):
        x = np.load(fp).astype(np.float64)
        if x.ndim != 2:
            raise ValueError(f"Expected 2D feature array, got {x.shape} in {fp}")
        if sum_x is None:
            sum_x = np.zeros((x.shape[1],), dtype=np.float64)
            sum_x2 = np.zeros((x.shape[1],), dtype=np.float64)
        sum_x += x.sum(axis=0)
        sum_x2 += (x * x).sum(axis=0)
        count += x.shape[0]

    mean = sum_x / max(count, 1)
    var = np.maximum(sum_x2 / max(count, 1) - mean * mean, eps)
    norm = FeatureNormalizer(eps=eps)
    norm.state = NormState(mean=mean.astype(np.float32), std=np.sqrt(var).astype(np.float32))
    return norm


# -----------------------------------------------------------------------------
# CTC targets: drop EOS, report true lengths (padding is ignored via lengths)
# -----------------------------------------------------------------------------
def build_ctc_targets(
    out_ids: torch.Tensor, out_lens: torch.Tensor, eos_idx: int, pad_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    bsz = out_ids.size(0)
    targets: List[torch.Tensor] = []
    lens: List[int] = []

    for i in range(bsz):
        L = int(out_lens[i].item())
        seq = out_ids[i, :L]
        keep = seq != eos_idx
        seq = seq[keep]
        targets.append(seq)
        lens.append(int(seq.numel()))

    max_len = max(lens) if lens else 0
    padded = out_ids.new_full((bsz, max_len), fill_value=pad_idx)
    for i, seq in enumerate(targets):
        if seq.numel() > 0:
            padded[i, : seq.numel()] = seq
    return padded, torch.tensor(lens, device=out_ids.device, dtype=torch.long)


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
    # adapter shape / regularisation (defaults reproduce Sec. IV-B2)
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    adapter_norm: str = "layer"


class EMGToFrozenLLM(nn.Module):
    """Frozen decoder-only LLM conditioned on an EMG prefix; character-level AR."""

    #: parameter-name prefixes that are trained (everything else is frozen)
    TRAINABLE_PREFIXES = ("adapter.", "char_embed.", "lm_head.", "ctc_head.", "soft_prompt", "inst_prompt")

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        hf_cfg = AutoConfig.from_pretrained(cfg.base_model)
        hidden_size = getattr(hf_cfg, "hidden_size", None) or getattr(hf_cfg, "n_embd", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden size from the HF config")
        self.hidden_size = int(hidden_size)

        dtype = pick_dtype()
        try:
            self.llm = AutoModel.from_pretrained(cfg.base_model, dtype=dtype)
        except TypeError:  # transformers < 4.56
            self.llm = AutoModel.from_pretrained(cfg.base_model, torch_dtype=dtype)

        for p in self.llm.parameters():
            p.requires_grad_(False)

        # Everything fed to the frozen LLM as `inputs_embeds` must sit on the scale
        # of that LLM's own token embeddings, otherwise its pretrained prior cannot
        # engage. Measure the backbone's scale once and reuse it for the adapter
        # output, the character table and the prompts.
        emb_w = self.llm.get_input_embeddings().weight
        self.embed_rms = float(emb_w.float().pow(2).mean(dim=-1).sqrt().mean())

        self.adapter = EMGAdapterV2(
            in_dim=cfg.in_dim,
            hidden_size=self.hidden_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            norm=cfg.adapter_norm,
            out_rms=self.embed_rms,
        )
        self.char_embed = nn.Embedding(cfg.vocab_size, self.hidden_size)
        self.lm_head = nn.Linear(self.hidden_size, cfg.vocab_size, bias=False)
        self.ctc_head = nn.Linear(self.hidden_size, cfg.vocab_size)

        self.soft_prompt = nn.Parameter(torch.randn(1, cfg.soft_prompt_len, self.hidden_size) * self.embed_rms)
        self.inst_prompt = nn.Parameter(torch.randn(1, cfg.inst_prompt_len, self.hidden_size) * self.embed_rms)

        self._init_char_embed()

    def _init_char_embed(self) -> None:
        """
        Seed the character table from the backbone's own token embeddings.

        `nn.Embedding` defaults to N(0, 1), which for Llama-3.2-3B is 47.8x the
        per-token RMS of `model.embed_tokens` (0.989 vs 0.0207) -- and training
        does not fix it: after 400 epochs the table had moved only to 0.989. Every
        character therefore reached the frozen LLM far outside the distribution it
        was pretrained on. Where a character is a single token in the backbone's
        vocabulary we copy that token's embedding; otherwise we fall back to noise
        at the correct scale.
        """
        from transformers import AutoTokenizer

        emb = self.llm.get_input_embeddings().weight.detach()
        with torch.no_grad():
            self.char_embed.weight.normal_(0.0, self.embed_rms)
            try:
                tok = AutoTokenizer.from_pretrained(self.cfg.base_model)
            except Exception:
                return
            n_seeded = 0
            for idx in range(len(TextTransform.BASE_CHARS)):
                ch = TextTransform.BASE_CHARS[idx]
                ids = tok.encode(ch, add_special_tokens=False)
                if len(ids) == 1 and 0 <= ids[0] < emb.size(0):
                    row = emb[ids[0]].to(self.char_embed.weight.dtype)
                    self.char_embed.weight[idx] = row
                    # Llama-3.2 ties its output head to embed_tokens, so the same
                    # row is also the readout direction the frozen stack already
                    # produces for that character. Seeding lm_head with it lets the
                    # backbone's computation transfer instead of being decoded by a
                    # randomly-initialised head.
                    self.lm_head.weight[idx] = row
                    n_seeded += 1
            self._n_char_seeded = n_seeded

    @property
    def subsample_factor(self) -> int:
        return int(self.cfg.subsample_factor)

    # -- checkpointing -------------------------------------------------------
    def trainable_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Only the trained modules. The frozen backbone is reloaded from
        `cfg.base_model`, so checkpoints stay ~100 MB instead of ~6.6 GB.
        """
        sd = self.state_dict()
        return {k: v for k, v in sd.items() if k.startswith(self.TRAINABLE_PREFIXES)}

    def load_trainable_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Load a slim checkpoint, or a legacy full one (frozen LLM weights ignored)."""
        filtered = {k: v for k, v in state.items() if k.startswith(self.TRAINABLE_PREFIXES)}
        missing, unexpected = self.load_state_dict(filtered, strict=False)
        still_missing = [k for k in missing if k.startswith(self.TRAINABLE_PREFIXES)]
        if still_missing:
            raise RuntimeError(f"Checkpoint is missing trainable tensors: {still_missing[:8]}")
        if unexpected:
            raise RuntimeError(f"Checkpoint has unexpected tensors: {unexpected[:8]}")

    # -- forward -------------------------------------------------------------
    def _emg_prefix(
        self, emg_feats: torch.Tensor, emg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Adapter output, its subsampled mask, and the subsampled lengths."""
        # Pass true lengths so padded frames are excluded from the convolutions,
        # the normalisation statistics and the encoder's self-attention.
        emg_embeds = self.adapter(emg_feats, lengths=emg_mask.long().sum(dim=1))  # (B, T', H)
        Tprime = emg_embeds.size(1)

        emg_lens = emg_mask.long().sum(dim=1)
        emg_lens_sub = ceil_div(emg_lens, self.subsample_factor).clamp(min=1, max=Tprime)
        t = torch.arange(Tprime, device=emg_feats.device).unsqueeze(0)
        emg_mask_sub = (t < emg_lens_sub.unsqueeze(1)).long()
        emg_embeds = emg_embeds * emg_mask_sub.unsqueeze(-1).to(emg_embeds.dtype)
        return emg_embeds, emg_mask_sub, emg_lens_sub

    def forward(
        self,
        emg_feats: torch.Tensor,  # (B, T, D)
        emg_mask: torch.Tensor,   # (B, T) bool
        in_ids: torch.Tensor,     # (B, L)
        pad_idx: int,
    ) -> Dict[str, torch.Tensor]:
        B = emg_feats.size(0)
        emg_embeds, emg_mask_sub, emg_lens_sub = self._emg_prefix(emg_feats, emg_mask)

        ctc_logits = self.ctc_head(emg_embeds)  # (B, T', V)

        char_embeds = self.char_embed(in_ids)
        char_mask = (in_ids != pad_idx).long()

        # U = [U_soft ; U_inst ; H_emg ; U_char(BOS, y_1..y_{L-1})]   (Eq. 7)
        soft = self.soft_prompt.expand(B, -1, -1)
        inst = self.inst_prompt.expand(B, -1, -1)
        prefix = torch.cat([soft, inst, emg_embeds], dim=1)
        inputs = torch.cat([prefix, char_embeds], dim=1)

        prefix_mask = torch.ones(B, soft.size(1) + inst.size(1), device=inputs.device, dtype=torch.long)
        attn_mask = torch.cat([prefix_mask, emg_mask_sub, char_mask], dim=1)

        llm_dtype = next(self.llm.parameters()).dtype
        out = self.llm(inputs_embeds=inputs.to(dtype=llm_dtype), attention_mask=attn_mask, use_cache=False)
        hidden = out.last_hidden_state

        ar_hidden = hidden[:, prefix.size(1) :, :].to(self.lm_head.weight.dtype)
        return {
            "ar_logits": self.lm_head(ar_hidden),
            "ctc_logits": ctc_logits,
            "emg_lens_sub": emg_lens_sub,
        }

    # -- inference interface (Algorithm 1, else-branch) ----------------------
    @torch.no_grad()
    def prepare_context(
        self, emg_feats: torch.Tensor, emg_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Fixed EMG prefix embeddings + mask for decoding (batch = 1)."""
        assert emg_feats.size(0) == 1, "prepare_context decodes one utterance at a time"
        emg_embeds, emg_mask_sub, _ = self._emg_prefix(emg_feats, emg_mask)

        prefix_embeds = torch.cat([self.soft_prompt, self.inst_prompt, emg_embeds], dim=1)
        llm_dtype = next(self.llm.parameters()).dtype
        prefix_mask = torch.cat(
            [
                torch.ones(
                    1,
                    self.soft_prompt.size(1) + self.inst_prompt.size(1),
                    device=emg_feats.device,
                    dtype=torch.long,
                ),
                emg_mask_sub,
            ],
            dim=1,
        )
        return prefix_embeds.to(dtype=llm_dtype), prefix_mask

    @torch.no_grad()
    def ctc_posteriors(self, emg_feats: torch.Tensor, emg_mask: torch.Tensor) -> torch.Tensor:
        """Cached CTC logits {p_t} for one utterance: (T', V)."""
        emg_embeds, _, _ = self._emg_prefix(emg_feats, emg_mask)
        return self.ctc_head(emg_embeds)[0]

    @torch.no_grad()
    def next_token_logits(
        self,
        prefix_embeds: torch.Tensor,
        prefix_mask: torch.Tensor,
        prefix_ids: torch.Tensor,
        pad_idx: int,
    ) -> torch.Tensor:
        """Uncached next-token logits (kept for compatibility; prefer PrefixCachedAR)."""
        char_embeds = self.char_embed(prefix_ids)
        char_mask = (prefix_ids != pad_idx).long()
        inputs = torch.cat([prefix_embeds.expand(prefix_ids.size(0), -1, -1), char_embeds], dim=1)
        attn_mask = torch.cat([prefix_mask.expand(prefix_ids.size(0), -1), char_mask], dim=1)

        llm_dtype = next(self.llm.parameters()).dtype
        out = self.llm(inputs_embeds=inputs.to(dtype=llm_dtype), attention_mask=attn_mask, use_cache=False)
        lengths = char_mask.sum(dim=1).clamp(min=1) - 1
        hidden = out.last_hidden_state[:, prefix_embeds.size(1) :, :]
        last_h = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths]
        return self.lm_head(last_h.to(self.lm_head.weight.dtype))


# -----------------------------------------------------------------------------
# Validation decoding (neural-only, used for checkpoint selection)
# -----------------------------------------------------------------------------
@torch.no_grad()
# -----------------------------------------------------------------------------
# Distributed helpers
# -----------------------------------------------------------------------------
def _dist_info() -> Tuple[int, int]:
    """(rank, world_size); (0, 1) when not running under torchrun."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()
    return 0, 1


def _is_main() -> bool:
    return _dist_info()[0] == 0


def _gather_triples(local: List[Tuple[int, str, str]]) -> List[Tuple[int, str, str]]:
    """All-gather per-rank (index, ref, hyp) lists into one list on every rank."""
    rank, world = _dist_info()
    if world == 1:
        return list(local)
    buckets: List[Optional[List[Tuple[int, str, str]]]] = [None] * world
    dist.all_gather_object(buckets, local)
    out: List[Tuple[int, str, str]] = []
    for b in buckets:
        if b:
            out.extend(b)
    return out


def setup_distributed() -> Tuple[int, int, int]:
    """
    Initialise the process group from torchrun's environment.

    Returns (rank, world_size, local_rank). Falls back to single-process when
    the script is launched with plain `python`.
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return dist.get_rank(), dist.get_world_size(), local_rank


def validate(
    model: EMGToFrozenLLM,
    tt,
    val_loader,
    device: torch.device,
    beam_size: int,
    max_len: int,
    alpha: float,
    delta: float,
    mode: str = "ar_beam",
) -> Tuple[float, float]:
    """
    Corpus WER/CER on the validation split under neural-only decoding.

    Validation dominates wall-clock here (beam search is batch-of-one and
    sequential: ~2.4 s/utterance against ~3.5 s for a whole training epoch), so
    under DDP each rank decodes the stride-`world_size` slice of the split and
    the (index, ref, hyp) triples are gathered before scoring. Striding by index
    avoids the duplicate padding a DistributedSampler would introduce, which
    would otherwise double-count utterances in a corpus-level metric.
    """
    model.eval()
    rank, world = _dist_info()
    local: List[Tuple[int, str, str]] = []

    for i, batch in enumerate(tqdm(val_loader, desc="Validating", leave=False, disable=rank != 0)):
        if i % world != rank:
            continue
        emg_feats, emg_mask, _in, _out, _cm, texts = batch
        emg_feats = emg_feats.to(device)
        emg_mask = emg_mask.to(device)

        if mode == "ctc_greedy":
            ctc_logits = model.ctc_posteriors(emg_feats, emg_mask)
            ids = ctc_greedy_decode_ids(ctc_logits, blank_id=tt.PAD_IDX)
        else:
            prefix_embeds, prefix_mask = model.prepare_context(emg_feats, emg_mask)
            dec = PrefixCachedAR(model, tt, prefix_embeds, prefix_mask, capacity=max(1, beam_size))
            ids = dec.beam(beam_size=beam_size, max_len=max_len, alpha=alpha, delta=delta)
        local.append((i, texts[0], tt.ids_to_text(ids)))

    merged = _gather_triples(local)
    merged.sort(key=lambda r: r[0])
    refs = [r[1] for r in merged]
    hyps = [r[2] for r in merged]

    s = score_corpus(refs, hyps, tt)
    return s.wer, s.cer


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train the EMG->frozen-LLM interface (Table I settings).")

    ap.add_argument("--train_dir", type=str, default="data/train_emg")
    ap.add_argument("--val_dir", type=str, default="data/val_emg")
    ap.add_argument("--artifacts_dir", type=str, default="artifacts")

    default_base = "models/llama3.2-3B" if Path("models/llama3.2-3B").exists() else "meta-llama/Llama-3.2-3B"
    ap.add_argument("--base_model", type=str, default=default_base)

    # --- Table I: training and EMG->LLM interface (fixed) ---
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lambda_ctc", type=float, default=0.2)
    ap.add_argument("--soft_prompt_len", type=int, default=8)
    ap.add_argument("--inst_prompt_len", type=int, default=8)
    ap.add_argument("--subsample_factor", type=int, default=4)

    ap.add_argument(
        "--supervision",
        choices=["ar_ctc", "ar_only", "ctc_only"],
        default="ar_ctc",
        help="Training-supervision variant for the Table III ablation.",
    )

    # --- regularisation: the 400-utterance training set overfits hard without it ---
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                    help="EMG feature augmentation (gain jitter, noise, time/channel masking, warp).")
    ap.add_argument("--aug_strength", type=float, default=1.0,
                    help="Uniform multiplier on every augmentation strength.")
    ap.add_argument("--dropout", type=float, default=0.1, help="Adapter dropout.")
    ap.add_argument("--label_smoothing", type=float, default=0.0, help="AR cross-entropy label smoothing.")
    ap.add_argument("--adapter_norm", choices=["layer", "batch"], default="layer",
                    help="'layer' is padding-safe; 'batch' reproduces the original BatchNorm.")
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=4)

    # --- schedule ---
    ap.add_argument("--warmup_frac", type=float, default=0.05,
                    help="Fraction of total steps spent linearly warming up the LR.")
    ap.add_argument("--lr_schedule", choices=["cosine", "constant"], default="cosine")
    ap.add_argument("--min_lr_frac", type=float, default=0.05,
                    help="Cosine floor as a fraction of the peak LR.")
    ap.add_argument("--early_stop", type=int, default=0,
                    help="Stop after this many evaluations without a new best (0 = never).")

    # --- Table I: checkpoint selection on validation (neural-only AR beam) ---
    ap.add_argument("--beam_size", type=int, default=6)
    ap.add_argument("--max_decode_len", type=int, default=128)
    ap.add_argument("--ar_len_alpha", type=float, default=0.6)
    ap.add_argument("--ar_len_delta", type=float, default=5.0)

    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--eval_every", type=int, default=1)
    ap.add_argument("--recompute_norm", action="store_true")
    ap.add_argument("--resume", type=str, default="")
    ap.add_argument("--tag", type=str, default="", help="Suffix for checkpoint filenames.")

    return ap.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    rank, world, local_rank = setup_distributed()
    # Ranks must not draw identical augmentation streams, but the split itself
    # and the model init have to be identical everywhere.
    seed_all(args.seed)

    train_dir, val_dir = Path(args.train_dir), Path(args.val_dir)
    art_dir = Path(args.artifacts_dir)
    if _is_main():
        art_dir.mkdir(parents=True, exist_ok=True)
    for d in (train_dir, val_dir):
        if not d.exists():
            raise FileNotFoundError(f"data dir not found: {d}")

    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    amp_dtype = pick_dtype()
    use_amp = torch.cuda.is_available()
    # GradScaler is only needed for FP16; BF16 has FP32 dynamic range.
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)

    # Normalizer: TRAIN only (Sec. III-C), reused for val/test.
    norm_path = art_dir / "emg_norm.pkl"
    if _is_main():
        if norm_path.exists() and not args.recompute_norm:
            _ = load_normalizer(str(norm_path))
        else:
            save_normalizer(fit_normalizer_from_dir(train_dir, pattern="*.npy"), str(norm_path))
    if world > 1:
        dist.barrier()  # every rank must read the same normalizer file

    # Effective batch size is held at args.batch_size regardless of world size,
    # so the recipe selected by the single-GPU sweep is preserved exactly: the
    # number of optimizer steps per epoch is unchanged.
    per_rank_bs = max(1, args.batch_size // world)
    if _is_main() and per_rank_bs * world != args.batch_size:
        print(f"WARNING: batch_size {args.batch_size} not divisible by world size {world}; "
              f"effective batch is {per_rank_bs * world}")

    aug = AugmentConfig(enabled=bool(args.augment)).scaled(float(args.aug_strength))
    train_loader, train_ds, tt = make_loader(
        data_dir=train_dir,
        normalizer_path=str(norm_path),
        batch_size=per_rank_bs,
        shuffle=True,
        num_workers=args.num_workers,
        augment=aug,
        seed=args.seed + rank,
        distributed=world > 1,
    )
    # Validation is never augmented.
    val_loader, _, _ = make_loader(
        data_dir=val_dir, normalizer_path=str(norm_path), batch_size=1, shuffle=False, num_workers=0
    )

    sample_feats, _, _ = train_ds[0]
    in_dim = int(sample_feats.shape[1])

    mcfg = ModelConfig(
        base_model=args.base_model,
        vocab_size=tt.VOCAB_SIZE,
        in_dim=in_dim,
        subsample_factor=args.subsample_factor,
        soft_prompt_len=args.soft_prompt_len,
        inst_prompt_len=args.inst_prompt_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        dropout=args.dropout,
        adapter_norm=args.adapter_norm,
    )
    core = EMGToFrozenLLM(mcfg).to(device)

    n_train = sum(p.numel() for p in core.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in core.parameters() if not p.requires_grad)
    if _is_main():
        print(f"Trainable parameters: {n_train/1e6:.2f}M | frozen backbone: {n_frozen/1e6:.1f}M")
        print(f"Supervision: {args.supervision} | LLM dtype: {amp_dtype}")
        print(f"World size: {world} | per-rank batch: {per_rank_bs} | effective batch: {per_rank_bs*world}")

    # `model` is what we run forward/backward through; `core` keeps the plain
    # module for decoding (PrefixCachedAR, prepare_context) and checkpointing,
    # neither of which goes through the DDP wrapper.
    if world > 1:
        model = nn.parallel.DistributedDataParallel(
            core,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,  # buffers here are deterministic (PosEnc)
        )
    else:
        model = core

    # Objective weights (Eq. 10)
    w_ar = 0.0 if args.supervision == "ctc_only" else 1.0
    w_ctc = 0.0 if args.supervision == "ar_only" else float(args.lambda_ctc)
    # CTC-only models cannot be selected on AR decoding, so use CTC greedy instead.
    val_mode = "ctc_greedy" if args.supervision == "ctc_only" else "ar_beam"

    ce_loss_fn = nn.CrossEntropyLoss(
        ignore_index=tt.PAD_IDX, label_smoothing=float(args.label_smoothing)
    )
    ctc_loss_fn = nn.CTCLoss(blank=tt.PAD_IDX, zero_infinity=True)

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Linear warmup then cosine decay over the whole run.
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * max(1, args.epochs)
    warmup_steps = max(1, int(total_steps * float(args.warmup_frac)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if args.lr_schedule == "constant":
            return 1.0
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        prog = min(1.0, max(0.0, prog))
        floor = float(args.min_lr_frac)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    suffix = f"_{args.tag}" if args.tag else ""
    best_path = art_dir / f"best_checkpoint{suffix}.pt"
    latest_path = art_dir / f"latest_checkpoint{suffix}.pt"

    start_epoch, best_val_wer = 1, float("inf")
    best_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        core.load_trainable_state(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_wer = float(ckpt.get("best_val_wer", best_val_wer))
        print(f"Resumed from {args.resume} at epoch {start_epoch} (best_val_wer={best_val_wer:.4f})")

    if _is_main():
        (art_dir / f"train_config{suffix}.json").write_text(
            json.dumps(
                {
                    **vars(args),
                    "model_cfg": asdict(mcfg),
                    "vocab": tt.VOCAB,
                    "special": {"PAD_IDX": tt.PAD_IDX, "BOS_IDX": tt.BOS_IDX, "EOS_IDX": tt.EOS_IDX},
                    "loss_weights": {"ar": w_ar, "ctc": w_ctc},
                    "world_size": world,
                    "per_rank_batch_size": per_rank_bs,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def save(path: Path, epoch: int) -> None:
        if not _is_main():
            return
        torch.save(
            {
                "model": core.trainable_state_dict(),
                "optimizer": opt.state_dict(),
                "epoch": epoch,
                "best_val_wer": best_val_wer,
                "model_cfg": asdict(mcfg),
                "vocab": tt.VOCAB,
                "supervision": args.supervision,
                "slim": True,
            },
            path,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "ar": 0.0, "ctc": 0.0}
        n_batches = 0

        # Reshuffles the per-rank shards; without it every epoch sees the same
        # split of the data across ranks.
        if world > 1 and hasattr(train_loader, "sampler") and train_loader.sampler is not None:
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", disable=not _is_main())
        for batch in pbar:
            emg_feats, emg_mask, in_ids, out_ids, char_mask, _texts = batch
            emg_feats, emg_mask = emg_feats.to(device), emg_mask.to(device)
            in_ids, out_ids = in_ids.to(device), out_ids.to(device)
            char_mask = char_mask.to(device)

            out_lens = char_mask.long().sum(dim=1)
            ctc_targets, ctc_target_lens = build_ctc_targets(out_ids, out_lens, tt.EOS_IDX, tt.PAD_IDX)
            emg_lens = emg_mask.long().sum(dim=1)

            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                out = model(emg_feats, emg_mask, in_ids, pad_idx=tt.PAD_IDX)
                ar_logits, ctc_logits = out["ar_logits"], out["ctc_logits"]

                ar_loss = ce_loss_fn(ar_logits.reshape(-1, ar_logits.size(-1)), out_ids.reshape(-1))

                ctc_input_lens = out["emg_lens_sub"].clamp(max=ctc_logits.size(1))
                log_probs = F.log_softmax(ctc_logits.float(), dim=-1).transpose(0, 1)
                ctc_loss = ctc_loss_fn(log_probs, ctc_targets, ctc_input_lens, ctc_target_lens)

                loss = w_ar * ar_loss + w_ctc * ctc_loss

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], float(args.grad_clip)
                )
            scaler.step(opt)
            scaler.update()
            sched.step()

            running["loss"] += float(loss.item())
            running["ar"] += float(ar_loss.item())
            running["ctc"] += float(ctc_loss.item())
            n_batches += 1
            pbar.set_postfix(
                loss=running["loss"] / max(n_batches, 1),
                ar=running["ar"] / max(n_batches, 1),
                ctc=running["ctc"] / max(n_batches, 1),
            )

        if epoch % args.eval_every == 0:
            # Decoding runs on the plain module: PrefixCachedAR reaches into
            # model internals that the DDP wrapper does not expose.
            val_wer, val_cer = validate(
                core, tt, val_loader, device,
                beam_size=args.beam_size,
                max_len=args.max_decode_len,
                alpha=args.ar_len_alpha,
                delta=args.ar_len_delta,
                mode=val_mode,
            )
            if _is_main():
                print(
                    f"Epoch {epoch}: val WER={val_wer:.4f} CER={val_cer:.4f} "
                    f"({val_mode}, K={args.beam_size}, Lmax={args.max_decode_len}, "
                    f"a={args.ar_len_alpha}, d={args.ar_len_delta})"
                )
            save(latest_path, epoch)
            if val_wer < best_val_wer:
                best_val_wer = float(val_wer)
                best_epoch = epoch
                save(best_path, epoch)
                if _is_main():
                    print(f"  -> new best: {best_path} (val WER={best_val_wer:.4f})")

            if args.early_stop and epoch - best_epoch >= int(args.early_stop):
                if _is_main():
                    print(f"Early stop at epoch {epoch}: no improvement for "
                          f"{epoch - best_epoch} epochs (best {best_val_wer:.4f} @ {best_epoch}).")
                break

    if _is_main():
        print(f"Done. Best val WER = {best_val_wer:.4f}")
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
