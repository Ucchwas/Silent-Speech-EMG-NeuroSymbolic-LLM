import math
from typing import Optional

import torch
import torch.nn as nn


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """(B,) valid lengths -> (B, T) bool mask, True where valid."""
    t = torch.arange(max_len, device=lengths.device).unsqueeze(0)
    return t < lengths.unsqueeze(1)


def _apply_mask_bct(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    """Zero padded frames of a (B, C, T) tensor given a (B, T) bool mask."""
    if mask is None:
        return x
    return x * mask.unsqueeze(1).to(x.dtype)


class MaskedNorm(nn.Module):
    """
    Normalisation for variable-length (B, C, T) sequences.

    mode="layer" (default): LayerNorm over the channel axis. Independent of
    batch composition and of how much padding a batch happens to contain, so
    training statistics and inference statistics cannot diverge.

    mode="batch": the BatchNorm1d of the original implementation. Kept for
    comparison, but note it computes its statistics over (B, T) *including*
    padded frames -- with utterances of 175-398 frames a batch of 8 is ~20%
    padding, so the running statistics it accumulates are biased toward zero
    and do not describe the batch-of-one seen at inference.
    """

    def __init__(self, ch: int, mode: str = "layer"):
        super().__init__()
        self.mode = mode
        if mode == "batch":
            self.norm = nn.BatchNorm1d(ch)
        elif mode == "layer":
            self.norm = nn.LayerNorm(ch)
        else:
            raise ValueError(f"unknown norm mode: {mode}")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.mode == "batch":
            return _apply_mask_bct(self.norm(x), mask)
        # LayerNorm expects channels last
        y = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return _apply_mask_bct(y, mask)


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable conv:
      depthwise Conv1d(groups=in_ch) + pointwise Conv1d(1x1) + Norm + GELU + Dropout
    Input/Output shape: (B, C, T)
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int = 5,
        s: int = 1,
        p: Optional[int] = None,
        dropout: float = 0.1,
        norm: str = "layer",
    ):
        super().__init__()
        p = (k // 2) if p is None else int(p)
        self.stride = int(s)
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.norm = MaskedNorm(out_ch, norm)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Zero the padding before convolving so it cannot leak into valid frames.
        x = _apply_mask_bct(x, mask)
        x = self.dw(x)
        x = self.pw(x)
        if mask is not None and self.stride > 1:
            mask = mask[:, :: self.stride][:, : x.size(-1)]
        x = self.norm(x, mask)
        x = self.act(x)
        return _apply_mask_bct(self.drop(x), mask)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation over time for (B, C, T), padding-aware."""

    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        hidden = max(1, ch // max(1, int(r)))
        self.fc = nn.Sequential(
            nn.Conv1d(ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is None:
            pooled = x.mean(dim=-1, keepdim=True)
        else:
            m = mask.unsqueeze(1).to(x.dtype)
            pooled = (x * m).sum(dim=-1, keepdim=True) / m.sum(dim=-1, keepdim=True).clamp_min(1.0)
        return _apply_mask_bct(x * self.fc(pooled), mask)


class TCNBlock(nn.Module):
    """
    Length-preserving residual TCN block:
      depthwise-separable conv -> depthwise dilated conv -> pointwise -> Norm -> Dropout
      with a residual connection.
    """

    def __init__(self, ch: int, k: int = 5, dilation: int = 1, dropout: float = 0.1, norm: str = "layer"):
        super().__init__()
        k = int(k)
        dilation = int(dilation)
        self.conv = DepthwiseSeparableConv1d(ch, ch, k=k, s=1, p=k // 2, dropout=dropout, norm=norm)
        self.dil = nn.Conv1d(
            ch, ch, kernel_size=k, stride=1, padding=dilation * (k // 2),
            dilation=dilation, groups=ch, bias=False,
        )
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, bias=False)
        self.norm = MaskedNorm(ch, norm)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        res = x
        x = self.conv(x, mask)
        x = _apply_mask_bct(x, mask)
        x = self.dil(x)
        x = self.pw(x)
        x = self.norm(x, mask)
        x = self.drop(x)
        return _apply_mask_bct(self.act(x + res), mask)


class PosEnc(nn.Module):
    """Sinusoidal positional encoding for (B, T, D)."""

    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        pe = torch.zeros(int(max_len), int(d_model))
        pos = torch.arange(0, int(max_len)).unsqueeze(1)
        div = torch.exp(torch.arange(0, int(d_model), 2) * (-math.log(10000.0) / int(d_model)))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EMGAdapterV2(nn.Module):
    """
    EMG adapter (Sec. IV-B2):
      (B, T, in_dim) -> (B, ceil(T/4), hidden_size)

    Subsampling is fixed to factor 4 (two stride-2 depthwise-separable convs),
    which must match the CTC input-length computation in train/inference.

    Padding handling: every stage receives the frame mask, so padded frames are
    zeroed after each convolution and excluded from self-attention via
    `src_key_padding_mask`. Without this the encoder attends over padding during
    batched training but never at inference (batch of one), which is a silent
    train/inference mismatch.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        norm: str = "layer",
        out_rms: float = 0.02,
    ):
        super().__init__()
        self.subsample_factor = 4

        self.in_proj = nn.Linear(int(in_dim), int(d_model))
        self.sub1 = DepthwiseSeparableConv1d(d_model, d_model, k=5, s=2, dropout=dropout, norm=norm)
        self.sub2 = DepthwiseSeparableConv1d(d_model, d_model, k=5, s=2, dropout=dropout, norm=norm)

        self.tcn1 = TCNBlock(d_model, k=5, dilation=1, dropout=dropout, norm=norm)
        self.se = SEBlock(d_model)
        self.tcn2 = TCNBlock(d_model, k=5, dilation=2, dropout=dropout, norm=norm)
        self.tcn3 = TCNBlock(d_model, k=5, dilation=4, dropout=dropout, norm=norm)

        self.pe = PosEnc(d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d_model),
            nhead=int(n_heads),
            dim_feedforward=4 * int(d_model),
            dropout=float(dropout),
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tx = nn.TransformerEncoder(enc_layer, num_layers=int(n_layers))

        self.drop = nn.Dropout(float(dropout))
        self.out_proj = nn.Linear(int(d_model), int(hidden_size))
        # Puts the conditioning sequence on the frozen backbone's embedding scale;
        # `out_rms` is the backbone's own mean per-token embedding RMS, and stays
        # learnable so the model can still adjust it. See forward().
        self.out_norm = nn.LayerNorm(int(hidden_size))
        self.out_scale = nn.Parameter(torch.tensor(float(out_rms)))
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x       : (B, T, in_dim)
        lengths : (B,) valid frame counts. If None, all frames are treated as valid.
        returns : (B, ceil(T/4), hidden_size)
        """
        B, T, _ = x.shape
        mask = lengths_to_mask(lengths, T) if lengths is not None else None

        x = self.in_proj(x).transpose(1, 2)          # (B, d_model, T)
        x = self.sub1(x, mask)
        if mask is not None:
            mask = mask[:, ::2][:, : x.size(-1)]
        x = self.sub2(x, mask)
        if mask is not None:
            mask = mask[:, ::2][:, : x.size(-1)]

        x = self.tcn1(x, mask)
        x = self.se(x, mask)
        x = self.tcn2(x, mask)
        x = self.tcn3(x, mask)

        x = x.transpose(1, 2)                        # (B, T', d_model)
        x = self.pe(x)
        pad_mask = (~mask) if mask is not None else None
        x = self.tx(x, src_key_padding_mask=pad_mask)

        x = self.drop(x)
        x = self.out_proj(x)

        # The adapter output is consumed as `inputs_embeds` by a FROZEN LLM, so it
        # has to look like that LLM's own token embeddings. Left unnormalised it
        # does not: measured against Llama-3.2-3B, out_proj emitted a per-step RMS
        # of 2.53 versus 0.0207 for model.embed_tokens -- 122x out of distribution,
        # which also made the correctly-scaled soft/instruction prompts (RMS 0.021)
        # numerically negligible by comparison. Normalising here and rescaling to
        # `out_scale` puts every conditioning vector on the backbone's own scale.
        x = self.out_norm(x) * self.out_scale
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return x
