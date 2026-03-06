import math
import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable conv:
      depthwise Conv1d(groups=in_ch) + pointwise Conv1d(1x1) + BN + GELU + Dropout
    Input/Output shape: (B, C, T)
    """

    def __init__(self, in_ch: int, out_ch: int, k: int = 5, s: int = 1, p: int | None = None, dropout: float = 0.1):
        super().__init__()
        p = (k // 2) if p is None else int(p)
        self.dw = nn.Conv1d(in_ch, in_ch, kernel_size=k, stride=s, padding=p, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.act(x)
        return self.drop(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation over time dimension for (B, C, T)."""

    def __init__(self, ch: int, r: int = 8):
        super().__init__()
        r = max(1, int(r))
        hidden = max(1, ch // r)
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(ch, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w


class TCNBlock(nn.Module):
    """
    Length-preserving TCN block:
      - depthwise-separable conv (non-dilated, padding=k//2)
      - depthwise dilated conv (padding=dilation*(k//2))
      - pointwise conv + BN + Dropout + GELU
      - residual
    """

    def __init__(self, ch: int, k: int = 5, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        k = int(k)
        dilation = int(dilation)
        pad = k // 2
        pad_dil = dilation * (k // 2)

        self.conv = DepthwiseSeparableConv1d(ch, ch, k=k, s=1, p=pad, dropout=dropout)

        self.dil = nn.Conv1d(
            ch,
            ch,
            kernel_size=k,
            stride=1,
            padding=pad_dil,
            dilation=dilation,
            groups=ch,
            bias=False,
        )
        self.pw = nn.Conv1d(ch, ch, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm1d(ch)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x = self.conv(x)
        x = self.dil(x)
        x = self.pw(x)
        x = self.bn(x)
        x = self.drop(x)
        return self.act(x + res)


class PosEnc(nn.Module):
    """Sinusoidal positional encoding for (B, T, D)."""

    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        d_model = int(d_model)
        max_len = int(max_len)

        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class EMGAdapterV2(nn.Module):
    """
    EMG adapter:
      (B, T, in_dim) -> (B, ceil(T/4), hidden_size)

    Subsampling is fixed to factor 4:
      two stride-2 depthwise-separable convs with padding k//2
      output length: ceil(T/2) then ceil(/2) = ceil(T/4)

    This must match the CTC input-length computation in train/inference.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.subsample_factor = 4  # IMPORTANT: keep consistent with train_emg_llm.py length math

        self.in_proj = nn.Linear(int(in_dim), int(d_model))

        # conv subsampling by 4 (two stride-2 DS-convs)
        self.sub1 = DepthwiseSeparableConv1d(d_model, d_model, k=5, s=2, dropout=dropout)
        self.sub2 = DepthwiseSeparableConv1d(d_model, d_model, k=5, s=2, dropout=dropout)

        # TCN stack (length-preserving)
        self.tcn = nn.Sequential(
            TCNBlock(d_model, k=5, dilation=1, dropout=dropout),
            SEBlock(d_model),
            TCNBlock(d_model, k=5, dilation=2, dropout=dropout),
            TCNBlock(d_model, k=5, dilation=4, dropout=dropout),
        )

        self.pe = PosEnc(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
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

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, in_dim)
        returns: (B, ceil(T/4), hidden_size)
        """
        x = self.in_proj(x)            # (B, T, d_model)
        x = x.transpose(1, 2)          # (B, d_model, T)

        x = self.sub1(x)               # (B, d_model, ceil(T/2))
        x = self.sub2(x)               # (B, d_model, ceil(T/4))

        x = self.tcn(x)                # (B, d_model, ceil(T/4))

        x = x.transpose(1, 2)          # (B, ceil(T/4), d_model)
        x = self.pe(x)
        x = self.tx(x)

        x = self.drop(x)
        return self.out_proj(x)        # (B, ceil(T/4), hidden_size)