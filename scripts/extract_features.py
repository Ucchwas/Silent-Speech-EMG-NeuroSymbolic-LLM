#!/usr/bin/env python3
"""
scripts/extract_features.py

EMG preprocessing and feature extraction (Sec. III-C).

Pipeline, per utterance:
  1. per-channel DC removal
  2. high-pass filter to suppress baseline drift
  3. notch filter at the mains frequency (and its harmonics)
  4. segmentation into utterances using the dataset boundaries
  5. framing with a 27 ms window and a 10 ms hop
  6. decomposition of each channel into a low-frequency component (134 Hz
     low-pass; envelope-like activation) and the high-frequency residual
  7. 14 descriptors per channel per frame:
       - 5 scalars derived from the low/high components:
           w_h  mean of the low component
           p_w  RMS power of the low component
           p_r  RMS power of the rectified high component
           z_p  zero-crossing rate of the high component
           r_h  mean of the rectified high component
       - 9 low-frequency STFT magnitude bins of the low component
  8. concatenation across the 8 channels -> a 112-dimensional frame vector

Output matches `data/extracted_emg_features/`: one `<utt>.npy` of shape
(T, 112) float32 plus a `<utt>.json` sidecar carrying the transcript.

Per-feature z-normalisation is NOT applied here. It is fitted on the training
split only inside train_emg_llm.py and reused for val/test, which is what
keeps the statistics leak-free.

NOTE: this script needs the raw Silent Speech EMG recordings. The repository
ships only the already-extracted 112-dim features, so the defaults below
reproduce the documented recipe but have not been re-validated against raw
audio-free EMG in this checkout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy import signal as sps
except ImportError as e:  # pragma: no cover
    raise SystemExit("scipy is required for feature extraction: pip install scipy") from e


N_SCALARS = 5
N_STFT_BINS = 9
FEATS_PER_CHANNEL = N_SCALARS + N_STFT_BINS  # 14


# -----------------------------------------------------------------------------
# Filtering
# -----------------------------------------------------------------------------
def highpass(x: np.ndarray, fs: float, cutoff: float = 2.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth high-pass to remove baseline drift."""
    b, a = sps.butter(order, cutoff / (fs / 2.0), btype="highpass")
    return sps.filtfilt(b, a, x, axis=0)


def notch(x: np.ndarray, fs: float, freq: float = 60.0, q: float = 30.0,
          harmonics: int = 3) -> np.ndarray:
    """Zero-phase notch at the mains frequency and its first few harmonics."""
    y = x
    f = freq
    k = 0
    while k < harmonics and f < fs / 2.0:
        b, a = sps.iirnotch(f / (fs / 2.0), q)
        y = sps.filtfilt(b, a, y, axis=0)
        f += freq
        k += 1
    return y


def lowpass(x: np.ndarray, fs: float, cutoff: float = 134.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth low-pass separating the envelope-like component."""
    nyq = fs / 2.0
    if cutoff >= nyq:
        return x
    b, a = sps.butter(order, cutoff / nyq, btype="lowpass")
    return sps.filtfilt(b, a, x, axis=0)


# -----------------------------------------------------------------------------
# Framing
# -----------------------------------------------------------------------------
def frame_signal(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Split a 1-D signal into (n_frames, win) without copying where possible."""
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def zero_crossing_rate(frames: np.ndarray) -> np.ndarray:
    """Fraction of sign changes within each frame."""
    s = np.signbit(frames)
    return np.mean(s[:, 1:] != s[:, :-1], axis=1)


def channel_features(
    x: np.ndarray, fs: float, win: int, hop: int, lp_cutoff: float, n_bins: int
) -> np.ndarray:
    """14 descriptors per frame for one channel: (n_frames, 14)."""
    w = lowpass(x, fs, cutoff=lp_cutoff)   # low-frequency (envelope-like) component
    p = x - w                              # high-frequency residual
    r = np.abs(p)                          # rectified high component

    fw = frame_signal(w, win, hop)
    fp = frame_signal(p, win, hop)
    fr = frame_signal(r, win, hop)

    w_h = fw.mean(axis=1)
    p_w = np.sqrt(np.mean(fw ** 2, axis=1))
    p_r = np.sqrt(np.mean(fr ** 2, axis=1))
    z_p = zero_crossing_rate(fp)
    r_h = fr.mean(axis=1)
    scalars = np.stack([w_h, p_w, p_r, z_p, r_h], axis=1)  # (n, 5)

    # Magnitude spectrum of the low component; keep the n_bins lowest bins.
    windowed = fw * np.hanning(win)[None, :]
    n_fft = int(2 ** np.ceil(np.log2(max(win, 2 * (n_bins - 1)))))
    mag = np.abs(np.fft.rfft(windowed, n=n_fft, axis=1))[:, :n_bins]  # (n, 9)

    return np.concatenate([scalars, mag], axis=1).astype(np.float32)


def extract_utterance(
    emg: np.ndarray,
    fs: float = 1000.0,
    win_ms: float = 27.0,
    hop_ms: float = 10.0,
    hp_cutoff: float = 2.0,
    mains: float = 60.0,
    lp_cutoff: float = 134.0,
    n_bins: int = N_STFT_BINS,
) -> np.ndarray:
    """
    Raw multi-channel EMG (n_samples, n_channels) -> (n_frames, 14*n_channels).

    With the 8-channel facial montage this yields the 112-dim vectors used
    throughout the paper.
    """
    if emg.ndim != 2:
        raise ValueError(f"expected (samples, channels), got {emg.shape}")

    x = emg.astype(np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    x = highpass(x, fs, cutoff=hp_cutoff)
    x = notch(x, fs, freq=mains)

    win = max(2, int(round(win_ms * fs / 1000.0)))
    hop = max(1, int(round(hop_ms * fs / 1000.0)))

    per_ch = [
        channel_features(x[:, c], fs, win, hop, lp_cutoff, n_bins)
        for c in range(x.shape[1])
    ]
    n = min(f.shape[0] for f in per_ch)
    return np.concatenate([f[:n] for f in per_ch], axis=1).astype(np.float32)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Extract 112-dim EMG features (Sec. III-C).")
    ap.add_argument("--src", required=True,
                    help="Directory of raw EMG recordings (*.npy of shape (samples, channels)).")
    ap.add_argument("--out", required=True, help="Output directory for features + sidecars.")
    ap.add_argument("--pattern", default="*.npy")
    ap.add_argument("--fs", type=float, default=1000.0, help="Sampling rate in Hz.")
    ap.add_argument("--win_ms", type=float, default=27.0)
    ap.add_argument("--hop_ms", type=float, default=10.0)
    ap.add_argument("--hp_cutoff", type=float, default=2.0)
    ap.add_argument("--mains", type=float, default=60.0)
    ap.add_argument("--lp_cutoff", type=float, default=134.0)
    args = ap.parse_args()

    src, out = Path(args.src), Path(args.out)
    if not src.exists():
        raise FileNotFoundError(src)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(src.glob(args.pattern))
    if not files:
        raise FileNotFoundError(f"no files matching {args.pattern} in {src}")

    n_done = 0
    for fp in files:
        emg = np.load(fp)
        feats = extract_utterance(
            emg, fs=args.fs, win_ms=args.win_ms, hop_ms=args.hop_ms,
            hp_cutoff=args.hp_cutoff, mains=args.mains, lp_cutoff=args.lp_cutoff,
        )
        np.save(out / fp.name, feats)

        side = fp.with_suffix(".json")
        if side.exists():
            (out / side.name).write_text(side.read_text(encoding="utf-8"), encoding="utf-8")
        n_done += 1

    print(f"[extract_features] wrote {n_done} files to {out} "
          f"(dim = {FEATS_PER_CHANNEL} x n_channels)")


if __name__ == "__main__":
    main()
