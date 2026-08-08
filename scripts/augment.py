#!/usr/bin/env python3
"""
scripts/augment.py

Training-time augmentation for EMG feature sequences.

Why this exists: the closed-vocabulary silent split has only 400 training
utterances against a ~17M-parameter adapter. Without augmentation the adapter
memorises the training set outright -- measured on this data, CTC-greedy
reaches WER 0.016 / 96% exact-match on train while sitting at WER 0.684 / 4%
on validation. Nothing in the decoder can recover from an encoder that has not
generalised, so this is the highest-leverage part of the recipe.

Everything operates on the extracted 112-dim feature frames, laid out as
8 channels x 14 descriptors (channel c occupies dims [14c, 14c+14)), so
channel-level transforms address genuine per-electrode variation rather than
arbitrary feature indices.

Transforms (all applied to a copy, all off by default at strength 0):
  * gain jitter    per-channel multiplicative gain -- electrode impedance and
                   placement differ from session to session
  * noise          additive Gaussian noise on standardised features
  * time mask      SpecAugment-style spans zeroed along time
  * channel mask   whole electrodes dropped, forcing redundancy across channels
  * time warp      resampling the time axis, i.e. speaking-rate variation
  * shift          small circular/temporal offset
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


FEATS_PER_CHANNEL = 14


@dataclass
class AugmentConfig:
    """Strengths for each transform. 0 disables that transform."""

    enabled: bool = False

    gain_std: float = 0.10        # per-channel log-gain sigma
    noise_std: float = 0.10       # additive Gaussian sigma (features are z-scored)

    time_mask_prob: float = 0.5   # probability of applying any time masking
    time_mask_num: int = 2        # number of masked spans
    time_mask_max: float = 0.10   # max span length as a fraction of T

    channel_mask_prob: float = 0.3
    channel_mask_num: int = 1     # electrodes dropped when it fires

    time_warp_prob: float = 0.5
    time_warp_range: float = 0.10  # resample factor ~ U(1-r, 1+r)

    shift_max: int = 5            # max temporal shift in frames

    def scaled(self, s: float) -> "AugmentConfig":
        """Uniformly scale every strength (1.0 = as configured, 0.5 = half)."""
        return AugmentConfig(
            enabled=self.enabled,
            gain_std=self.gain_std * s,
            noise_std=self.noise_std * s,
            time_mask_prob=self.time_mask_prob * s,
            time_mask_num=self.time_mask_num,
            time_mask_max=self.time_mask_max * s,
            channel_mask_prob=self.channel_mask_prob * s,
            channel_mask_num=self.channel_mask_num,
            time_warp_prob=self.time_warp_prob * s,
            time_warp_range=self.time_warp_range * s,
            shift_max=int(round(self.shift_max * s)),
        )


def _time_warp(x: np.ndarray, factor: float) -> np.ndarray:
    """Resample the time axis by `factor` with linear interpolation."""
    T, D = x.shape
    new_T = max(4, int(round(T * factor)))
    src = np.linspace(0.0, T - 1.0, num=new_T)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo).astype(np.float32)[:, None]
    return (1.0 - w) * x[lo] + w * x[hi]


def augment_features(
    x: np.ndarray, cfg: AugmentConfig, rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """
    Apply the configured augmentations to one (T, D) feature array.

    Returns a new array; `x` is never modified in place. The time axis may
    change length (time warp), which downstream length bookkeeping handles.
    """
    if not cfg.enabled:
        return x
    rng = rng or np.random.default_rng()
    y = np.array(x, dtype=np.float32, copy=True)
    T, D = y.shape
    n_ch = max(1, D // FEATS_PER_CHANNEL)

    # --- speaking-rate variation -------------------------------------------
    if cfg.time_warp_range > 0 and rng.random() < cfg.time_warp_prob:
        f = float(rng.uniform(1.0 - cfg.time_warp_range, 1.0 + cfg.time_warp_range))
        y = _time_warp(y, f).astype(np.float32)
        T = y.shape[0]

    # --- temporal shift -----------------------------------------------------
    if cfg.shift_max > 0:
        s = int(rng.integers(-cfg.shift_max, cfg.shift_max + 1))
        if s:
            y = np.roll(y, s, axis=0)
            if s > 0:
                y[:s] = 0.0
            else:
                y[s:] = 0.0

    # --- per-channel gain ---------------------------------------------------
    if cfg.gain_std > 0:
        g = np.exp(rng.normal(0.0, cfg.gain_std, size=n_ch)).astype(np.float32)
        for c in range(n_ch):
            y[:, c * FEATS_PER_CHANNEL : (c + 1) * FEATS_PER_CHANNEL] *= g[c]

    # --- whole-electrode dropout -------------------------------------------
    if cfg.channel_mask_num > 0 and rng.random() < cfg.channel_mask_prob:
        for c in rng.choice(n_ch, size=min(cfg.channel_mask_num, n_ch), replace=False):
            y[:, c * FEATS_PER_CHANNEL : (c + 1) * FEATS_PER_CHANNEL] = 0.0

    # --- time masking -------------------------------------------------------
    if cfg.time_mask_num > 0 and cfg.time_mask_max > 0 and rng.random() < cfg.time_mask_prob:
        span_max = max(1, int(T * cfg.time_mask_max))
        for _ in range(cfg.time_mask_num):
            w = int(rng.integers(1, span_max + 1))
            if w >= T:
                continue
            t0 = int(rng.integers(0, T - w))
            y[t0 : t0 + w] = 0.0

    # --- additive noise -----------------------------------------------------
    if cfg.noise_std > 0:
        y += rng.normal(0.0, cfg.noise_std, size=y.shape).astype(np.float32)

    return y.astype(np.float32)
