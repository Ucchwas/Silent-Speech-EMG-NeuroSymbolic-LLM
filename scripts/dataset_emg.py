#!/usr/bin/env python3
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from scripts.data_utils import TextTransform, load_normalizer
    from scripts.augment import AugmentConfig, augment_features
except ImportError:
    from data_utils import TextTransform, load_normalizer
    from augment import AugmentConfig, augment_features


def _json_for_npy(npy_path: Path) -> Optional[Path]:
    """
    Match the split_data.py behavior:
      - preferred: same stem (X_silent.npy -> X_silent.json)
      - fallback: remove "_silent" (X_silent.npy -> X.json)
    """
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
        if j is None:
            continue
        pairs.append((npy_path, j))
    return pairs


def read_text_from_json(json_path: Path) -> str:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    # common keys seen across variants
    return obj.get("text", obj.get("transcript", obj.get("label", "")))


@dataclass
class Sample:
    feats: np.ndarray  # (T, D)
    text: str
    utt_id: str


class EMGDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        normalizer_path: Optional[str] = None,
        augment: Optional["AugmentConfig"] = None,
        seed: int = 1234,
    ):
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise FileNotFoundError(f"data_dir not found: {self.data_dir}")

        self.pairs = scan_pairs(self.data_dir)
        if not self.pairs:
            raise FileNotFoundError(f"No (npy,json) pairs found in {self.data_dir}")

        self.tt = TextTransform()
        self.augment = augment
        self._seed = int(seed)

        self.normalizer = None
        if normalizer_path is not None and Path(normalizer_path).exists():
            self.normalizer = load_normalizer(normalizer_path)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        npy_path, json_path = self.pairs[idx]
        feats = np.load(npy_path).astype(np.float32)  # (T, D)
        if feats.ndim != 2:
            raise ValueError(f"Expected (T,D) features, got {feats.shape} in {npy_path}")

        if self.normalizer is not None:
            feats = self.normalizer.transform(feats).astype(np.float32)

        # Augment AFTER normalisation, so the noise/gain strengths are in
        # z-scored units and mean the same thing for every channel.
        if self.augment is not None and getattr(self.augment, "enabled", False):
            rng = np.random.default_rng((self._seed * 1_000_003 + idx * 7919 + random.randrange(1 << 30)))
            feats = augment_features(feats, self.augment, rng).astype(np.float32)

        raw_text = read_text_from_json(json_path)
        clean_text = self.tt.clean(raw_text)  # paper-faithful normalization

        utt_id = npy_path.stem
        return feats, clean_text, utt_id


def collate_fn(batch, pad_idx: int, tt: TextTransform):
    """
    Returns:
      emg_feats:  (B, T_max, D)
      emg_mask:   (B, T_max) bool
      in_ids:     (B, L_max) long     (BOS + chars)
      out_ids:    (B, L_max) long     (chars + EOS)
      char_mask:  (B, L_max) bool     (valid positions in out_ids)
      texts:      list[str] cleaned text
    """
    feats_list, texts, utt_ids = zip(*batch)

    # EMG padding
    lengths = [x.shape[0] for x in feats_list]
    T_max = max(lengths)
    D = feats_list[0].shape[1]

    emg_feats = torch.zeros((len(batch), T_max, D), dtype=torch.float32)
    emg_mask = torch.zeros((len(batch), T_max), dtype=torch.bool)

    for i, x in enumerate(feats_list):
        t = x.shape[0]
        emg_feats[i, :t] = torch.from_numpy(x)
        emg_mask[i, :t] = True

    # Text -> ids
    # in_ids:  BOS + chars
    # out_ids: chars + EOS
    seqs = [tt.text_to_ids(txt, add_bos_eos=True) for txt in texts]
    # seq: [BOS, ..., EOS]
    in_seqs = [s[:-1] for s in seqs]
    out_seqs = [s[1:] for s in seqs]

    L_max = max(len(s) for s in out_seqs)
    in_ids = torch.full((len(batch), L_max), pad_idx, dtype=torch.long)
    out_ids = torch.full((len(batch), L_max), pad_idx, dtype=torch.long)
    char_mask = torch.zeros((len(batch), L_max), dtype=torch.bool)

    for i, (inp, outp) in enumerate(zip(in_seqs, out_seqs)):
        L = len(outp)
        in_ids[i, :L] = torch.tensor(inp, dtype=torch.long)
        out_ids[i, :L] = torch.tensor(outp, dtype=torch.long)
        char_mask[i, :L] = True

    return emg_feats, emg_mask, in_ids, out_ids, char_mask, list(texts)


def make_loader(
    data_dir: Path,
    normalizer_path: Optional[str],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 0,
    augment: Optional[AugmentConfig] = None,
    seed: int = 1234,
    distributed: bool = False,
):
    ds = EMGDataset(str(data_dir), normalizer_path=normalizer_path, augment=augment, seed=seed)
    tt = ds.tt
    pad_idx = tt.PAD_IDX

    # Under DDP the training set is split across ranks by a DistributedSampler,
    # which owns the shuffling; DataLoader(shuffle=...) must then be False.
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(ds, shuffle=shuffle, seed=seed, drop_last=False)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=lambda b: collate_fn(b, pad_idx=pad_idx, tt=tt),
    )
    return loader, ds, tt