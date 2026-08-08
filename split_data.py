#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import random
from pathlib import Path
from typing import List, Optional, Tuple


def find_pairs(src_dir: Path, pattern: str) -> List[Tuple[Path, Optional[Path]]]:
    """
    Returns list of (npy_path, json_path_or_None).
    We try:
      1) same stem: X_silent.npy -> X_silent.json
      2) without "_silent": X_silent.npy -> X.json
    """
    npys = sorted(src_dir.glob(pattern))
    pairs: List[Tuple[Path, Optional[Path]]] = []
    for npy in npys:
        j1 = npy.with_suffix(".json")
        j2 = npy.parent / f"{npy.stem.replace('_silent', '')}.json"
        if j1.exists():
            pairs.append((npy, j1))
        elif j2.exists():
            pairs.append((npy, j2))
        else:
            pairs.append((npy, None))
    return pairs


def ensure_empty_dir(d: Path, clean: bool) -> None:
    if d.exists() and clean:
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def copy_split(pairs: List[Tuple[Path, Optional[Path]]], out_dir: Path) -> int:
    """
    Copy *.npy and corresponding *.json into out_dir.
    JSON is renamed to match the npy stem, so dataset_emg.py can pair by same stem.
    Returns number of missing json sidecars.
    """
    missing = 0
    for npy_path, json_path in pairs:
        shutil.copy2(npy_path, out_dir / npy_path.name)
        if json_path is None:
            missing += 1
            continue
        shutil.copy2(json_path, out_dir / f"{npy_path.stem}.json")
    return missing


def load_manifest(path: Path) -> dict:
    """
    Read a split manifest: {"train": [...], "val": [...], "test": [...]} of
    utterance stems (e.g. "102_silent").

    A manifest pins the exact partition, which is the only way to reproduce a
    split bit-for-bit later. Prefer it over --shuffle/--seed when the numbers
    have to match a previous run.
    """
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    missing = [k for k in ("train", "val", "test") if k not in obj]
    if missing:
        raise ValueError(f"manifest {path} is missing keys: {missing}")
    return {k: [str(s) for s in obj[k]] for k in ("train", "val", "test")}


def write_manifest(path: Path, splits: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: [p.stem for p, _ in v] for k, v in splits.items()}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split silent EMG features into 400/50/50 train/val/test (500 total).")

    p.add_argument("--src_dir", type=str, default="data/extracted_emg_features",
                   help="Directory containing extracted EMG feature files and JSON sidecars.")
    p.add_argument("--pattern", type=str, default="*_silent.npy",
                   help="Glob pattern for EMG feature files.")
    p.add_argument("--out_root", type=str, default="data",
                   help="Output root directory. Creates train_emg/val_emg/test_emg under this.")

    p.add_argument("--total", type=int, default=500, help="Total number of utterances to use (paper uses 500).")
    p.add_argument("--train_n", type=int, default=400, help="Train count.")
    p.add_argument("--val_n", type=int, default=50, help="Validation count.")
    p.add_argument("--test_n", type=int, default=50, help="Test count.")

    p.add_argument("--shuffle", action="store_true",
                   help="Shuffle before splitting. Default: deterministic sorted order.")
    p.add_argument("--seed", type=int, default=42, help="Random seed used only if --shuffle is set.")
    p.add_argument("--clean", action="store_true",
                   help="If set, delete existing split folders before writing.")

    p.add_argument("--manifest", type=str, default="",
                   help="Reproduce an exact split from a JSON manifest "
                        "{'train':[stems],'val':[...],'test':[...]}. Overrides the counts and shuffling.")
    p.add_argument("--write_manifest", type=str, default="artifacts/split_manifest.json",
                   help="Where to record the split that was produced (set empty to skip).")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    src_dir = Path(args.src_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"src_dir not found: {src_dir}")

    out_root = Path(args.out_root)
    train_dir = out_root / "train_emg"
    val_dir = out_root / "val_emg"
    test_dir = out_root / "test_emg"

    ensure_empty_dir(train_dir, clean=args.clean)
    ensure_empty_dir(val_dir, clean=args.clean)
    ensure_empty_dir(test_dir, clean=args.clean)

    pairs = find_pairs(src_dir, args.pattern)
    print(f"[split_data] Found {len(pairs)} files matching {args.pattern} in {src_dir}")

    if args.manifest:
        wanted = load_manifest(Path(args.manifest))
        by_stem = {npy.stem: (npy, js) for npy, js in pairs}
        missing = [s for part in wanted.values() for s in part if s not in by_stem]
        if missing:
            raise ValueError(
                f"manifest references {len(missing)} utterances absent from {src_dir}, "
                f"e.g. {missing[:5]}"
            )
        train_pairs = [by_stem[s] for s in wanted["train"]]
        val_pairs = [by_stem[s] for s in wanted["val"]]
        test_pairs = [by_stem[s] for s in wanted["test"]]
        print(f"[split_data] Using manifest {args.manifest}")
    else:
        need = args.train_n + args.val_n + args.test_n
        if need != args.total:
            raise ValueError(f"train_n+val_n+test_n must equal total. Got {need} vs total={args.total}.")
        if len(pairs) < args.total:
            raise ValueError(f"Not enough files: found {len(pairs)}, need {args.total}.")

        pairs = pairs[: args.total]
        if args.shuffle:
            random.Random(args.seed).shuffle(pairs)
            print(f"[split_data] Shuffled with seed={args.seed}")
        else:
            print("[split_data] Deterministic sorted order (no shuffle)")

        train_pairs = pairs[: args.train_n]
        val_pairs = pairs[args.train_n : args.train_n + args.val_n]
        test_pairs = pairs[args.train_n + args.val_n : args.train_n + args.val_n + args.test_n]

    overlap = (
        {p.stem for p, _ in train_pairs}
        & ({p.stem for p, _ in val_pairs} | {p.stem for p, _ in test_pairs})
    ) | ({p.stem for p, _ in val_pairs} & {p.stem for p, _ in test_pairs})
    if overlap:
        raise ValueError(f"splits overlap on {len(overlap)} utterances, e.g. {sorted(overlap)[:5]}")

    print(f"[split_data] Train: {len(train_pairs)} | Val: {len(val_pairs)} | Test: {len(test_pairs)}")

    miss_tr = copy_split(train_pairs, train_dir)
    miss_va = copy_split(val_pairs, val_dir)
    miss_te = copy_split(test_pairs, test_dir)

    if (miss_tr + miss_va + miss_te) > 0:
        print(f"[split_data] WARNING: missing JSON sidecars -> train:{miss_tr}, val:{miss_va}, test:{miss_te}")
    else:
        print("[split_data] All JSON sidecars found.")

    if args.write_manifest:
        mpath = Path(args.write_manifest)
        write_manifest(mpath, {"train": train_pairs, "val": val_pairs, "test": test_pairs})
        print(f"[split_data] Recorded split manifest -> {mpath}")

    print("[split_data] Done.")


if __name__ == "__main__":
    main()