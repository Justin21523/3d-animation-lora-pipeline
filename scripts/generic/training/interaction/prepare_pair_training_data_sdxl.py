#!/usr/bin/env python3
"""
Prepare pair datasets for Kohya SDXL training (repeat_{concept} folder format).

Input (from composer):
  {pairs_root}/{A}__{B}/images/*.png
  {pairs_root}/{A}__{B}/captions/*.txt

Output (Kohya format):
  {out_root}/{A}__{B}/{repeats}_{concept}/<basename>.png
  {out_root}/{A}__{B}/{repeats}_{concept}/<basename>.txt

By default uses symlinks to avoid duplicating data.
"""

from __future__ import annotations

import argparse
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple


@dataclass(frozen=True)
class PairDataset:
    name: str
    images_dir: Path
    captions_dir: Path


def _safe_link(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        os.symlink(str(src), str(dst))
    elif mode == "hardlink":
        os.link(str(src), str(dst))
    elif mode == "copy":
        shutil.copy2(src, dst)
    else:
        raise ValueError(f"Unknown link mode: {mode}")


def _iter_pair_datasets(root: Path) -> Iterable[PairDataset]:
    for d in sorted([p for p in root.iterdir() if p.is_dir()]):
        images_dir = d / "images"
        captions_dir = d / "captions"
        if images_dir.is_dir() and captions_dir.is_dir():
            yield PairDataset(name=d.name, images_dir=images_dir, captions_dir=captions_dir)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-root", type=Path, required=True, help="Root containing <A>__<B>/images and captions")
    ap.add_argument("--out-root", type=Path, required=True, help="Output root for Kohya train_data_dir")
    ap.add_argument("--repeats", type=int, default=5, help="Folder repeat count (e.g., 5 => 5_pair)")
    ap.add_argument(
        "--concept",
        default="pair",
        help="Concept name used in repeat folder (e.g., 'pair' => 5_pair)",
    )
    ap.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output folder already exists")
    ap.add_argument("--limit", type=int, default=None, help="Optional limit number of pair datasets")
    args = ap.parse_args()

    if args.repeats <= 0:
        raise SystemExit("--repeats must be > 0")

    out_root = args.out_root
    out_root.mkdir(parents=True, exist_ok=True)
    repeat_dirname = f"{int(args.repeats)}_{str(args.concept)}"

    prepared = 0
    total_images = 0
    total_missing_caps = 0

    for ds in _iter_pair_datasets(args.pairs_root):
        if args.limit is not None and prepared >= int(args.limit):
            break

        out_pair_dir = out_root / ds.name
        out_concept_dir = out_pair_dir / repeat_dirname

        if args.skip_existing and out_concept_dir.exists():
            prepared += 1
            continue

        out_concept_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(ds.images_dir.glob("*.png"))
        missing_caps = 0
        for img in images:
            cap = ds.captions_dir / f"{img.stem}.txt"
            if not cap.exists():
                missing_caps += 1
                continue

            _safe_link(img, out_concept_dir / img.name, args.mode)
            _safe_link(cap, out_concept_dir / cap.name, args.mode)
            total_images += 1

        total_missing_caps += missing_caps
        prepared += 1

    print(
        f"Prepared {prepared} pair datasets under {out_root} "
        f"({repeat_dirname}), linked_images={total_images}, missing_captions={total_missing_caps}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

