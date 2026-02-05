#!/usr/bin/env python3
"""
Assemble a Kohya-compatible dataset for P3D multi-character SDXL checkpoint training.

Inputs:
  - Filtered single-character data (tier_a + tier_b) produced by:
      scripts/generic/training/quality_filters/synthetic_quality_pipeline.py
  - Pair interaction dataset produced by:
      scripts/batch/run_p3d_multichar_interactions_8k.sh

Output layout (Kohya expects repeats via folder name):
  {out_dir}/
    {solo_repeats}_p3d_solo/
      <unique_name>.png
      <unique_name>.txt
    {pair_repeats}_p3d_pairs/
      pair_00000.png
      pair_00000.txt

Notes:
  - Single-character images are renamed to avoid filename collisions when merged.
  - Captions must exist next to each image in the filtered tiers.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CopyStats:
    images: int = 0
    captions: int = 0
    missing_captions: int = 0
    skipped_existing: int = 0


def _iter_filtered_images(filtered_root: Path, tiers: List[str]) -> Iterable[Tuple[str, str, Path]]:
    """
    Yields (character, lora_type, image_path) for images under:
      filtered_root/{character}/{lora_type}/tier_{a|b}/...
    """
    for character_dir in sorted([p for p in filtered_root.iterdir() if p.is_dir()]):
        character = character_dir.name
        for lora_type_dir in sorted([p for p in character_dir.iterdir() if p.is_dir()]):
            lora_type = lora_type_dir.name
            for tier in tiers:
                tier_dir = lora_type_dir / f"tier_{tier.lower()}"
                if not tier_dir.exists():
                    continue
                for img in sorted(tier_dir.glob("*.png")):
                    yield (character, lora_type, img)


def _copy_pair_dataset(pair_dir: Path, out_dir: Path, overwrite: bool) -> CopyStats:
    images_dir = pair_dir / "images"
    captions_dir = pair_dir / "captions"
    if not images_dir.exists() or not captions_dir.exists():
        raise FileNotFoundError(f"Pair dataset must contain images/ and captions/: {pair_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stats = CopyStats()

    for img in sorted(images_dir.glob("*.png")):
        stem = img.stem
        cap = captions_dir / f"{stem}.txt"
        dst_img = out_dir / img.name
        dst_cap = out_dir / f"{stem}.txt"

        if dst_img.exists() and not overwrite:
            stats = CopyStats(
                images=stats.images,
                captions=stats.captions,
                missing_captions=stats.missing_captions,
                skipped_existing=stats.skipped_existing + 1,
            )
            continue

        shutil.copy2(img, dst_img)
        stats = CopyStats(
            images=stats.images + 1,
            captions=stats.captions,
            missing_captions=stats.missing_captions,
            skipped_existing=stats.skipped_existing,
        )

        if cap.exists():
            shutil.copy2(cap, dst_cap)
            stats = CopyStats(
                images=stats.images,
                captions=stats.captions + 1,
                missing_captions=stats.missing_captions,
                skipped_existing=stats.skipped_existing,
            )
        else:
            stats = CopyStats(
                images=stats.images,
                captions=stats.captions,
                missing_captions=stats.missing_captions + 1,
                skipped_existing=stats.skipped_existing,
            )

    return stats


def _copy_solo_filtered(
    filtered_root: Path,
    out_dir: Path,
    tiers: List[str],
    overwrite: bool,
) -> CopyStats:
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = CopyStats()

    for character, lora_type, img_path in _iter_filtered_images(filtered_root, tiers=tiers):
        caption_path = img_path.with_suffix(".txt")

        # Rename to avoid collisions across characters/types/tiers
        unique_stem = f"{character}_{lora_type}_{img_path.stem}"
        dst_img = out_dir / f"{unique_stem}.png"
        dst_cap = out_dir / f"{unique_stem}.txt"

        if dst_img.exists() and not overwrite:
            stats = CopyStats(
                images=stats.images,
                captions=stats.captions,
                missing_captions=stats.missing_captions,
                skipped_existing=stats.skipped_existing + 1,
            )
            continue

        shutil.copy2(img_path, dst_img)
        stats = CopyStats(
            images=stats.images + 1,
            captions=stats.captions,
            missing_captions=stats.missing_captions,
            skipped_existing=stats.skipped_existing,
        )

        if caption_path.exists():
            shutil.copy2(caption_path, dst_cap)
            stats = CopyStats(
                images=stats.images,
                captions=stats.captions + 1,
                missing_captions=stats.missing_captions,
                skipped_existing=stats.skipped_existing,
            )
        else:
            stats = CopyStats(
                images=stats.images,
                captions=stats.captions,
                missing_captions=stats.missing_captions + 1,
                skipped_existing=stats.skipped_existing,
            )

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Assemble Kohya dataset for P3D multi-character checkpoint training.")
    ap.add_argument("--filtered-root", type=Path, required=True, help="Filtered solo root (contains {char}/{type}/tier_a|b).")
    ap.add_argument("--pair-dataset", type=Path, required=True, help="Pair dataset root (contains images/ and captions/).")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output dataset root (Kohya format subfolders).")
    ap.add_argument("--solo-repeats", type=int, default=1)
    ap.add_argument("--pair-repeats", type=int, default=2)
    ap.add_argument("--tiers", nargs="+", default=["A", "B"], help="Which tiers to include (default: A B).")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    solo_dir = out_dir / f"{args.solo_repeats}_p3d_solo"
    pairs_dir = out_dir / f"{args.pair_repeats}_p3d_pairs"

    logger.info("Solo filtered root: %s", args.filtered_root)
    logger.info("Pair dataset: %s", args.pair_dataset)
    logger.info("Output dataset root: %s", out_dir)
    logger.info("Solo dir: %s", solo_dir)
    logger.info("Pairs dir: %s", pairs_dir)

    solo_stats = _copy_solo_filtered(
        filtered_root=args.filtered_root,
        out_dir=solo_dir,
        tiers=args.tiers,
        overwrite=args.overwrite,
    )
    logger.info("Solo copy stats: %s", asdict(solo_stats))

    pair_stats = _copy_pair_dataset(pair_dir=args.pair_dataset, out_dir=pairs_dir, overwrite=args.overwrite)
    logger.info("Pairs copy stats: %s", asdict(pair_stats))

    manifest = {
        "filtered_root": str(args.filtered_root),
        "pair_dataset": str(args.pair_dataset),
        "out_dir": str(out_dir),
        "solo_repeats": args.solo_repeats,
        "pair_repeats": args.pair_repeats,
        "tiers": args.tiers,
        "solo": asdict(solo_stats),
        "pairs": asdict(pair_stats),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Wrote manifest: %s", out_dir / "manifest.json")


if __name__ == "__main__":
    main()

