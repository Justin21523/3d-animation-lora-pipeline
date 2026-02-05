#!/usr/bin/env python3
"""
Assemble the FINAL Kohya dataset for P3D multi-character checkpoint training.

Combines:
  - Solo (tier_a+b filtered, tokenized captions)
  - Pairs (cleaned captions)
  - (Optional) Style generalization (no character tokens; p3d_style only)

Output layout:
  out_dir/
    {solo_repeats}_p3d_solo/
    {pair_repeats}_p3d_pairs/
    {style_repeats}_p3d_style_generic/   (only if style_repeats > 0)
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CopyStats:
    images: int = 0
    captions: int = 0
    missing_captions: int = 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_flat_dir(src_dir: Path, dst_dir: Path) -> CopyStats:
    ensure_dir(dst_dir)
    stats = CopyStats()
    for png in sorted(src_dir.glob("*.png")):
        txt = png.with_suffix(".txt")
        shutil.copy2(png, dst_dir / png.name)
        stats = CopyStats(images=stats.images + 1, captions=stats.captions, missing_captions=stats.missing_captions)
        if txt.exists():
            shutil.copy2(txt, dst_dir / txt.name)
            stats = CopyStats(images=stats.images, captions=stats.captions + 1, missing_captions=stats.missing_captions)
        else:
            stats = CopyStats(images=stats.images, captions=stats.captions, missing_captions=stats.missing_captions + 1)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solo-dir", type=Path, required=True, help="Flat dir of solo png/txt (already renamed).")
    ap.add_argument("--pairs-dir", type=Path, required=True, help="Flat dir of pair png/txt.")
    ap.add_argument(
        "--style-dir",
        type=Path,
        default=None,
        help="Flat dir of style generic png/txt (optional when --style-repeats=0).",
    )
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--solo-repeats", type=int, default=1)
    ap.add_argument("--pair-repeats", type=int, default=2)
    ap.add_argument("--style-repeats", type=int, default=0)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    out = args.out_dir
    ensure_dir(out)

    solo_out = out / f"{args.solo_repeats}_p3d_solo"
    pairs_out = out / f"{args.pair_repeats}_p3d_pairs"
    style_out = out / f"{args.style_repeats}_p3d_style_generic"

    logger.info("Copy solo...")
    solo_stats = copy_flat_dir(args.solo_dir, solo_out)
    logger.info("Copy pairs...")
    pair_stats = copy_flat_dir(args.pairs_dir, pairs_out)
    style_stats: Optional[CopyStats] = None
    if args.style_repeats > 0:
        if args.style_dir is None:
            raise ValueError("--style-dir is required when --style-repeats > 0")
        logger.info("Copy style generic...")
        style_stats = copy_flat_dir(args.style_dir, style_out)

    manifest = {
        "solo_dir": str(args.solo_dir),
        "pairs_dir": str(args.pairs_dir),
        "style_dir": str(args.style_dir) if args.style_dir else "",
        "out_dir": str(out),
        "solo_repeats": args.solo_repeats,
        "pair_repeats": args.pair_repeats,
        "style_repeats": args.style_repeats,
        "solo": asdict(solo_stats),
        "pairs": asdict(pair_stats),
        "style": asdict(style_stats) if style_stats else None,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Done. manifest=%s", out / "manifest.json")


if __name__ == "__main__":
    main()
