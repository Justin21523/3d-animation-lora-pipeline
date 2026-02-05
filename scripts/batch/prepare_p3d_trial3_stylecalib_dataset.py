#!/usr/bin/env python3
"""
Prepare a Trial-3 Kohya dataset focused on:
  - Preserving 2-character interaction stability
  - Reducing "collage / composited" feel by cleaning pair captions
  - Pulling overall lighting/material style toward reference images from:
      /mnt/data/ai_data/synthetic_lora_data/generated_data (excluding yokai-watch)

This script creates a NEW dataset root with:
  - 1_p3d_solo      (symlinks to images, copies captions)
  - 2_p3d_pairs     (symlinks to images, writes CLEANED captions)
  - 2_p3d_style_ref (copies sampled reference images + simple captions)

Why symlinks:
  - Avoid duplicating tens of thousands of PNGs
  - Allow caption edits without mutating the original dataset
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)


REMOVE_TOKENS_EXACT = {
    "left character",
    "right character",
    "simple background",
}

REMOVE_PREFIXES = (
    "template_",
    "gap_px_",
    "background_",
)


def split_caption(text: str) -> List[str]:
    parts = [p.strip() for p in text.strip().split(",") if p.strip()]
    return parts


def join_caption(parts: Sequence[str]) -> str:
    return ", ".join([p.strip() for p in parts if p.strip()]).strip() + "\n"


def clean_pair_caption(text: str) -> str:
    parts = split_caption(text)
    if len(parts) < 4:
        return text.strip() + "\n"

    cleaned: List[str] = []
    for p in parts:
        if p in REMOVE_TOKENS_EXACT:
            continue
        if any(p.startswith(pref) for pref in REMOVE_PREFIXES):
            continue
        cleaned.append(p)

    # Ensure key tokens remain in front: p3d_A, p3d_B, p3d_style, two characters
    # If input already follows that order, keep it. Otherwise don't over-correct here.
    return join_caption(cleaned)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, use_symlinks: bool) -> None:
    if dst.exists():
        return
    ensure_dir(dst.parent)
    if use_symlinks:
        dst.symlink_to(src)
    else:
        shutil.copy2(src, dst)


def iter_pngs(root: Path) -> Iterable[Path]:
    yield from root.glob("*.png")


@dataclass(frozen=True)
class SampleSource:
    character: str
    lora_type: str
    images_dir: Path


def find_reference_sources(generated_root: Path) -> List[SampleSource]:
    sources: List[SampleSource] = []
    for ch_dir in sorted([p for p in generated_root.iterdir() if p.is_dir()]):
        if ch_dir.name == "yokai-watch":
            continue
        for lora_type in ("action", "expression", "pose"):
            images_dir = ch_dir / lora_type / "images"
            if images_dir.exists():
                sources.append(SampleSource(character=ch_dir.name, lora_type=lora_type, images_dir=images_dir))
    return sources


def sample_reference_images(
    sources: List[SampleSource],
    per_character: int,
    seed: int,
) -> List[Tuple[str, str, Path]]:
    """
    Return list of (character, lora_type, image_path) samples.
    """
    rng = random.Random(seed)
    by_character: dict[str, List[Tuple[str, Path]]] = {}
    for s in sources:
        imgs = list(iter_pngs(s.images_dir))
        if not imgs:
            continue
        for img in imgs:
            by_character.setdefault(s.character, []).append((s.lora_type, img))

    sampled: List[Tuple[str, str, Path]] = []
    for character, items in sorted(by_character.items()):
        if len(items) <= per_character:
            for lora_type, img in items:
                sampled.append((character, lora_type, img))
            continue
        picks = rng.sample(items, per_character)
        for lora_type, img in picks:
            sampled.append((character, lora_type, img))
    return sampled


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dataset", type=Path, required=True, help="Existing Kohya dataset root.")
    ap.add_argument("--out-dataset", type=Path, required=True, help="New dataset root to create.")
    ap.add_argument(
        "--generated-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/generated_data"),
        help="Reference image root (exclude yokai-watch).",
    )
    ap.add_argument("--per-character", type=int, default=200, help="Reference samples per character.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use-symlinks", action="store_true", help="Symlink images instead of copying.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    base = args.base_dataset
    out = args.out_dataset
    if not base.exists():
        raise FileNotFoundError(base)
    ensure_dir(out)

    solo_in = base / "1_p3d_solo"
    pairs_in = base / "2_p3d_pairs"
    if not solo_in.exists() or not pairs_in.exists():
        raise FileNotFoundError("Expected base dataset to contain 1_p3d_solo and 2_p3d_pairs")

    solo_out = out / "1_p3d_solo"
    pairs_out = out / "2_p3d_pairs"
    style_out = out / "2_p3d_style_ref"
    ensure_dir(solo_out)
    ensure_dir(pairs_out)
    ensure_dir(style_out)

    # 1) Solo: link/copy images, copy captions (no rewrite)
    solo_pngs = list(iter_pngs(solo_in))
    logger.info("Solo files: %d images", len(solo_pngs))
    for png in solo_pngs:
        txt = png.with_suffix(".txt")
        link_or_copy(png, solo_out / png.name, use_symlinks=args.use_symlinks)
        if txt.exists():
            shutil.copy2(txt, solo_out / txt.name)

    # 2) Pairs: link/copy images, rewrite captions to remove compositing artifacts
    pair_pngs = list(iter_pngs(pairs_in))
    logger.info("Pair files: %d images", len(pair_pngs))
    for png in pair_pngs:
        txt = png.with_suffix(".txt")
        link_or_copy(png, pairs_out / png.name, use_symlinks=args.use_symlinks)
        if txt.exists():
            raw = txt.read_text(encoding="utf-8", errors="ignore")
            cleaned = clean_pair_caption(raw)
            (pairs_out / txt.name).write_text(cleaned, encoding="utf-8")

    # 3) Style reference: sample from generated_root and write simple captions
    sources = find_reference_sources(args.generated_root)
    logger.info("Reference sources found: %d", len(sources))
    samples = sample_reference_images(sources, per_character=args.per_character, seed=args.seed)
    logger.info("Reference samples: %d (per_character=%d)", len(samples), args.per_character)

    style_caption = "p3d_style, solo, a 3d animated character\n"
    for character, lora_type, img in samples:
        # Unique name to avoid collisions
        dst_stem = f"ref_{character}_{lora_type}_{img.stem}"
        dst_png = style_out / f"{dst_stem}.png"
        dst_txt = style_out / f"{dst_stem}.txt"
        shutil.copy2(img, dst_png)
        dst_txt.write_text(style_caption, encoding="utf-8")

    logger.info("Done. New dataset: %s", out)


if __name__ == "__main__":
    main()

