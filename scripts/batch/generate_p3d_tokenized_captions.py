#!/usr/bin/env python3
"""
Generate tokenized (p3d) Kohya-ready captions for synthetic single-character images.

This converts `prompts.json` into identity-removed captions using
`PromptToCaptionConverter`, then writes a `.txt` caption next to each `.png`
under the `images/` folder.

Resulting caption format (solo):
  p3d_<character>, p3d_style, solo, <identity-removed caption...>

Designed for the P3D multi-character SDXL checkpoint workflow:
  - docs/3d-training/SDXL_MULTI_CHARACTER_CHECKPOINT.md
  - configs/training/p3d_token_map_14chars.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import sys

# Ensure repo root is importable (so `scripts.*` imports work when executed via conda run)
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.generic.training.caption_engines.prompt_to_caption_converter import (
    PromptToCaptionConverter,
)

logger = logging.getLogger(__name__)


IMAGE_RE = re.compile(r"^prompt_(\d{4})_img_(\d{2}).*\.png$", re.IGNORECASE)


@dataclass(frozen=True)
class JobTarget:
    root: Path
    character: str
    lora_type: str

    @property
    def prompts_path(self) -> Path:
        return self.root / self.character / self.lora_type / "prompts.json"

    @property
    def images_dir(self) -> Path:
        return self.root / self.character / self.lora_type / "images"


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _iter_targets(root: Path) -> Iterable[JobTarget]:
    if not root.exists():
        return
    for character_dir in sorted([p for p in root.iterdir() if p.is_dir()]):
        character = character_dir.name
        for lora_type_dir in sorted([p for p in character_dir.iterdir() if p.is_dir()]):
            lora_type = lora_type_dir.name
            yield JobTarget(root=root, character=character, lora_type=lora_type)


def _index_images(images_dir: Path) -> Dict[int, List[Path]]:
    """
    Return {prompt_idx: [image_paths...]} from filenames like:
      prompt_0000_img_00_*.png
    """
    index: Dict[int, List[Path]] = {}
    if not images_dir.exists():
        return index

    for img in sorted(images_dir.glob("*.png")):
        m = IMAGE_RE.match(img.name)
        if not m:
            continue
        prompt_idx = int(m.group(1))
        index.setdefault(prompt_idx, []).append(img)
    return index


def _build_prefix(character: str, token_map: Dict[str, str], style_token: str, solo_token: str) -> str:
    if character not in token_map:
        raise KeyError(f"Missing character in token map: {character}")
    return f"{token_map[character]}, {style_token}, {solo_token}"


def _normalize_caption(text: str) -> str:
    text = (text or "").strip()
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,")


def _write_caption_file(path: Path, text: str, overwrite: bool) -> bool:
    if path.exists() and not overwrite:
        return False
    path.write_text(text + "\n", encoding="utf-8")
    return True


def generate_for_target(
    target: JobTarget,
    token_map: Dict[str, str],
    style_token: str,
    solo_token: str,
    overwrite: bool,
    min_tokens: int,
    max_tokens: int,
    target_tokens: int,
) -> Tuple[int, int]:
    """
    Returns (written_count, skipped_count).
    """
    prompts_path = target.prompts_path
    images_dir = target.images_dir

    if not prompts_path.exists():
        logger.warning("Missing prompts.json: %s", prompts_path)
        return (0, 0)
    if not images_dir.exists():
        logger.warning("Missing images dir: %s", images_dir)
        return (0, 0)

    data = _load_json(prompts_path)
    prompts = data.get("prompts", [])
    if not isinstance(prompts, list) or not prompts:
        logger.warning("No prompts found in: %s", prompts_path)
        return (0, 0)

    images_by_prompt = _index_images(images_dir)
    if not images_by_prompt:
        logger.warning("No images found in: %s", images_dir)
        return (0, 0)

    converter = PromptToCaptionConverter(
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        target_tokens=target_tokens,
        generic_subject="a 3d animated character",
    )

    prefix = _build_prefix(target.character, token_map, style_token, solo_token)

    written = 0
    skipped = 0

    for prompt_idx, images in images_by_prompt.items():
        if prompt_idx >= len(prompts):
            logger.warning(
                "Prompt index out of range: %s (%s/%s) prompt_idx=%d len(prompts)=%d",
                target.root,
                target.character,
                target.lora_type,
                prompt_idx,
                len(prompts),
            )
            continue

        raw_prompt = prompts[prompt_idx].get("prompt", "")
        converted = converter.convert(raw_prompt, character_name=target.character)
        caption_body = _normalize_caption(converted.caption if converted and converted.caption else "")
        full_caption = _normalize_caption(f"{prefix}, {caption_body}")

        for img_path in images:
            caption_path = img_path.with_suffix(".txt")
            did_write = _write_caption_file(caption_path, full_caption, overwrite=overwrite)
            if did_write:
                written += 1
            else:
                skipped += 1

    return (written, skipped)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate tokenized p3d captions next to synthetic images.")
    parser.add_argument(
        "--roots",
        nargs="+",
        required=True,
        help="One or more synthetic generated_data roots (each contains {character}/{lora_type}/prompts.json and images/).",
    )
    parser.add_argument(
        "--token-map",
        required=True,
        help="JSON map: {character_dir_name: p3d_token} (e.g. configs/training/p3d_token_map_14chars.json).",
    )
    parser.add_argument("--style-token", default="p3d_style")
    parser.add_argument("--solo-token", default="solo")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing .txt captions.")
    parser.add_argument("--min-tokens", type=int, default=60)
    parser.add_argument("--max-tokens", type=int, default=225)
    parser.add_argument("--target-tokens", type=int, default=100)
    parser.add_argument("--log-level", default="INFO")

    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    token_map_path = Path(args.token_map)
    token_map = _load_json(token_map_path)
    if not isinstance(token_map, dict):
        raise ValueError(f"token-map must be a JSON object: {token_map_path}")

    total_written = 0
    total_skipped = 0

    for root_str in args.roots:
        root = Path(root_str)
        logger.info("Processing root: %s", root)
        for target in _iter_targets(root):
            # Only handle expected lora types for solo data
            if target.lora_type not in {"action", "expression", "pose"}:
                continue
            written, skipped = generate_for_target(
                target=target,
                token_map=token_map,
                style_token=args.style_token,
                solo_token=args.solo_token,
                overwrite=args.overwrite,
                min_tokens=args.min_tokens,
                max_tokens=args.max_tokens,
                target_tokens=args.target_tokens,
            )
            total_written += written
            total_skipped += skipped
            logger.info(
                "  %s/%s: wrote=%d skipped=%d",
                target.character,
                target.lora_type,
                written,
                skipped,
            )

    logger.info("Done. Total captions: wrote=%d skipped=%d", total_written, total_skipped)


if __name__ == "__main__":
    main()
