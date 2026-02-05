#!/usr/bin/env python3
"""
Generate per-pair test prompts for pair-interaction LoRAs.

Writes:
  prompts/lora_testing/pairs/<A>__<B>.txt

Each prompt uses the training tokens:
  pair_<A>_<B>, left_<A>, right_<B>, pair_interaction
and varies the scene/background to verify generalization at low LoRA weights.

Important:
- Identity LoRAs are global (not region-masked). To reduce "identity blending",
  the evaluation prompts explicitly include each character trigger token (A and B)
  in addition to left/right tokens.
- For known "form" variants (e.g., *_seamonster), we also add explicit
  disambiguation phrases like "left character in human form" vs "right character
  in sea monster form".
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Tuple


SCENES = [
    "city street",
    "living room",
    "classroom",
    "park",
    "beach sunset",
    "forest trail",
    "night city",
    "supermarket aisle",
    "cafe interior",
    "train station",
    "snowy street",
    "studio background",
]

INTERACTIONS = [
    "standing side by side, friendly conversation",
    "walking together, relaxed mood",
    "facing each other, talking face to face",
    "high five gesture, celebratory moment",
    "handshake, friendly greeting",
    "both pointing in the same direction",
    "sitting together, casual hangout",
    "one character waving, the other smiling",
]


def _form_hint(name: str) -> str:
    n = name.lower()
    if "seamonster" in n or "sea_monster" in n:
        return "sea monster form"
    return "human form"


def _iter_pair_dirs(pairs_root: Path) -> Iterable[str]:
    for d in sorted([p for p in pairs_root.iterdir() if p.is_dir()]):
        if "__" in d.name:
            yield d.name


def _make_prompts(pair_dir_name: str) -> List[str]:
    a, b = pair_dir_name.split("__", 1)
    pair_token = f"pair_{a}_{b}"
    left = f"left_{a}"
    right = f"right_{b}"
    left_form = _form_hint(a)
    right_form = _form_hint(b)

    prompts: List[str] = []
    # Deterministic pairing: cycle interactions and scenes
    for i in range(min(len(SCENES), 12)):
        scene = SCENES[i]
        interaction = INTERACTIONS[i % len(INTERACTIONS)]
        prompts.append(
            ", ".join(
                [
                    # Identity triggers (for BEST identity LoRAs)
                    a,
                    b,
                    pair_token,
                    left,
                    right,
                    "pair_interaction",
                    "two different characters",
                    "distinct faces and distinct outfits",
                    f"left in {left_form}",
                    f"right in {right_form}",
                    interaction,
                    "full body",
                    scene,
                    "pixar style",
                ]
            )
        )
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-root", type=Path, required=True, help="Root containing <A>__<B>/images + captions")
    ap.add_argument(
        "--out-root",
        type=Path,
        default=Path("prompts/lora_testing/pairs"),
        help="Output directory for per-pair prompt files",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    written = 0

    for pair in _iter_pair_dirs(args.pairs_root):
        out_path = args.out_root / f"{pair}.txt"
        if out_path.exists() and not args.overwrite:
            continue
        prompts = _make_prompts(pair)
        out_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        written += 1

    print(f"Wrote {written} prompt files to: {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
