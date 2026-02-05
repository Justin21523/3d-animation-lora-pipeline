#!/usr/bin/env python3
"""
Discover character names from an existing generated_data folder.

Typical use:
  python scripts/batch/discover_characters_from_generated_data.py \
    --generated-data-root /mnt/data/ai_data/synthetic_lora_data/generated_data \
    --exclude yokai-watch \
    --format yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List, Sequence

import yaml


def _iter_character_dirs(generated_data_root: Path) -> Iterable[Path]:
    if not generated_data_root.exists():
        return
    for child in sorted(generated_data_root.iterdir()):
        if child.name.startswith("."):
            continue
        if child.is_dir():
            yield child


def _has_required_types(character_dir: Path, required_types: Sequence[str]) -> bool:
    if not required_types:
        return True
    required = set(required_types)
    present = {p.name for p in character_dir.iterdir() if p.is_dir()}
    return required.issubset(present)


def discover_characters(
    generated_data_root: Path,
    exclude: Sequence[str],
    required_types: Sequence[str],
) -> List[str]:
    exclude_set = {x.strip() for x in exclude if x.strip()}
    characters: List[str] = []
    for character_dir in _iter_character_dirs(generated_data_root):
        name = character_dir.name
        if name in exclude_set:
            continue
        if not _has_required_types(character_dir, required_types):
            continue
        characters.append(name)
    return characters


def main() -> int:
    parser = argparse.ArgumentParser(description="Discover characters from generated_data root.")
    parser.add_argument(
        "--generated-data-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/generated_data"),
        help="Path to generated_data/ containing per-character subdirectories.",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["yokai-watch"],
        help="Character directory names to exclude.",
    )
    parser.add_argument(
        "--require-types",
        nargs="*",
        default=[],
        help="Only include characters that have these lora_type subdirectories (e.g. pose action).",
    )
    parser.add_argument(
        "--format",
        choices=["yaml", "comma", "lines"],
        default="yaml",
        help="Output format.",
    )
    args = parser.parse_args()

    characters = discover_characters(
        generated_data_root=args.generated_data_root,
        exclude=args.exclude,
        required_types=args.require_types,
    )

    if args.format == "yaml":
        print(yaml.safe_dump({"characters": characters}, sort_keys=False).rstrip())
    elif args.format == "comma":
        print(",".join(characters))
    else:
        for name in characters:
            print(name)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

