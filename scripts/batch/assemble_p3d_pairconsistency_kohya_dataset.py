#!/usr/bin/env python3
"""
Assemble a Kohya dataset for "pair-consistency" fine-tuning.

Inputs:
  - Curated pair dataset (nested by pair directory, with .png + .txt per image)
  - Existing tokenized solo dataset (flat Kohya subset dir with .png + .txt)

Output (Kohya-style):
  out_dir/
    1_p3d_solo_anchor/
      <unique>.png
      <unique>.txt
    2_p3d_pairs_curated/
      <unique>.png
      <unique>.txt

By default, uses symlinks to avoid copying large files.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


PAIR_SUBSET = "2_p3d_pairs_curated"
SOLO_SUBSET = "1_p3d_solo_anchor"


CHAR_TOKENS_DEFAULT = [
    "p3d_luca",
    "p3d_giulia",
    "p3d_alberto",
    "p3d_alberto_seamonster",
    "p3d_luca_seamonster",
    "p3d_miguel",
    "p3d_ian_lightfoot",
    "p3d_barley_lightfoot",
    "p3d_elio",
    "p3d_bryce",
    "p3d_caleb",
    "p3d_orion",
    "p3d_russell",
    "p3d_tyler",
]


def _iter_pair_images(pairs_root: Path) -> Iterable[Tuple[str, Path]]:
    for pair_dir in sorted([p for p in pairs_root.iterdir() if p.is_dir()]):
        for png in sorted(pair_dir.glob("*.png")):
            yield pair_dir.name, png


def _parse_first_token(caption_path: Path) -> str:
    s = caption_path.read_text(encoding="utf-8").strip()
    if not s:
        return ""
    return s.split(",")[0].strip()


def _collect_solo_by_token(solo_dir: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for txt in solo_dir.glob("*.txt"):
        token = _parse_first_token(txt)
        if not token:
            continue
        png = txt.with_suffix(".png")
        if not png.exists():
            continue
        mapping.setdefault(token, []).append(png)
    return mapping


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.symlink_to(src)


def _safe_copy_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")


@dataclass(frozen=True)
class Counts:
    pairs: int
    solos: int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-root", required=True, help="Curated pairs root (contains per-pair subdirs).")
    ap.add_argument(
        "--solo-dir",
        default="/mnt/data/ai_data/synthetic_lora_data/kohya_dataset_p3d_multichar_20260129/1_p3d_solo",
        help="Tokenized solo directory (flat .png/.txt).",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output dataset directory (default: /mnt/data/.../kohya_dataset_p3d_pairconsistency_<ts>).",
    )
    ap.add_argument("--anchors-per-character", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-symlinks", action="store_true", help="Copy images instead of symlinking.")
    args = ap.parse_args()

    pairs_root = Path(args.pairs_root)
    if not pairs_root.exists():
        raise FileNotFoundError(pairs_root)

    solo_dir = Path(args.solo_dir)
    if not solo_dir.exists():
        raise FileNotFoundError(solo_dir)

    default_root = Path("/mnt/data/ai_data/synthetic_lora_data")
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        import datetime as _dt

        out_dir = default_root / f"kohya_dataset_p3d_pairconsistency_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    pair_out = out_dir / PAIR_SUBSET
    solo_out = out_dir / SOLO_SUBSET
    pair_out.mkdir(parents=True, exist_ok=True)
    solo_out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)

    # 1) Copy/symlink curated pairs (flatten)
    pair_count = 0
    for pair_name, png in _iter_pair_images(pairs_root):
        txt = png.with_suffix(".txt")
        if not txt.exists():
            raise FileNotFoundError(txt)

        stem = f"{pair_name}__{png.stem}"
        dst_png = pair_out / f"{stem}.png"
        dst_txt = pair_out / f"{stem}.txt"

        if args.no_symlinks:
            if not dst_png.exists():
                dst_png.write_bytes(png.read_bytes())
        else:
            _safe_symlink(png, dst_png)
        _safe_copy_text(txt, dst_txt)
        pair_count += 1

    # 2) Sample solo anchors per token
    solo_by_token = _collect_solo_by_token(solo_dir)

    missing = [t for t in CHAR_TOKENS_DEFAULT if t not in solo_by_token]
    if missing:
        raise RuntimeError(f"Missing solo anchors for tokens: {missing}")

    solo_count = 0
    for token in CHAR_TOKENS_DEFAULT:
        candidates = list(solo_by_token[token])
        if not candidates:
            raise RuntimeError(f"No solo candidates found for {token}")
        rng.shuffle(candidates)
        chosen = candidates[: args.anchors_per_character]

        for i, png in enumerate(chosen):
            txt = png.with_suffix(".txt")
            stem = f"{token}__{png.stem}"
            dst_png = solo_out / f"{stem}.png"
            dst_txt = solo_out / f"{stem}.txt"

            if args.no_symlinks:
                if not dst_png.exists():
                    dst_png.write_bytes(png.read_bytes())
            else:
                _safe_symlink(png, dst_png)
            _safe_copy_text(txt, dst_txt)
            solo_count += 1

    # Summary
    out_dir.mkdir(parents=True, exist_ok=True)
    print(out_dir)
    print(f"pairs={pair_count}")
    print(f"solos={solo_count}")


if __name__ == "__main__":
    main()

