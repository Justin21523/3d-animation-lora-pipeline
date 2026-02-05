#!/usr/bin/env python3
"""
Prepare Trial5 pair-consistency dataset with simplified captions.

Rationale:
  - Trial4 LoRA appeared too weak and/or captions too long (token truncation),
    so the model didn't learn the intended "shared lighting / cohesive scene"
    constraints.
  - For Trial5 we rewrite captions to be short, consistent, and front-load the
    key tokens: p3d_A, p3d_B, p3d_style, two characters...

Inputs:
  - Curated pairs directory (per-pair subdirs, each containing .png + .txt)
  - Tokenized solo directory (flat Kohya subset with .png + .txt, first token is p3d_<char>)

Output (Kohya-style):
  out_dir/
    1_p3d_solo_anchor/
      <token>__<src_stem>.png (symlink)
      <token>__<src_stem>.txt (rewritten)
    2_p3d_pairs_curated/
      <pair>__<src_stem>.png (symlink)
      <pair>__<src_stem>.txt (rewritten)
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from datetime import datetime
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


INTERACTIONS = [
    "walking side by side, talking",
    "laughing together",
    "high-five, hands touching",
    "handshake, eye contact",
    "hugging, close contact",
    "passing a small object hand-to-hand, hands touching",
    "sitting at a table facing each other, talking",
    "running together, synchronized motion",
    "dancing together, coordinated pose",
    "back-to-back pose, confident",
]


def read_first_token(txt: Path) -> str:
    s = txt.read_text(encoding="utf-8").strip()
    if not s:
        return ""
    return s.split(",")[0].strip()


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.symlink_to(src)


def write_text_if_missing(dst: Path, text: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    dst.write_text(text, encoding="utf-8")


def iter_pair_pngs(pairs_root: Path) -> Iterable[Tuple[str, Path]]:
    for pair_dir in sorted([p for p in pairs_root.iterdir() if p.is_dir()]):
        for png in sorted(pair_dir.glob("*.png")):
            yield pair_dir.name, png


def collect_solo_by_token(solo_dir: Path) -> Dict[str, List[Path]]:
    mapping: Dict[str, List[Path]] = {}
    for txt in solo_dir.glob("*.txt"):
        token = read_first_token(txt)
        if not token:
            continue
        png = txt.with_suffix(".png")
        if not png.exists():
            continue
        mapping.setdefault(token, []).append(png)
    return mapping


def rewrite_pair_caption(a: str, b: str, interaction: str) -> str:
    # Keep short to avoid CLIP truncation. Put constraints early.
    return (
        f"{a}, {b}, p3d_style, two characters, {interaction}, "
        "both characters visible, consistent lighting, consistent shadows, "
        "same material response, high quality 3d render"
    )


def rewrite_solo_caption(token: str) -> str:
    return f"{token}, p3d_style, solo, high quality 3d render"


@dataclass(frozen=True)
class Summary:
    out_dir: str
    pairs: int
    solos: int


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-root", required=True, help="Curated pairs root (per-pair subdirs).")
    ap.add_argument(
        "--solo-dir",
        default="/mnt/data/ai_data/synthetic_lora_data/kohya_dataset_p3d_multichar_20260129/1_p3d_solo",
        help="Tokenized solo directory (flat .png/.txt).",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output dataset directory (default: /mnt/data/.../kohya_dataset_p3d_pairconsistency_trial5_<ts>).",
    )
    ap.add_argument("--anchors-per-character", type=int, default=150)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--characters-file",
        default="",
        help="Optional txt file listing p3d_* tokens (one per line).",
    )
    args = ap.parse_args()

    pairs_root = Path(args.pairs_root)
    if not pairs_root.exists():
        raise FileNotFoundError(pairs_root)

    solo_dir = Path(args.solo_dir)
    if not solo_dir.exists():
        raise FileNotFoundError(solo_dir)

    if args.characters_file:
        toks: List[str] = []
        for line in Path(args.characters_file).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            toks.append(s)
        char_tokens = toks
    else:
        char_tokens = CHAR_TOKENS_DEFAULT

    rng = random.Random(args.seed)

    default_root = Path("/mnt/data/ai_data/synthetic_lora_data")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else default_root / f"kohya_dataset_p3d_pairconsistency_trial5_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    pair_out = out_dir / PAIR_SUBSET
    solo_out = out_dir / SOLO_SUBSET
    pair_out.mkdir(parents=True, exist_ok=True)
    solo_out.mkdir(parents=True, exist_ok=True)

    # 1) Pairs: symlink images, rewrite captions.
    pair_count = 0
    for pair_name, png in iter_pair_pngs(pairs_root):
        txt = png.with_suffix(".txt")
        if not txt.exists():
            raise FileNotFoundError(txt)

        # pair_name is expected "p3d_a__p3d_b"
        if "__" not in pair_name:
            raise ValueError(f"Unexpected pair dir name: {pair_name}")
        a, b = pair_name.split("__", 1)

        interaction = INTERACTIONS[(hash((pair_name, png.stem)) & 0x7FFFFFFF) % len(INTERACTIONS)]
        new_caption = rewrite_pair_caption(a, b, interaction=interaction)

        stem = f"{pair_name}__{png.stem}"
        dst_png = pair_out / f"{stem}.png"
        dst_txt = pair_out / f"{stem}.txt"
        safe_symlink(png, dst_png)
        write_text_if_missing(dst_txt, new_caption + "\n")
        pair_count += 1

    # 2) Solo anchors: sample per character token; rewrite captions.
    solo_by_token = collect_solo_by_token(solo_dir)
    missing = [t for t in char_tokens if t not in solo_by_token]
    if missing:
        raise RuntimeError(f"Missing solo anchors for tokens: {missing}")

    solo_count = 0
    for token in char_tokens:
        candidates = list(solo_by_token[token])
        rng.shuffle(candidates)
        chosen = candidates[: args.anchors_per_character]
        if not chosen:
            raise RuntimeError(f"No solo candidates for {token}")

        for png in chosen:
            stem = f"{token}__{png.stem}"
            dst_png = solo_out / f"{stem}.png"
            dst_txt = solo_out / f"{stem}.txt"
            safe_symlink(png, dst_png)
            write_text_if_missing(dst_txt, rewrite_solo_caption(token) + "\n")
            solo_count += 1

    print(out_dir)
    print(f"pairs={pair_count}")
    print(f"solos={solo_count}")


if __name__ == "__main__":
    main()

