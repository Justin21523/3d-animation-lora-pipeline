#!/usr/bin/env python3
"""
Convert pair-style interaction captions to multi-character checkpoint captions.

Why:
  compose_interaction_dataset.py (caption-mode=pair) writes captions like:
    p3d_style, pair_p3d_A_p3d_B, left_p3d_A, right_p3d_B, two 3d animated characters, ...

For multi-character SDXL checkpoint training, we want the base identity tokens to be
present directly:
  p3d_style, p3d_A, p3d_B, two 3d animated characters, ...

This script rewrites each caption by:
  - extracting A/B from left_/right_ tokens
  - removing pair_/left_/right_ tokens
  - inserting A/B tokens right after the style trigger

It does NOT touch images.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple


def _split_caption(text: str) -> List[str]:
    # Captions are written as ", ".join(parts)
    parts = [p.strip() for p in text.strip().split(",") if p.strip()]
    return parts


def _extract_left_right_tokens(parts: List[str]) -> Tuple[str, str]:
    left = ""
    right = ""
    for p in parts:
        if p.startswith("left_") and len(p) > len("left_"):
            left = p[len("left_") :].strip()
        elif p.startswith("right_") and len(p) > len("right_"):
            right = p[len("right_") :].strip()
    if not left or not right:
        raise ValueError("Missing left_/right_ tokens in caption")
    return left, right


def _rewrite(parts: List[str]) -> List[str]:
    if not parts:
        raise ValueError("Empty caption")

    style = parts[0]

    left_tok = ""
    right_tok = ""
    start_idx = 1

    # Case 1: legacy pair captions include left_/right_ tokens
    if any(p.startswith("left_") or p.startswith("right_") for p in parts):
        left_tok, right_tok = _extract_left_right_tokens(parts)
    # Case 2: already converted captions look like:
    #   p3d_style, p3d_A, p3d_B, ...
    elif style == "p3d_style" and len(parts) >= 3 and parts[1].startswith("p3d_") and parts[2].startswith("p3d_"):
        left_tok, right_tok = parts[1], parts[2]
        start_idx = 3
    else:
        raise ValueError("Unsupported caption format")

    filtered: List[str] = []
    for p in parts[start_idx:]:
        if p.startswith("pair_"):
            continue
        if p.startswith("left_"):
            continue
        if p.startswith("right_"):
            continue
        # Normalize count token to keep it stable for keep_tokens
        if p == "two 3d animated characters":
            filtered.append("two characters")
            continue
        filtered.append(p)

    # Multi-character SDXL checkpoint rule: identity tokens should come first.
    # Recommended order: p3d_A, p3d_B, p3d_style, two characters, ...
    return [left_tok, right_tok, style, *filtered]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captions-dir", type=Path, required=True)
    ap.add_argument("--ext", default=".txt")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    args = ap.parse_args()

    captions = sorted(args.captions_dir.glob(f"*{args.ext}"))
    if args.limit and args.limit > 0:
        captions = captions[: args.limit]

    changed = 0
    skipped = 0

    for p in captions:
        raw = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            skipped += 1
            continue

        parts = _split_caption(raw)
        try:
            new_parts = _rewrite(parts)
        except Exception:
            skipped += 1
            continue

        new_text = ", ".join(new_parts).strip() + "\n"
        if new_text == raw + "\n":
            continue

        changed += 1
        if not args.dry_run:
            p.write_text(new_text, encoding="utf-8")

    print(f"Processed {len(captions)} captions in {args.captions_dir}")
    print(f"  changed={changed} skipped={skipped} dry_run={bool(args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
