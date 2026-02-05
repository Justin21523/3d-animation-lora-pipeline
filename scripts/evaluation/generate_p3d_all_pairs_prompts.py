#!/usr/bin/env python3
"""
Generate an exhaustive prompt list for all P3D character pairs.

Outputs a .txt (one prompt per line, comment headers allowed) suitable for:
  python scripts/evaluation/p3d_multichar_checkpoint_smoke_test.py --prompts <file>
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import List


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
    "walking side by side, casual conversation",
    "laughing together",
    "high-five, hands touching",
    "handshake",
    "hugging, close contact, partial occlusion",
    "sitting at a table facing each other",
]


def build_prompts(char_tokens: List[str]) -> List[str]:
    prompts: List[str] = []
    for i, (a, b) in enumerate(combinations(char_tokens, 2), start=1):
        interaction = INTERACTIONS[(i - 1) % len(INTERACTIONS)]
        prompts.append(
            f"{a}, {b}, p3d_style, two characters, {interaction}, "
            "both characters visible, natural interaction, consistent lighting on both characters, "
            "medium wide shot, high quality 3d render"
        )
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="prompts/p3d/p3d_multichar_all_pairs_prompts.txt",
        help="Output prompt file (one prompt per line).",
    )
    ap.add_argument(
        "--characters",
        default="",
        help="Optional path to a text file containing p3d_* character tokens (one per line).",
    )
    args = ap.parse_args()

    if args.characters:
        char_path = Path(args.characters)
        char_tokens = []
        for line in char_path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            char_tokens.append(s)
    else:
        char_tokens = CHAR_TOKENS_DEFAULT

    if len(char_tokens) < 2:
        raise ValueError("Need at least 2 character tokens")

    prompts = build_prompts(char_tokens)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "# Auto-generated: exhaustive P3D pair prompts",
        f"# num_characters={len(char_tokens)} num_pairs={len(prompts)}",
        "# Format: p3d_A, p3d_B, p3d_style, two characters, ...",
        "",
    ]
    out_path.write_text("\n".join(header + prompts) + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

