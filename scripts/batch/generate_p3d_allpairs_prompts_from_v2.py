#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PromptTemplate:
    index: int
    body: str  # begins right after "p3d_A, p3d_B, p3d_style, two characters, "


def read_nonempty_lines(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def parse_pair_list(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line in read_nonempty_lines(path):
        parts = [p.strip() for p in line.split(",")]
        tokens = [p for p in parts if p.startswith("p3d_") and p != "p3d_style"]
        if len(tokens) < 2:
            continue
        pairs.append((tokens[0], tokens[1]))
    return pairs


def build_templates(v2_path: Path) -> list[PromptTemplate]:
    src = read_nonempty_lines(v2_path)
    if len(src) != 50:
        raise SystemExit(f"Expected 50 prompts in v2 file, got {len(src)}: {v2_path}")

    templates: list[PromptTemplate] = []
    header_re = re.compile(r"^(p3d_[^,]+),\s*(p3d_[^,]+),\s*p3d_style,\s*two characters,\s*")
    for idx, line in enumerate(src, start=1):
        m = header_re.match(line)
        if not m:
            raise SystemExit(f"Line {idx} does not start with expected header: {line[:120]}")
        a_tok = m.group(1).strip()
        b_tok = m.group(2).strip()
        body = line[m.end() :]

        # The v2 prompts were authored with explicit character names (Luca/Miguel/Orion).
        # Convert those names to role placeholders based on the ORIGINAL pair tokens
        # of each line, so that we can safely re-target the same motion template to
        # any other pair.
        name_to_token = {
            "Luca": "p3d_luca",
            "Miguel": "p3d_miguel",
            "Orion": "p3d_orion",
        }
        for name, token in name_to_token.items():
            role: str | None = None
            if a_tok == token:
                role = "Character A"
            elif b_tok == token:
                role = "Character B"
            if role is not None:
                body = re.sub(rf"\b{name}\b", role, body)

        templates.append(PromptTemplate(index=idx, body=body))
    return templates


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate all-pairs prompt matrix from the 50 v2 video prompts.")
    ap.add_argument(
        "--v2-prompts",
        default="prompts/p3d/p3d_luca_orion_miguel_pair_prompts_50_v2_video.txt",
        help="Source v2 prompt file (50 lines).",
    )
    ap.add_argument(
        "--pairs",
        default="prompts/p3d/p3d_multichar_all_pairs_prompts.txt",
        help="Pair list file (14 chars -> 91 pairs).",
    )
    ap.add_argument(
        "--out-dir",
        default="prompts/p3d/pairs_allchars_50actions_v1",
        help="Output directory (per-pair .txt files + index).",
    )
    ap.add_argument(
        "--out-all",
        default="prompts/p3d/p3d_multichar_all_pairs_prompts_50actions_v1_video.txt",
        help="Output single-file containing all prompts (one per line).",
    )
    args = ap.parse_args()

    v2_path = Path(args.v2_prompts)
    pairs_path = Path(args.pairs)
    out_dir = Path(args.out_dir)
    out_all = Path(args.out_all)

    templates = build_templates(v2_path)
    pairs = parse_pair_list(pairs_path)
    if len(pairs) != 91:
        raise SystemExit(f"Expected 91 pairs, got {len(pairs)} from {pairs_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    all_lines: list[str] = []
    index_lines: list[str] = ["pair_id\tpair_a\tpair_b\taction_id\tprompt_file\tprompt"]

    for pair_id, (a, b) in enumerate(pairs, start=1):
        pair_stem = f"{a}__{b}"
        pair_file = out_dir / f"{pair_stem}.txt"
        pair_lines: list[str] = []
        for tmpl in templates:
            line = f"{a}, {b}, p3d_style, two characters, {tmpl.body}"
            pair_lines.append(line)
            all_lines.append(line)
            index_lines.append(
                f"{pair_id}\t{a}\t{b}\t{tmpl.index}\t{pair_file.as_posix()}\t{line}"
            )
        pair_file.write_text("\n".join(pair_lines) + "\n", encoding="utf-8")

    out_all.write_text("\n".join(all_lines) + "\n", encoding="utf-8")
    (out_dir / "index.tsv").write_text("\n".join(index_lines) + "\n", encoding="utf-8")

    print(f"Wrote per-pair files: {out_dir} (count={len(pairs)})")
    print(f"Wrote all prompts: {out_all} (lines={len(all_lines)})")
    print(f"Wrote index: {out_dir / 'index.tsv'}")


if __name__ == "__main__":
    main()
