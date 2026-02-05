#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def read_prompts(path: Path) -> list[str]:
    prompts: list[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser(description="Export one-prompt-per-file captions for batch tools.")
    ap.add_argument("--prompts", required=True, help="Prompt list file (one prompt per line).")
    ap.add_argument("--out-dir", required=True, help="Output directory for .txt caption files.")
    ap.add_argument("--prefix", default="prompt", help="Filename prefix.")
    ap.add_argument("--start", type=int, default=1, help="Start index.")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = ap.parse_args()

    prompts_path = Path(args.prompts)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(prompts_path)
    if not prompts:
        raise SystemExit(f"No prompts found in {prompts_path}")

    start = max(1, int(args.start))
    for offset, prompt in enumerate(prompts):
        idx = start + offset
        out_path = out_dir / f"{args.prefix}_{idx:05d}.txt"
        if out_path.exists() and not args.overwrite:
            continue
        out_path.write_text(prompt + "\n", encoding="utf-8")

    print(f"Wrote {len(prompts)} captions to: {out_dir}")


if __name__ == "__main__":
    main()

