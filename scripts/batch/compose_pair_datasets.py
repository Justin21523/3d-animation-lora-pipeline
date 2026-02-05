#!/usr/bin/env python3
"""
Generate per-pair interaction datasets for pair-specific LoRA training.

For each (A,B) character pair, generates a dataset directory:
  {out_root}/{A}__{B}/images/*.png
  {out_root}/{A}__{B}/captions/*.txt

Uses compose_interaction_dataset.py with --fixed-pair and caption-mode=pair.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_token_map(path: Optional[Path]) -> Optional[Path]:
    if not path:
        return None
    if not path.exists():
        raise SystemExit(f"char-token-map not found: {path}")
    # validate JSON
    _ = json.loads(path.read_text(encoding="utf-8"))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutout-root", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--num-images", type=int, default=320)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--trigger", default="pair_interaction")
    ap.add_argument("--min-cutouts-per-char", type=int, default=10)
    ap.add_argument("--canvas", type=int, default=1024)
    ap.add_argument("--body-height", type=float, default=0.62)
    ap.add_argument("--char-token-map", type=Path, default=None)
    ap.add_argument("--only", default=None, help="Optional subset: 'a,b;c,d' (order-insensitive)")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--python", default="python", help="Python executable to run composer")
    ap.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Number of concurrent pair jobs (each job is a separate Python process).",
    )
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    token_map_path = _load_token_map(args.char_token_map)

    characters = sorted([p.name for p in args.cutout_root.iterdir() if p.is_dir()])
    if len(characters) < 2:
        raise SystemExit("Need at least 2 character folders in cutout-root.")

    only_pairs = None
    if args.only:
        only_pairs = set()
        chunks = [c.strip() for c in args.only.split(";") if c.strip()]
        for ch in chunks:
            parts = [p.strip() for p in ch.split(",") if p.strip()]
            if len(parts) != 2:
                raise SystemExit("--only must be 'a,b;c,d'")
            a, b = sorted(parts)
            only_pairs.add((a, b))

    composer = PROJECT_ROOT / "scripts/generic/training/interaction/compose_interaction_dataset.py"
    args.out_root.mkdir(parents=True, exist_ok=True)

    jobs: List[tuple[str, str, Path, List[str]]] = []
    for a, b in combinations(characters, 2):
        pair = tuple(sorted((a, b)))
        if only_pairs is not None and pair not in only_pairs:
            continue

        out_dir = args.out_root / f"{pair[0]}__{pair[1]}"
        if args.skip_existing and (out_dir / "images").exists():
            LOGGER.info("Skip existing: %s", out_dir)
            continue

        cmd: List[str] = [
            str(args.python),
            str(composer),
            "--cutout-root",
            str(args.cutout_root),
            "--out-dir",
            str(out_dir),
            "--num-images",
            str(args.num_images),
            "--seed",
            str(args.seed),
            "--trigger",
            str(args.trigger),
            "--min-cutouts-per-char",
            str(args.min_cutouts_per_char),
            "--canvas",
            str(args.canvas),
            "--body-height",
            str(args.body_height),
            "--fixed-pair",
            f"{pair[0]},{pair[1]}",
            "--caption-mode",
            "pair",
            "--unique-cutout-combos",
        ]
        if token_map_path:
            cmd += ["--char-token-map", str(token_map_path)]

        jobs.append((pair[0], pair[1], out_dir, cmd))

    if not jobs:
        print(f"Done. Wrote 0 pair datasets under: {args.out_root}")
        return 0

    workers = max(1, int(args.workers))
    # Avoid accidental oversubscription on small machines
    cpu = os.cpu_count() or 1
    if workers > cpu * 2:
        LOGGER.warning("workers=%d is very high for cpu_count=%d; this may thrash disk/CPU.", workers, cpu)

    def _run_job(a: str, b: str, out_dir: Path, cmd: List[str]) -> tuple[str, str, Path]:
        subprocess.run(cmd, check=True)
        return (a, b, out_dir)

    total = 0
    LOGGER.info("Starting %d pair jobs with workers=%d", len(jobs), workers)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_run_job, a, b, out_dir, cmd) for (a, b, out_dir, cmd) in jobs]
        for fut in as_completed(futs):
            a, b, out_dir = fut.result()
            total += 1
            LOGGER.info("[%d/%d] Done pair %s,%s -> %s", total, len(jobs), a, b, out_dir)

    print(f"Done. Wrote {total} pair datasets under: {args.out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
