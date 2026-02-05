#!/usr/bin/env python3
"""
Quick, dependency-light image stats for comparing datasets.

Computes simple luminance/contrast/saturation statistics over a random sample.
This is useful for catching brightness drift (e.g., "too bright / too glossy").
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def iter_images(root: Path, exclude_dirnames: Sequence[str]) -> Iterable[Path]:
    exclude_set = set(exclude_dirnames)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        dirnames[:] = [d for d in dirnames if d not in exclude_set]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in IMG_EXTS:
                yield p


def sample_paths(paths: Sequence[Path], n: int, seed: int) -> List[Path]:
    if n <= 0 or n >= len(paths):
        return list(paths)
    rng = random.Random(seed)
    return rng.sample(list(paths), k=n)


def rgb_to_stats(arr_u8: np.ndarray) -> Tuple[float, float, float]:
    """
    Args:
        arr_u8: HxWx3 uint8
    Returns:
        (luma_mean_0_1, luma_std_0_1, sat_mean_0_1)
    """
    arr = arr_u8.astype(np.float32) / 255.0
    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]
    # Rec.709 luma
    y = 0.2126 * r + 0.7152 * g + 0.0722 * b
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    # HSV saturation proxy: (max-min)/max, stable for quick comparisons
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    sat_mean = float(np.mean(sat))
    return y_mean, y_std, sat_mean


@dataclass(frozen=True)
class Stats:
    num_images: int
    luma_mean: float
    luma_std: float
    sat_mean: float


def compute_stats(paths: Sequence[Path]) -> Stats:
    luma_means: List[float] = []
    luma_stds: List[float] = []
    sat_means: List[float] = []

    for p in paths:
        im = Image.open(p).convert("RGB")
        arr = np.array(im, dtype=np.uint8)
        y_mean, y_std, sat_mean = rgb_to_stats(arr)
        luma_means.append(y_mean)
        luma_stds.append(y_std)
        sat_means.append(sat_mean)

    return Stats(
        num_images=len(paths),
        luma_mean=float(np.mean(luma_means)) if luma_means else 0.0,
        luma_std=float(np.mean(luma_stds)) if luma_stds else 0.0,
        sat_mean=float(np.mean(sat_means)) if sat_means else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Dataset A root directory (recursive).")
    ap.add_argument("--b", required=True, help="Dataset B root directory (recursive).")
    ap.add_argument("--sample", type=int, default=500, help="Random sample size per dataset (0=all).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--exclude-dirname",
        action="append",
        default=[],
        help="Directory name to exclude (can be repeated).",
    )
    ap.add_argument("--out", default="", help="Write JSON report to this path.")
    args = ap.parse_args()

    a_root = Path(args.a)
    b_root = Path(args.b)
    if not a_root.exists():
        raise FileNotFoundError(a_root)
    if not b_root.exists():
        raise FileNotFoundError(b_root)

    a_all = list(iter_images(a_root, exclude_dirnames=args.exclude_dirname))
    b_all = list(iter_images(b_root, exclude_dirnames=args.exclude_dirname))
    if not a_all:
        raise ValueError(f"No images found under: {a_root}")
    if not b_all:
        raise ValueError(f"No images found under: {b_root}")

    a_paths = sample_paths(a_all, n=args.sample, seed=args.seed)
    b_paths = sample_paths(b_all, n=args.sample, seed=args.seed + 1)

    a_stats = compute_stats(a_paths)
    b_stats = compute_stats(b_paths)

    report = {
        "a_root": str(a_root),
        "b_root": str(b_root),
        "sample": args.sample,
        "seed": args.seed,
        "a": asdict(a_stats),
        "b": asdict(b_stats),
        "delta": {
            "luma_mean": a_stats.luma_mean - b_stats.luma_mean,
            "luma_std": a_stats.luma_std - b_stats.luma_std,
            "sat_mean": a_stats.sat_mean - b_stats.sat_mean,
        },
    }

    out_text = json.dumps(report, indent=2)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text, encoding="utf-8")
    print(out_text)


if __name__ == "__main__":
    main()
