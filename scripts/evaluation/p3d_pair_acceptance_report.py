#!/usr/bin/env python3
"""
Compute lightweight QA stats over a batch-generated image directory.

This is NOT a semantic "two people" detector. It provides practical flags that
correlate with common failures:
  - excessive blur
  - very low contrast (flat / washed)
  - overexposure / underexposure
  - very high noise proxy (optional)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image


IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def iter_images(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in IMG_EXTS:
                yield p


def rgb_to_luma(arr: np.ndarray) -> np.ndarray:
    arr_f = arr.astype(np.float32) / 255.0
    r = arr_f[:, :, 0]
    g = arr_f[:, :, 1]
    b = arr_f[:, :, 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def saturation_proxy(arr: np.ndarray) -> float:
    arr_f = arr.astype(np.float32) / 255.0
    mx = np.max(arr_f, axis=2)
    mn = np.min(arr_f, axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    return float(np.mean(sat))


def laplacian_var(gray: np.ndarray) -> float:
    # simple 2D Laplacian kernel
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    # naive conv (small, OK for QA)
    g = gray.astype(np.float32)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    acc = np.zeros((h - 2, w - 2), dtype=np.float32)
    acc += k[0, 1] * g[0 : h - 2, 1 : w - 1]
    acc += k[1, 0] * g[1 : h - 1, 0 : w - 2]
    acc += k[1, 1] * g[1 : h - 1, 1 : w - 1]
    acc += k[1, 2] * g[1 : h - 1, 2:w]
    acc += k[2, 1] * g[2:h, 1 : w - 1]
    return float(np.var(acc))


@dataclass(frozen=True)
class Row:
    index: int
    image_path: str
    prompt_path: str
    sharpness: float
    luma_mean: float
    luma_std: float
    sat_mean: float
    overexposed_frac: float
    underexposed_frac: float
    flags: str


INDEX_RE = re.compile(r"img_(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


def parse_index(path: Path) -> int:
    m = INDEX_RE.search(path.name)
    if not m:
        return -1
    return int(m.group(1))


def compute_metrics(img_path: Path) -> Tuple[float, float, float, float, float, float]:
    im = Image.open(img_path).convert("RGB")
    arr = np.array(im, dtype=np.uint8)
    y = rgb_to_luma(arr)
    y_mean = float(np.mean(y))
    y_std = float(np.std(y))
    sat = saturation_proxy(arr)
    over = float(np.mean(y > 0.95))
    under = float(np.mean(y < 0.05))
    gray_u8 = (y * 255.0).astype(np.uint8)
    sharp = laplacian_var(gray_u8)
    return sharp, y_mean, y_std, sat, over, under


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True, help="Directory containing img_*.png and img_*.txt")
    ap.add_argument("--out", required=True, help="Output report directory")
    ap.add_argument("--blur-threshold", type=float, default=35.0, help="Flag if sharpness < threshold")
    ap.add_argument("--low-contrast", type=float, default=0.08, help="Flag if luma_std < threshold")
    ap.add_argument("--overexposed", type=float, default=0.08, help="Flag if overexposed fraction > threshold")
    ap.add_argument("--underexposed", type=float, default=0.08, help="Flag if underexposed fraction > threshold")
    ap.add_argument("--max-rows", type=int, default=0, help="0=all")
    ap.add_argument("--copy-flagged", type=int, default=0, help="Copy up to N flagged images into out/flagged")
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    imgs = [p for p in iter_images(images_dir) if parse_index(p) >= 0]
    imgs.sort(key=parse_index)
    if args.max_rows and args.max_rows > 0:
        imgs = imgs[: args.max_rows]
    if not imgs:
        raise SystemExit(f"No img_*.png found under {images_dir}")

    rows: List[Row] = []
    flagged: List[Row] = []

    for img_path in imgs:
        idx = parse_index(img_path)
        prompt_path = img_path.with_suffix(".txt")
        sharp, y_mean, y_std, sat, over, under = compute_metrics(img_path)

        flags: List[str] = []
        if sharp < args.blur_threshold:
            flags.append("blurry")
        if y_std < args.low_contrast:
            flags.append("low_contrast")
        if over > args.overexposed:
            flags.append("overexposed")
        if under > args.underexposed:
            flags.append("underexposed")

        row = Row(
            index=idx,
            image_path=str(img_path),
            prompt_path=str(prompt_path) if prompt_path.exists() else "",
            sharpness=sharp,
            luma_mean=y_mean,
            luma_std=y_std,
            sat_mean=sat,
            overexposed_frac=over,
            underexposed_frac=under,
            flags=",".join(flags),
        )
        rows.append(row)
        if flags:
            flagged.append(row)

    # CSV
    csv_path = out_dir / "acceptance_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    # Summary JSON
    summary = {
        "images_dir": str(images_dir),
        "count": len(rows),
        "flagged_count": len(flagged),
        "flag_rate": float(len(flagged) / max(1, len(rows))),
        "thresholds": {
            "blur_sharpness_lt": args.blur_threshold,
            "low_contrast_luma_std_lt": args.low_contrast,
            "overexposed_frac_gt": args.overexposed,
            "underexposed_frac_gt": args.underexposed,
        },
        "stats": {
            "sharpness_mean": float(np.mean([r.sharpness for r in rows])),
            "luma_mean_mean": float(np.mean([r.luma_mean for r in rows])),
            "luma_std_mean": float(np.mean([r.luma_std for r in rows])),
            "sat_mean_mean": float(np.mean([r.sat_mean for r in rows])),
        },
    }
    (out_dir / "acceptance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.copy_flagged and args.copy_flagged > 0:
        flagged_dir = out_dir / "flagged"
        flagged_dir.mkdir(parents=True, exist_ok=True)
        for r in flagged[: args.copy_flagged]:
            src = Path(r.image_path)
            dst = flagged_dir / src.name
            try:
                dst.write_bytes(src.read_bytes())
            except Exception:
                pass

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

