#!/usr/bin/env python3
"""
Chroma Key Cutout Bank Builder
==============================

Converts green-screen (or magenta/cyan) single-character renders into RGBA cutouts.

Input:
  - images: PNGs (1024x1024) with solid chroma background

Output (per image):
  - RGBA cutout PNG + JSON metadata (bbox, center, scale)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from PIL import Image, ImageChops, ImageFilter


@dataclass(frozen=True)
class KeyColor:
    name: str
    rgb: Tuple[int, int, int]


KEY_COLORS = [
    KeyColor("green", (0, 255, 0)),
    KeyColor("magenta", (255, 0, 255)),
    KeyColor("cyan", (0, 255, 255)),
]


def _mean_corner_rgb(img: Image.Image, patch: int = 24) -> Tuple[int, int, int]:
    rgb = img.convert("RGB")
    w, h = rgb.size
    patch = min(patch, w // 4, h // 4)
    if patch <= 0:
        return (0, 0, 0)

    corners = [
        rgb.crop((0, 0, patch, patch)),
        rgb.crop((w - patch, 0, w, patch)),
        rgb.crop((0, h - patch, patch, h)),
        rgb.crop((w - patch, h - patch, w, h)),
    ]
    total = [0.0, 0.0, 0.0]
    count = 0
    for c in corners:
        px = c.getdata()
        for r, g, b in px:
            total[0] += r
            total[1] += g
            total[2] += b
            count += 1
    if count == 0:
        return (0, 0, 0)
    return (int(total[0] / count), int(total[1] / count), int(total[2] / count))


def _closest_key_color(rgb: Tuple[int, int, int]) -> KeyColor:
    def dist2(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> int:
        return sum((a[i] - b[i]) ** 2 for i in range(3))

    return min(KEY_COLORS, key=lambda kc: dist2(rgb, kc.rgb))


def _rgb_distance_mask(img: Image.Image, key_rgb: Tuple[int, int, int], tolerance: int) -> Image.Image:
    """
    Returns an L-mode mask where 255 means background (key color), 0 means foreground.
    """
    rgb = img.convert("RGB")
    key = Image.new("RGB", rgb.size, key_rgb)
    diff = ImageChops.difference(rgb, key).convert("L")
    # Background is "close" to key => low diff; invert to get bg mask.
    # Create binary-ish mask with softness.
    bg = diff.point(lambda v: 255 if v < tolerance else 0)
    # Soften edges a bit
    bg = bg.filter(ImageFilter.GaussianBlur(radius=1.2))
    return bg


def _alpha_from_bg_mask(bg_mask: Image.Image) -> Image.Image:
    # bg_mask=255(background) -> alpha=0, bg_mask=0(foreground) -> alpha=255
    return bg_mask.point(lambda v: 255 - v)


def _bbox_from_alpha(alpha: Image.Image, threshold: int = 16) -> Optional[Tuple[int, int, int, int]]:
    a = alpha.point(lambda v: 255 if v >= threshold else 0)
    bbox = a.getbbox()
    return bbox


def _save_cutout(out_png: Path, out_json: Path, rgba: Image.Image, bbox: Optional[Tuple[int, int, int, int]], meta: Dict[str, Any]) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    rgba.save(out_png, optimize=True)
    payload = {"bbox": bbox, "meta": meta}
    out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", type=Path, required=True, help="Directory of green-screen PNGs")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output cutout directory")
    ap.add_argument("--tolerance", type=int, default=42, help="Color distance threshold (higher removes more bg)")
    ap.add_argument("--alpha-threshold", type=int, default=16, help="Alpha threshold for bbox detection")
    ap.add_argument("--max-images", type=int, default=0, help="Limit processed images (0=all)")
    args = ap.parse_args()

    imgs = sorted(args.in_dir.glob("*.png"))
    if args.max_images and args.max_images > 0:
        imgs = imgs[: args.max_images]

    for p in imgs:
        with Image.open(p) as im:
            im.load()
            corner_rgb = _mean_corner_rgb(im)
            key = _closest_key_color(corner_rgb)
            bg_mask = _rgb_distance_mask(im, key.rgb, tolerance=args.tolerance)
            alpha = _alpha_from_bg_mask(bg_mask)
            rgba = im.convert("RGBA")
            rgba.putalpha(alpha)

            bbox = _bbox_from_alpha(alpha, threshold=args.alpha_threshold)
            meta = {
                "source": str(p),
                "key_color": key.name,
                "corner_rgb": corner_rgb,
                "tolerance": args.tolerance,
            }

            out_png = args.out_dir / p.name
            out_json = args.out_dir / f"{p.stem}.json"
            _save_cutout(out_png, out_json, rgba, bbox, meta)

    print(f"Processed {len(imgs)} images -> {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

