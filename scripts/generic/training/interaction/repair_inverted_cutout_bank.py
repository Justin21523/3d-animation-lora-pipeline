#!/usr/bin/env python3
"""
Repair inverted RGBA cutouts in-place.

Some cutouts may have inverted alpha (background opaque, subject transparent).
This script detects likely inversion by checking corner alpha values and, if
needed, inverts alpha and clears RGB in fully transparent pixels to reduce
resize halos. It also refreshes the saved bbox in the sidecar JSON if present.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image


def _corner_alpha_mean(alpha: Image.Image) -> float:
    w, h = alpha.size
    pts = [(5, 5), (w - 6, 5), (5, h - 6), (w - 6, h - 6)]
    vals = [alpha.getpixel(p) for p in pts]
    return float(sum(vals) / len(vals))


def _fix_image(path: Path) -> Tuple[bool, Optional[Tuple[int, int, int, int]]]:
    with Image.open(path) as im:
        im = im.convert("RGBA")
        alpha = im.getchannel("A")
        inverted = _corner_alpha_mean(alpha) > 127.0
        if inverted:
            alpha = Image.eval(alpha, lambda a: 255 - a)
            im.putalpha(alpha)

        arr = np.array(im, dtype=np.uint8)
        mask0 = arr[:, :, 3] == 0
        arr[mask0, 0:3] = 0
        fixed = Image.fromarray(arr, mode="RGBA")

        bbox = fixed.getchannel("A").getbbox()
        if inverted:
            fixed.save(path, optimize=True)
        return inverted, bbox


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutout-root", type=Path, required=True, help="Root that contains per-character cutout folders")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    pngs = sorted(args.cutout_root.glob("*/*.png"))
    fixed = 0
    checked = 0

    for p in pngs:
        checked += 1
        try:
            inverted, bbox = _fix_image(p)
        except Exception:
            continue

        if inverted:
            fixed += 1
            if args.dry_run:
                continue

            sidecar = p.with_suffix(".json")
            if sidecar.exists() and bbox is not None:
                try:
                    data = json.loads(sidecar.read_text(encoding="utf-8"))
                    data["bbox"] = list(bbox)
                    meta = data.get("meta")
                    if isinstance(meta, dict):
                        meta["alpha_inverted_fix_applied"] = True
                    sidecar.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                except Exception:
                    pass

    print(f"Checked {checked} cutouts; repaired {fixed} inverted images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

