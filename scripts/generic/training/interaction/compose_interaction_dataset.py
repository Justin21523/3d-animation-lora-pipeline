#!/usr/bin/env python3
"""
Compose Universal Two-Character Interaction Dataset
===================================================

Takes single-character RGBA cutouts and composes 1024x1024 images with:
  - two characters on a simple background
  - simple shadows
  - auto-generated captions for training a universal "interaction/composition" LoRA

Captions are identity-agnostic (no character names), so the LoRA can be combined
with identity LoRAs at inference time.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter


DEFAULT_TEMPLATES = [
    {"id": "side_by_side_standing", "desc": "standing side by side", "min_body": 0.55, "gap": 0.06},
    {"id": "walking_together", "desc": "walking together side by side", "min_body": 0.55, "gap": 0.08},
    {"id": "talking_face_to_face", "desc": "talking face to face", "min_body": 0.55, "gap": 0.04, "face_each_other": True},
    {"id": "pointing_same_direction", "desc": "both pointing in the same direction", "min_body": 0.55, "gap": 0.06},
    {"id": "celebrating_together", "desc": "celebrating together", "min_body": 0.55, "gap": 0.08},
    {"id": "sitting_together", "desc": "sitting together", "min_body": 0.45, "gap": 0.08},
]


BACKGROUND_PRESETS = [
    {"id": "plain", "kind": "solid", "color": (245, 245, 245)},
    {"id": "warm", "kind": "solid", "color": (250, 244, 235)},
    {"id": "cool", "kind": "solid", "color": (235, 245, 250)},
    {"id": "gradient", "kind": "gradient", "top": (250, 250, 250), "bottom": (235, 235, 235)},
]


def _load_cutouts(char_dir: Path) -> List[Tuple[Path, Optional[Tuple[int, int, int, int]]]]:
    items: List[Tuple[Path, Optional[Tuple[int, int, int, int]]]] = []
    for png in sorted(char_dir.glob("*.png")):
        meta = char_dir / f"{png.stem}.json"
        bbox = None
        if meta.exists():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                bbox = data.get("bbox")
                if isinstance(bbox, list) and len(bbox) == 4:
                    bbox = tuple(int(x) for x in bbox)  # type: ignore[assignment]
                else:
                    bbox = None
            except Exception:
                bbox = None
        items.append((png, bbox))
    return items


def _alpha_bbox(im: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    if im.mode != "RGBA":
        im = im.convert("RGBA")
    return im.getchannel("A").getbbox()


def _prepare_cutout(png: Path, bbox: Optional[Tuple[int, int, int, int]]) -> Image.Image:
    im = Image.open(png).convert("RGBA")
    im.load()
    # Prefer explicit bbox, else compute from alpha to remove excess transparent border
    crop = bbox or _alpha_bbox(im)
    if crop:
        im = im.crop(crop)
    return im


def _compute_layout(
    canvas: int,
    target_height_px: int,
    template: Dict[str, Any],
    a_size: Tuple[int, int],
    b_size: Tuple[int, int],
) -> Tuple[int, int, int, int, int]:
    """
    Returns:
      (a_target_h, b_target_h, a_center_x, b_center_x, desired_gap_px)
    Ensures the two characters fit side-by-side without overlap.
    """
    w = h = int(canvas)
    desired_gap_px = int(w * float(template.get("gap", 0.06)))

    aw, ah = a_size
    bw, bh = b_size
    ah = max(1, int(ah))
    bh = max(1, int(bh))

    # Start from desired target heights; scale both down if total width would exceed canvas
    a_h = int(target_height_px)
    b_h = int(target_height_px)

    def scaled_w(sw: int, sh: int, th: int) -> int:
        return max(1, int(sw * (th / float(max(1, sh)))))

    # Iteratively shrink until fit
    for _ in range(6):
        a_w = scaled_w(aw, ah, a_h)
        b_w = scaled_w(bw, bh, b_h)
        total = a_w + b_w + desired_gap_px
        if total <= int(w * 0.92):
            break
        # shrink both proportionally
        shrink = (w * 0.92) / float(max(1, total))
        a_h = max(1, int(a_h * shrink))
        b_h = max(1, int(b_h * shrink))

    a_w = scaled_w(aw, ah, a_h)
    b_w = scaled_w(bw, bh, b_h)

    # Place centers so bounding boxes don't overlap and are roughly centered
    mid = w // 2
    a_center_x = int(mid - (desired_gap_px / 2.0 + a_w / 2.0))
    b_center_x = int(mid + (desired_gap_px / 2.0 + b_w / 2.0))

    # Clamp to margins
    margin = int(w * 0.04)
    a_center_x = max(margin + a_w // 2, a_center_x)
    b_center_x = min(w - margin - b_w // 2, b_center_x)
    return a_h, b_h, a_center_x, b_center_x, desired_gap_px


def _make_background(w: int, h: int, preset: Dict[str, Any]) -> Image.Image:
    kind = preset["kind"]
    if kind == "solid":
        return Image.new("RGB", (w, h), tuple(preset["color"]))
    if kind == "gradient":
        top = preset["top"]
        bottom = preset["bottom"]
        bg = Image.new("RGB", (w, h))
        draw = ImageDraw.Draw(bg)
        for y in range(h):
            t = y / max(1, h - 1)
            col = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
            draw.line([(0, y), (w, y)], fill=col)
        return bg
    return Image.new("RGB", (w, h), (245, 245, 245))


def _add_shadow(canvas: Image.Image, center_x: int, ground_y: int, radius_x: int, radius_y: int, opacity: int = 90) -> None:
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    box = (center_x - radius_x, ground_y - radius_y, center_x + radius_x, ground_y + radius_y)
    draw.ellipse(box, fill=(0, 0, 0, opacity))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
    canvas.alpha_composite(shadow)


def _paste_cutout(
    base_rgba: Image.Image,
    cutout: Image.Image,
    target_height_px: int,
    center_x: int,
    ground_y: int,
) -> None:
    src = cutout.convert("RGBA")

    w, h = src.size
    if h <= 0:
        return
    scale = target_height_px / float(h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    src = src.resize((new_w, new_h), Image.LANCZOS)

    x = int(center_x - new_w / 2)
    y = int(ground_y - new_h)
    base_rgba.alpha_composite(src, dest=(x, y))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutout-root", type=Path, required=True, help="Root with per-character cutout folders")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output dataset dir (images/ + captions/)")
    ap.add_argument("--num-images", type=int, default=300, help="Total composite images to generate")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--trigger", default="pair_interaction", help="Optional trigger token to include in captions")
    ap.add_argument("--min-cutouts-per-char", type=int, default=10)
    ap.add_argument("--canvas", type=int, default=1024)
    ap.add_argument("--body-height", type=float, default=0.62, help="Target body height as fraction of canvas")
    ap.add_argument("--fixed-pair", default=None, help="Optional fixed pair: 'charA,charB' (no randomness)")
    ap.add_argument(
        "--caption-mode",
        default="universal",
        choices=["universal", "pair"],
        help="universal: identity-agnostic; pair: include character names/tokens",
    )
    ap.add_argument(
        "--char-token-map",
        type=Path,
        default=None,
        help="Optional JSON map {\"character\":\"token\"} used when caption-mode=pair. Defaults to character folder names.",
    )
    ap.add_argument(
        "--unique-cutout-combos",
        action="store_true",
        help="When using --fixed-pair, avoid repeating the same (A_cutout,B_cutout) combo if possible.",
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_images = args.out_dir / "images"
    out_captions = args.out_dir / "captions"
    out_images.mkdir(parents=True, exist_ok=True)
    out_captions.mkdir(parents=True, exist_ok=True)

    characters = sorted([p for p in args.cutout_root.iterdir() if p.is_dir()])
    bank: Dict[str, List[Tuple[Path, Optional[Tuple[int, int, int, int]]]]] = {}
    for c in characters:
        items = _load_cutouts(c)
        if len(items) >= args.min_cutouts_per_char:
            bank[c.name] = items

    if len(bank) < 2:
        raise SystemExit("Not enough characters with cutouts to compose pairs.")

    char_names = sorted(bank.keys())
    templates = DEFAULT_TEMPLATES

    w = h = int(args.canvas)
    ground_y = int(h * 0.90)
    target_height = int(h * float(args.body_height))

    token_map: Dict[str, str] = {}
    if args.char_token_map:
        token_map = json.loads(args.char_token_map.read_text(encoding="utf-8"))

    fixed_pair: Optional[Tuple[str, str]] = None
    if args.fixed_pair:
        parts = [p.strip() for p in str(args.fixed_pair).split(",") if p.strip()]
        if len(parts) != 2:
            raise SystemExit("--fixed-pair must be 'charA,charB'")
        a, b = parts
        if a not in bank or b not in bank:
            raise SystemExit(f"--fixed-pair characters must exist in cutout-root. Got: {a},{b}")
        fixed_pair = (a, b)

    unique_pairs: Optional[List[Tuple[int, int]]] = None
    if fixed_pair and args.unique_cutout_combos:
        # Precompute unique cutout index combinations for max diversity.
        # If cartesian product is smaller than num_images, we’ll emit all unique combos.
        a_name, b_name = fixed_pair
        a_n = len(bank[a_name])
        b_n = len(bank[b_name])
        combos = [(ai, bi) for ai in range(a_n) for bi in range(b_n)]
        rng.shuffle(combos)
        unique_pairs = combos

    for i in range(args.num_images):
        if fixed_pair:
            a, b = fixed_pair
        else:
            a, b = rng.sample(char_names, 2)
        template = rng.choice(templates)
        bg_preset = rng.choice(BACKGROUND_PRESETS)

        bg = _make_background(w, h, bg_preset).convert("RGBA")

        if fixed_pair and unique_pairs is not None:
            if i >= len(unique_pairs):
                # Fall back to random if we exhaust unique combos
                a_item = rng.choice(bank[a])
                b_item = rng.choice(bank[b])
            else:
                ai, bi = unique_pairs[i]
                a_item = bank[a][ai]
                b_item = bank[b][bi]
        else:
            a_item = rng.choice(bank[a])
            b_item = rng.choice(bank[b])

        im_a = _prepare_cutout(a_item[0], a_item[1])
        im_b = _prepare_cutout(b_item[0], b_item[1])

        a_h, b_h, left_x, right_x, desired_gap_px = _compute_layout(
            canvas=w,
            target_height_px=target_height,
            template=template,
            a_size=im_a.size,
            b_size=im_b.size,
        )

        # Shadows first (scaled)
        shadow_rx = int(max(24, (w * 0.08)))
        shadow_ry = int(max(10, (h * 0.02)))
        _add_shadow(bg, left_x, ground_y, radius_x=shadow_rx, radius_y=shadow_ry)
        _add_shadow(bg, right_x, ground_y, radius_x=shadow_rx, radius_y=shadow_ry)

        _paste_cutout(bg, im_a, target_height_px=a_h, center_x=left_x, ground_y=ground_y)
        _paste_cutout(bg, im_b, target_height_px=b_h, center_x=right_x, ground_y=ground_y)

        img_name = f"pair_{i:05d}.png"
        cap_name = f"pair_{i:05d}.txt"
        bg.save(out_images / img_name, optimize=True)

        parts = [
            args.trigger,
            "two 3d animated characters",
            template["desc"],
            "both characters visible",
            "left character",
            "right character",
            "simple background",
            "pixar style",
            "smooth shading",
            f"background_{bg_preset['id']}",
            f"template_{template['id']}",
            f"gap_px_{desired_gap_px}",
        ]

        if args.caption_mode == "pair":
            a_tok = str(token_map.get(a, a))
            b_tok = str(token_map.get(b, b))
            parts.insert(1, f"pair_{a_tok}_{b_tok}")
            parts.insert(2, f"left_{a_tok}")
            parts.insert(3, f"right_{b_tok}")

        caption = ", ".join(parts)
        (out_captions / cap_name).write_text(caption, encoding="utf-8")

    meta = {
        "num_images": args.num_images,
        "seed": args.seed,
        "trigger": args.trigger,
        "caption_mode": args.caption_mode,
        "fixed_pair": args.fixed_pair,
        "unique_cutout_combos": bool(args.unique_cutout_combos),
        "templates": templates,
        "backgrounds": BACKGROUND_PRESETS,
        "cutout_root": str(args.cutout_root),
    }
    (args.out_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote dataset: {args.out_dir} ({args.num_images} images)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
