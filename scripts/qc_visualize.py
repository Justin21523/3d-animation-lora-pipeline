#!/usr/bin/env python
"""
Visualize samples from detections/pose/segmentation metadata.
Saves contact sheets for quick QC.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QC visualization for detections/pose/segmentation.")
    parser.add_argument("--detections", type=str, help="Detections metadata (parquet/csv).")
    parser.add_argument("--poses", type=str, help="Poses metadata (parquet/csv).")
    parser.add_argument("--fg", type=str, help="Foreground metadata (parquet/csv).")
    parser.add_argument("--samples", type=int, default=4, help="Number of samples to visualize.")
    parser.add_argument("--output", type=str, default="qc_contact_sheet.png", help="Output image path.")
    return parser.parse_args()


def load_meta(path: str | None):
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    try:
        return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)
    except Exception:
        return None


def draw_detections(df_det: pd.DataFrame, samples: int):
    images = []
    for _, row in df_det.head(samples).iterrows():
        img_path = Path(row["image_path"])
        if not img_path.exists():
            continue
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            draw = ImageDraw.Draw(img)
            x1, y1, x2, y2 = row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
            draw.text((x1, y1), f"{row['class_name']}:{row['score']:.2f}", fill="red")
            images.append(img.copy())
    return images


def draw_poses(df_pose: pd.DataFrame, samples: int):
    images = []
    for _, row in df_pose.head(samples).iterrows():
        pose_img_path = Path(row["pose_image_path"])
        if not pose_img_path.exists():
            continue
        with Image.open(pose_img_path) as img:
            images.append(img.convert("RGB").copy())
    return images


def draw_fg(df_fg: pd.DataFrame, samples: int):
    images = []
    for _, row in df_fg.head(samples).iterrows():
        rgba_path = Path(row.get("rgba_path") or "")
        if not rgba_path.exists():
            continue
        with Image.open(rgba_path) as img:
            images.append(img.convert("RGBA").copy())
    return images


def make_contact_sheet(images, cols=2, bg_color=(30, 30, 30)):
    if not images:
        return None
    widths, heights = zip(*(img.size for img in images))
    max_w, max_h = max(widths), max(heights)
    rows = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * max_w, rows * max_h), color=bg_color)
    for idx, img in enumerate(images):
        r = idx // cols
        c = idx % cols
        sheet.paste(img.convert("RGB"), (c * max_w, r * max_h))
    return sheet


def main() -> None:
    args = parse_args()
    det_df = load_meta(args.detections)
    pose_df = load_meta(args.poses)
    fg_df = load_meta(args.fg)

    all_imgs = []
    if det_df is not None:
        all_imgs.extend(draw_detections(det_df, args.samples))
    if pose_df is not None:
        all_imgs.extend(draw_poses(pose_df, args.samples))
    if fg_df is not None:
        all_imgs.extend(draw_fg(fg_df, args.samples))

    sheet = make_contact_sheet(all_imgs, cols=2)
    if sheet is None:
        print("No images to visualize.")
        return
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    print(f"Saved QC contact sheet to {out_path}")


if __name__ == "__main__":
    main()
