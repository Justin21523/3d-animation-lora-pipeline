#!/usr/bin/env python3
"""
Compute lightweight QA stats over a batch-generated video directory.

This is NOT a semantic "two people" detector. It provides practical flags that
correlate with common failures:
  - very low sharpness (blurry)
  - over/under exposure
  - temporal flicker proxy (frame-to-frame brightness instability)
  - unexpected frame count

Designed for short AnimateDiff clips (e.g. 16 frames), but works for any mp4.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import imageio.v2 as imageio
import numpy as np


VID_EXTS = {".mp4"}


def iter_videos(root: Path) -> Iterable[Path]:
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in VID_EXTS:
                yield p


def rgb_to_luma(arr: np.ndarray) -> np.ndarray:
    arr_f = arr.astype(np.float32) / 255.0
    r = arr_f[:, :, 0]
    g = arr_f[:, :, 1]
    b = arr_f[:, :, 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def laplacian_var(gray: np.ndarray) -> float:
    # simple 2D Laplacian kernel (naive conv; OK for QA)
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
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


def read_frames(video_path: Path, max_frames: int = 0) -> List[np.ndarray]:
    reader = imageio.get_reader(str(video_path))
    frames: List[np.ndarray] = []
    try:
        for i, frame in enumerate(reader):
            frames.append(frame)
            if max_frames and (i + 1) >= max_frames:
                break
    finally:
        try:
            reader.close()
        except Exception:
            pass
    return frames


def compute_metrics(frames: List[np.ndarray]) -> Tuple[int, float, float, float, float]:
    if not frames:
        return 0, 0.0, 0.0, 0.0, 0.0
    luma_means: List[float] = []
    over_fracs: List[float] = []
    under_fracs: List[float] = []
    sharpness: List[float] = []

    for frame in frames:
        # frame can be RGBA
        if frame.ndim != 3 or frame.shape[2] < 3:
            continue
        rgb = frame[:, :, :3].astype(np.uint8)
        y = rgb_to_luma(rgb)
        luma_means.append(float(np.mean(y)))
        over_fracs.append(float(np.mean(y > 0.95)))
        under_fracs.append(float(np.mean(y < 0.05)))
        gray_u8 = (y * 255.0).astype(np.uint8)
        sharpness.append(laplacian_var(gray_u8))

    if not luma_means:
        return 0, 0.0, 0.0, 0.0, 0.0

    frame_count = len(luma_means)
    luma_mean = float(np.mean(luma_means))
    flicker = float(np.std(luma_means))  # brightness instability proxy
    sharp_mean = float(np.mean(sharpness)) if sharpness else 0.0
    over = float(np.mean(over_fracs))
    under = float(np.mean(under_fracs))
    # Return values split to keep Row simple
    return frame_count, luma_mean, flicker, sharp_mean, max(over, under)


@dataclass(frozen=True)
class Row:
    video_id: str
    video_path: str
    caption_path: str
    frame_count: int
    luma_mean: float
    flicker: float
    sharpness_mean: float
    exposure_extreme: float
    flags: str


VID_ID_RE = re.compile(r"^(?P<id>.+)\.mp4$", re.IGNORECASE)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos-dir", required=True, help="Directory containing *.mp4")
    ap.add_argument("--captions-dir", default="", help="Directory containing <id>.txt captions (optional)")
    ap.add_argument("--out", required=True, help="Output report directory")
    ap.add_argument("--expected-frames", type=int, default=16, help="Flag if frame_count != expected (0 disables)")
    ap.add_argument("--blur-threshold", type=float, default=22.0, help="Flag if sharpness_mean < threshold")
    ap.add_argument("--flicker-threshold", type=float, default=0.06, help="Flag if flicker > threshold")
    ap.add_argument("--exposure-threshold", type=float, default=0.10, help="Flag if extreme exposure frac > threshold")
    ap.add_argument("--max-videos", type=int, default=0, help="0=all")
    ap.add_argument("--max-frames", type=int, default=0, help="0=all frames, otherwise read only first N frames")
    ap.add_argument("--copy-flagged", type=int, default=0, help="Copy up to N flagged videos into out/flagged")
    args = ap.parse_args()

    videos_dir = Path(args.videos_dir)
    captions_dir = Path(args.captions_dir) if args.captions_dir else None
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    vids = list(iter_videos(videos_dir))
    vids.sort(key=lambda p: p.name)
    if args.max_videos and args.max_videos > 0:
        vids = vids[: args.max_videos]
    if not vids:
        raise SystemExit(f"No .mp4 found under {videos_dir}")

    rows: List[Row] = []
    flagged: List[Row] = []

    for vid_path in vids:
        m = VID_ID_RE.match(vid_path.name)
        if not m:
            continue
        vid_id = m.group("id")
        caption_path = ""
        if captions_dir is not None:
            cp = captions_dir / f"{vid_id}.txt"
            if cp.exists():
                caption_path = str(cp)

        frames = read_frames(vid_path, max_frames=args.max_frames)
        frame_count, luma_mean, flicker, sharp_mean, exposure_extreme = compute_metrics(frames)

        flags: List[str] = []
        if args.expected_frames and frame_count != args.expected_frames:
            flags.append("bad_frame_count")
        if sharp_mean < args.blur_threshold:
            flags.append("blurry")
        if flicker > args.flicker_threshold:
            flags.append("flicker")
        if exposure_extreme > args.exposure_threshold:
            flags.append("exposure_extreme")

        row = Row(
            video_id=vid_id,
            video_path=str(vid_path),
            caption_path=caption_path,
            frame_count=frame_count,
            luma_mean=luma_mean,
            flicker=flicker,
            sharpness_mean=sharp_mean,
            exposure_extreme=exposure_extreme,
            flags=",".join(flags),
        )
        rows.append(row)
        if flags:
            flagged.append(row)

    csv_path = out_dir / "video_acceptance_report.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))

    summary = {
        "videos_dir": str(videos_dir),
        "captions_dir": str(captions_dir) if captions_dir is not None else "",
        "count": len(rows),
        "flagged_count": len(flagged),
        "flag_rate": float(len(flagged) / max(1, len(rows))),
        "thresholds": {
            "expected_frames": args.expected_frames,
            "blur_sharpness_lt": args.blur_threshold,
            "flicker_std_luma_gt": args.flicker_threshold,
            "exposure_extreme_frac_gt": args.exposure_threshold,
        },
        "stats": {
            "sharpness_mean": float(np.mean([r.sharpness_mean for r in rows])),
            "flicker_mean": float(np.mean([r.flicker for r in rows])),
            "luma_mean_mean": float(np.mean([r.luma_mean for r in rows])),
        },
    }
    (out_dir / "video_acceptance_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.copy_flagged and args.copy_flagged > 0:
        flagged_dir = out_dir / "flagged"
        flagged_dir.mkdir(parents=True, exist_ok=True)
        for r in flagged[: args.copy_flagged]:
            src = Path(r.video_path)
            dst = flagged_dir / src.name
            try:
                dst.write_bytes(src.read_bytes())
            except Exception:
                pass

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

