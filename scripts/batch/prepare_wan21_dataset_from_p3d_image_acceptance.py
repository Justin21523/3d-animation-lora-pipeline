#!/usr/bin/env python3
"""
Prepare a Wan2.1 LoRA dataset from P3D image-acceptance outputs (img_*.png + img_*.txt)
using a *mild* QC filter (not too strict).

Input:
  - acceptance_report.csv (from scripts/evaluation/p3d_pair_acceptance_report.py)
  - image_path + prompt_path in each row

Output:
  <output_dir>/
    videos/<img_id>.mp4
    metadata.jsonl          (contains `video` key + prompt/text/caption)
    accepted_ids.txt
    rejected_ids.txt
    filter_summary.json

Notes:
  - This produces "still videos" by looping a single frame (image) for N frames.
  - Useful for identity/consistency training; less useful for learning motion dynamics.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


def parse_size(value: str) -> tuple[int, int]:
    token = value.lower().replace(" ", "")
    if "x" in token:
        w_str, h_str = token.split("x", 1)
    elif "*" in token:
        w_str, h_str = token.split("*", 1)
    else:
        raise ValueError("size must be like 832x480 or 832*480")
    return int(w_str), int(h_str)


def safe_float(value: str) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def safe_int(value: str) -> int:
    try:
        return int(value)
    except Exception:
        return -1


@dataclass(frozen=True)
class Row:
    index: int
    image_path: str
    prompt_path: str
    sharpness: float
    luma_mean: float
    luma_std: float
    overexposed_frac: float
    underexposed_frac: float
    flags: str

    @staticmethod
    def from_csv(d: dict[str, str]) -> "Row":
        return Row(
            index=safe_int(d.get("index", "")),
            image_path=d.get("image_path", ""),
            prompt_path=d.get("prompt_path", ""),
            sharpness=safe_float(d.get("sharpness", "")),
            luma_mean=safe_float(d.get("luma_mean", "")),
            luma_std=safe_float(d.get("luma_std", "")),
            overexposed_frac=safe_float(d.get("overexposed_frac", "")),
            underexposed_frac=safe_float(d.get("underexposed_frac", "")),
            flags=(d.get("flags", "") or "").strip(),
        )


def load_rows(report_csv: Path) -> List[Row]:
    with report_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [Row.from_csv(d) for d in reader]
    rows = [r for r in rows if r.index >= 0 and r.image_path]
    rows.sort(key=lambda r: r.index)
    return rows


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def build_ffmpeg_cmd(
    image_path: Path,
    video_path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    frames: int,
    overwrite: bool,
) -> list[str]:
    duration_sec = frames / max(1, fps)
    scale_pad = (
        f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        "format=yuv420p"
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    cmd.append("-y" if overwrite else "-n")
    cmd += [
        "-loop",
        "1",
        "-i",
        str(image_path),
        "-t",
        f"{duration_sec:.3f}",
        "-r",
        str(fps),
        "-vf",
        scale_pad,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    return cmd


def passes_mild(
    row: Row,
    *,
    min_sharpness: float,
    min_luma_std: float,
    min_luma_mean: float,
    max_luma_mean: float,
    max_overexposed_frac: float,
    max_underexposed_frac: float,
    require_no_flags: bool,
) -> bool:
    if require_no_flags and row.flags:
        return False
    if row.sharpness < min_sharpness:
        return False
    if row.luma_std < min_luma_std:
        return False
    if row.luma_mean < min_luma_mean:
        return False
    if row.luma_mean > max_luma_mean:
        return False
    if row.overexposed_frac > max_overexposed_frac:
        return False
    if row.underexposed_frac > max_underexposed_frac:
        return False
    return True


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-csv", required=True, help="acceptance_report.csv")
    ap.add_argument("--out-dir", required=True, help="Output dataset directory")
    ap.add_argument("--size", default="832x480")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-items", type=int, default=0, help="0=all")

    # Mild defaults (not too strict): only drop obvious failures.
    ap.add_argument("--min-sharpness", type=float, default=25.0)
    ap.add_argument("--min-luma-std", type=float, default=0.05)
    ap.add_argument("--min-luma-mean", type=float, default=0.12)
    ap.add_argument("--max-luma-mean", type=float, default=0.93)
    ap.add_argument("--max-overexposed-frac", type=float, default=0.12)
    ap.add_argument("--max-underexposed-frac", type=float, default=0.12)
    ap.add_argument("--require-no-flags", type=int, default=0)
    args = ap.parse_args()

    report_csv = Path(args.report_csv)
    rows = load_rows(report_csv)
    if not rows:
        raise SystemExit(f"no rows found in {report_csv}")

    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]

    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    accepted: List[Row] = []
    rejected: List[Row] = []
    for r in rows:
        ok = passes_mild(
            r,
            min_sharpness=args.min_sharpness,
            min_luma_std=args.min_luma_std,
            min_luma_mean=args.min_luma_mean,
            max_luma_mean=args.max_luma_mean,
            max_overexposed_frac=args.max_overexposed_frac,
            max_underexposed_frac=args.max_underexposed_frac,
            require_no_flags=bool(args.require_no_flags),
        )
        (accepted if ok else rejected).append(r)

    width, height = parse_size(args.size)
    fps = max(1, int(args.fps))
    frames = max(1, int(args.frames))

    meta_path = out_dir / "metadata.jsonl"
    accepted_ids: List[str] = []
    rejected_ids: List[str] = []

    with meta_path.open("w", encoding="utf-8") as mf:
        for r in accepted:
            img = Path(r.image_path)
            if not img.exists():
                rejected.append(r)
                continue
            vid_id = img.stem
            out_mp4 = videos_dir / f"{vid_id}.mp4"

            prompt = ""
            if r.prompt_path and Path(r.prompt_path).exists():
                prompt = read_text(Path(r.prompt_path))
            else:
                maybe = img.with_suffix(".txt")
                if maybe.exists():
                    prompt = read_text(maybe)
            if not prompt:
                rejected.append(r)
                continue

            if args.overwrite or not out_mp4.exists():
                cmd = build_ffmpeg_cmd(
                    image_path=img,
                    video_path=out_mp4,
                    width=width,
                    height=height,
                    fps=fps,
                    frames=frames,
                    overwrite=True,
                )
                subprocess.run(cmd, check=True)

            mf.write(
                json.dumps(
                    {
                        "file_name": f"videos/{vid_id}.mp4",
                        "video": f"videos/{vid_id}.mp4",
                        "prompt": prompt,
                        "text": prompt,
                        "caption": prompt,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            accepted_ids.append(vid_id)

    for r in rejected:
        img = Path(r.image_path)
        rejected_ids.append(img.stem if img.stem else f"index_{r.index}")

    write_lines(out_dir / "accepted_ids.txt", accepted_ids)
    write_lines(out_dir / "rejected_ids.txt", rejected_ids)

    summary = {
        "report_csv": str(report_csv),
        "rows": len(rows),
        "accepted": len(accepted_ids),
        "rejected": len(rejected_ids),
        "output_dir": str(out_dir),
        "settings": {
            "size": args.size,
            "fps": fps,
            "frames": frames,
            "overwrite": bool(args.overwrite),
            "max_items": args.max_items,
        },
        "thresholds": {
            "min_sharpness": args.min_sharpness,
            "min_luma_std": args.min_luma_std,
            "min_luma_mean": args.min_luma_mean,
            "max_luma_mean": args.max_luma_mean,
            "max_overexposed_frac": args.max_overexposed_frac,
            "max_underexposed_frac": args.max_underexposed_frac,
            "require_no_flags": bool(args.require_no_flags),
        },
    }
    (out_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

