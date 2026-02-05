#!/usr/bin/env python3
"""
Prepare a Wan2.1 LoRA dataset from existing mp4 clips + txt captions.

Input:
  <input_dir>/
    videos/*.mp4
    captions/*.txt
    (optional) metadata.jsonl

Output:
  <output_dir>/
    videos/*.mp4          (optionally resized / re-timed via ffmpeg)
    captions/*.txt
    metadata.jsonl        (Wan2.1 / DiffSynth-Studio compatible; includes `video` key)

Notes:
  - Uses ffmpeg for resizing/padding, fps normalization, and frame limiting.
  - Designed for short clips (e.g. AnimateDiff 16 frames).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable


def parse_size(value: str) -> tuple[int, int]:
    token = value.lower().replace(" ", "")
    if "x" in token:
        w_str, h_str = token.split("x", 1)
    elif "*" in token:
        w_str, h_str = token.split("*", 1)
    else:
        raise ValueError("size must be like 832x480 or 832*480")
    return int(w_str), int(h_str)


def iter_ids(videos_dir: Path) -> Iterable[str]:
    for p in sorted(videos_dir.glob("*.mp4")):
        yield p.stem


def build_ffmpeg_cmd(
    src: Path,
    dst: Path,
    *,
    width: int,
    height: int,
    fps: int,
    frames: int,
    overwrite: bool,
) -> list[str]:
    scale_pad = (
        f"scale=w={width}:h={height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
        f"fps={fps},format=yuv420p"
    )
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    cmd.append("-y" if overwrite else "-n")
    cmd += [
        "-i",
        str(src),
        "-vf",
        scale_pad,
        "-frames:v",
        str(frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dst),
    ]
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", required=True, help="Dataset dir containing videos/ and captions/")
    ap.add_argument("--output-dir", required=True, help="Output dataset directory")
    ap.add_argument("--size", default="832x480", help="Output size, e.g. 832x480")
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--accepted-ids", default="", help="Optional newline-separated id list to include")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-transcode", action="store_true", help="Copy mp4 as-is (still writes metadata)")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    src_videos = input_dir / "videos"
    src_caps = input_dir / "captions"
    if not src_videos.is_dir() or not src_caps.is_dir():
        raise SystemExit(f"Expected videos/ and captions/ under: {input_dir}")

    width, height = parse_size(args.size)
    fps = max(1, int(args.fps))
    frames = max(1, int(args.frames))

    accepted: set[str] | None = None
    if args.accepted_ids:
        p = Path(args.accepted_ids)
        if not p.exists():
            raise SystemExit(f"accepted ids file not found: {p}")
        accepted = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}

    out_dir = Path(args.output_dir)
    out_videos = out_dir / "videos"
    out_caps = out_dir / "captions"
    out_videos.mkdir(parents=True, exist_ok=True)
    out_caps.mkdir(parents=True, exist_ok=True)

    meta_path = out_dir / "metadata.jsonl"
    exported = 0
    with meta_path.open("w", encoding="utf-8") as mf:
        for vid_id in iter_ids(src_videos):
            if accepted is not None and vid_id not in accepted:
                continue
            src_vid = src_videos / f"{vid_id}.mp4"
            src_cap = src_caps / f"{vid_id}.txt"
            if not src_vid.exists() or not src_cap.exists():
                continue

            dst_vid = out_videos / src_vid.name
            dst_cap = out_caps / src_cap.name
            if args.overwrite or not dst_cap.exists():
                shutil.copy2(src_cap, dst_cap)

            if args.no_transcode:
                if args.overwrite or not dst_vid.exists():
                    shutil.copy2(src_vid, dst_vid)
            else:
                if dst_vid.exists() and not args.overwrite:
                    pass
                else:
                    cmd = build_ffmpeg_cmd(
                        src=src_vid,
                        dst=dst_vid,
                        width=width,
                        height=height,
                        fps=fps,
                        frames=frames,
                        overwrite=True,
                    )
                    subprocess.run(cmd, check=True)

            text = dst_cap.read_text(encoding="utf-8", errors="ignore").strip()
            mf.write(
                json.dumps(
                    {
                        "file_name": f"videos/{vid_id}.mp4",
                        "video": f"videos/{vid_id}.mp4",
                        "prompt": text,
                        "text": text,
                        "caption": text,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            exported += 1

    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# Wan2.1 LoRA Video Dataset",
                "",
                f"- size: {width}x{height}",
                f"- fps: {fps}",
                f"- frames: {frames}",
                "",
                "Trainer expects:",
                "- `metadata.jsonl` with `video` keys (relative paths under dataset dir)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps({"exported": exported, "output_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()

