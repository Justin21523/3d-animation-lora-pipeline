"""
Frame extraction helpers (stub-friendly).
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir, list_media_files


@dataclass
class ExtractFramesConfig:
    input_videos_dir: str = "data_raw"
    output_dir: str = "data_frames/frames"
    metadata_path: str = "metadata/frames.parquet"
    fps: float = 1.0
    use_stub: bool = True
    stub_frame_count: int = 5
    stub_width: int = 512
    stub_height: int = 512
    overwrite: bool = False
    log_dir: Optional[str] = "logs"


def extract_frames(config: ExtractFramesConfig, logger=None) -> List[Dict]:
    """
    Extract frames from videos or generate stub frames for testing.
    """
    logger = logger or setup_logging("extract_frames", config.log_dir)
    output_dir = ensure_dir(config.output_dir)
    records: List[Dict] = []

    videos = list_media_files(config.input_videos_dir, suffixes={".mp4", ".mkv", ".mov"})
    if not videos and not config.use_stub:
        logger.warning("No videos found in %s. Enable use_stub to generate dummy frames.", config.input_videos_dir)
        return records
    if config.use_stub and not videos:
        logger.info("Using stub mode; generating %d dummy frames.", config.stub_frame_count)
        records.extend(_generate_stub_frames(output_dir, config))
    else:
        for video_path in videos:
            logger.info("Processing video: %s", video_path)
            records.extend(_process_video(video_path, output_dir, config, logger))

    metadata_path = _write_metadata(records, config.metadata_path, logger)
    logger.info("Wrote %d frame records to %s", len(records), metadata_path)
    return records


def _process_video(video_path: Path, output_dir: Path, config: ExtractFramesConfig, logger) -> List[Dict]:
    video_id = video_path.stem
    video_output_dir = ensure_dir(output_dir / video_id)
    frame_pattern = str(video_output_dir / "frame_%06d.png")

    if config.use_stub or not _ffmpeg_available():
        logger.info("Stub mode active or ffmpeg missing; generating placeholder frames.")
        return _generate_stub_frames(video_output_dir, config, video_id=video_id)

    cmd = [
        "ffmpeg",
        "-y" if config.overwrite else "-n",
        "-i",
        str(video_path),
        "-vf",
        f"fps={config.fps}",
        frame_pattern,
    ]
    subprocess.run(cmd, check=True)

    frame_paths = sorted(video_output_dir.glob("frame_*.png"))
    records: List[Dict] = []
    for idx, frame_path in enumerate(frame_paths):
        frame_id = f"{video_id}_{idx:06d}"
        records.append(
            {
                "frame_id": frame_id,
                "video_id": video_id,
                "video_path": str(video_path),
                "frame_index": idx,
                "timestamp": None,
                "scene_id": None,
                "image_path": str(frame_path),
                "width": None,
                "height": None,
                "is_kept_dedupe": True,
                "phash": None,
                "ssim_ref_id": None,
            }
        )
    return records


def _generate_stub_frames(output_dir: Path, config: ExtractFramesConfig, video_id: str = "stub") -> List[Dict]:
    records: List[Dict] = []
    output_dir = ensure_dir(output_dir)
    for idx in range(config.stub_frame_count):
        frame_id = f"{video_id}_{idx:06d}"
        frame_path = output_dir / f"frame_{idx:06d}.png"
        _write_dummy_image(frame_path, config.stub_width, config.stub_height, idx)
        records.append(
            {
                "frame_id": frame_id,
                "video_id": video_id,
                "video_path": None,
                "frame_index": idx,
                "timestamp": None,
                "scene_id": None,
                "image_path": str(frame_path),
                "width": config.stub_width,
                "height": config.stub_height,
                "is_kept_dedupe": True,
                "phash": hashlib.sha1(frame_id.encode("utf-8")).hexdigest(),
                "ssim_ref_id": None,
            }
        )
    return records


def _write_dummy_image(path: Path, width: int, height: int, seed: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        # Fallback: write a small text marker when Pillow is unavailable.
        path.write_text(f"stub frame {seed} ({width}x{height})", encoding="utf-8")
        return

    img = Image.new("RGB", (width, height), color=(seed * 3 % 255, seed * 7 % 255, seed * 11 % 255))
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), f"frame {seed}", fill=(255, 255, 255))
    img.save(path)


def _write_metadata(records: List[Dict], metadata_path: str | Path, logger) -> Path:
    target_path = Path(metadata_path)
    ensure_dir(target_path.parent)
    if not records:
        target_path.touch()
        return target_path

    try:
        import pandas as pd

        df = pd.DataFrame(records)
        if target_path.suffix == ".parquet":
            df.to_parquet(target_path, index=False)
        else:
            df.to_csv(target_path, index=False)
        return target_path
    except Exception as exc:  # pragma: no cover - best-effort fallback
        logger.warning("Falling back to CSV metadata due to %s", exc)
        csv_path = target_path.with_suffix(".csv")
        import csv

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return csv_path


def _ffmpeg_available() -> bool:
    return subprocess.call(["which", "ffmpeg"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) == 0

