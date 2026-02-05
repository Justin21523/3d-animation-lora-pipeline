"""
Stub animation generator: produces simple frame sequences.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir
from anime_pipeline.inference.lora_controlnet import _load_diffusers_pipeline, _run_diffusers, _select_dtype


@dataclass
class AnimationConfig:
    pose_metadata_path: str = "metadata/poses.parquet"
    output_dir: str = "outputs/animation/frames"
    output_metadata_path: str = "outputs/animation/frames/metadata.parquet"
    frame_size: int = 512
    num_frames: int = 8
    seed: int = 123
    prompt: str = "1girl, best quality, anime style"
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    sd_model_path: Optional[str] = None
    controlnet_path: Optional[str] = None
    lora_path: Optional[str] = None
    animatediff_path: Optional[str] = None  # if provided, will use AnimateDiff pipeline
    motion_module_path: Optional[str] = None  # optional motion module for AnimateDiff
    motion_bucket_id: Optional[int] = None
    fps_id: Optional[int] = None
    interp_steps: Optional[int] = None  # frame interpolation steps inside AnimateDiff if supported
    cond_aug: Optional[float] = None
    scheduler: str = "dpmpp_2m"
    device: str = "cpu"
    dtype: str = "fp16"
    fps: int = 12
    video_path: Optional[str] = None  # if set, will attempt to mux frames into video
    use_stub: bool = True
    log_dir: Optional[str] = "logs"


def generate_animation(config: AnimationConfig, logger=None) -> List[Dict]:
    """
    Create animation frames; uses diffusers pipeline if available, else stub.
    """
    logger = logger or setup_logging("generate_animation", config.log_dir)
    poses = _load_rows(config.pose_metadata_path, logger)
    if not poses:
        logger.warning("No poses found at %s; generating empty animation.", config.pose_metadata_path)
        return []

    out_dir = ensure_dir(config.output_dir)
    rng = random.Random(config.seed)
    records: List[Dict] = []

    pipeline = None
    if not config.use_stub:
        if config.animatediff_path:
            pipeline = _load_animatediff_pipeline(config, logger)
        else:
            pipeline = _load_diffusers_pipeline(config, logger)
        if pipeline is None:
            logger.warning("Animation pipeline unavailable; using stub.")

    if pipeline and config.animatediff_path:
        frames = _run_animatediff_sequence(pipeline, poses, config, logger)
        for idx, frame_path in enumerate(frames):
            records.append(
                {
                    "frame_index": idx,
                    "pose_id": poses[idx % len(poses)].get("pose_id"),
                    "frame_path": str(frame_path),
                }
            )
    else:
        for idx in range(config.num_frames):
            pose = poses[idx % len(poses)]
            frame_path = out_dir / f"frame_{idx:04d}.png"
            if pipeline:
                _run_diffusers(
                    pipeline,
                    pose,
                    config,
                    seed=config.seed + idx,
                    out_path=frame_path,
                    logger=logger,
                )
            else:
                _write_stub_frame(frame_path, config.frame_size, text=f"frame {idx} pose {pose.get('pose_id')}", rng=rng)
            records.append(
                {
                    "frame_index": idx,
                    "pose_id": pose.get("pose_id"),
                    "frame_path": str(frame_path),
                }
            )

    if config.video_path:
        _try_mux_video(out_dir, config.video_path, config.fps, logger)

    meta_path = _write_records(records, config.output_metadata_path, logger)
    logger.info("Generated %d animation frames -> %s", len(records), meta_path)
    return records


def _load_rows(path: str | Path, logger) -> List[Dict]:
    path = Path(path)
    if not path.exists() and path.suffix == ".parquet":
        alt = path.with_suffix(".csv")
        if alt.exists():
            path = alt
    if not path.exists():
        return []
    try:
        import pandas as pd

        df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        return df.to_dict(orient="records")
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to load poses from %s due to %s", path, exc)
        return []


def _write_stub_frame(path: Path, size: int, text: str, rng: random.Random) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        path.write_text(f"stub frame: {text}", encoding="utf-8")
        return

    color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    img = Image.new("RGB", (size, size), color=color)
    draw = ImageDraw.Draw(img)
    draw.text((10, 10), text, fill=(255, 255, 255))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def _write_records(records: List[Dict], output_path: str | Path, logger) -> Path:
    target_path = Path(output_path)
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
    except Exception as exc:  # pragma: no cover
        logger.warning("Falling back to CSV due to %s", exc)
        import csv

        csv_path = target_path.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
        return csv_path


def _try_mux_video(frames_dir: Path, video_path: str | Path, fps: int, logger) -> None:
    video_path = Path(video_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_path = _which("ffmpeg")
    if not ffmpeg_path:
        logger.warning("ffmpeg not found; skipping video mux.")
        return
    cmd = [
        ffmpeg_path,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "frame_%04d.png"),
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    import subprocess

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("Muxed video to %s", video_path)
    except Exception as exc:
        logger.warning("Failed to mux video with ffmpeg: %s", exc)


def _which(cmd: str) -> Optional[str]:
    import shutil

    return shutil.which(cmd)


def _load_animatediff_pipeline(config: AnimationConfig, logger):
    try:
        import torch
        from diffusers import AnimateDiffPipeline, MotionAdapter  # type: ignore
    except Exception as exc:
        logger.warning("AnimateDiff not available: %s", exc)
        return None

    if not config.animatediff_path:
        logger.warning("animatediff_path missing; cannot build pipeline.")
        return None

    torch_dtype = _select_dtype(config.dtype, torch)
    adapter = None
    if config.motion_module_path:
        try:
            adapter = MotionAdapter.from_pretrained(config.motion_module_path, torch_dtype=torch_dtype)
        except Exception as exc:
            logger.warning("Failed to load motion module (%s); continuing without.", exc)
            adapter = None

    pipe = AnimateDiffPipeline.from_pretrained(
        config.animatediff_path,
        torch_dtype=torch_dtype,
        adapter=adapter,
    )
    pipe.to(config.device)
    return pipe


def _run_animatediff_sequence(pipe, poses: List[Dict], config: AnimationConfig, logger) -> List[Path]:
    generator = None
    try:
        import torch
        generator = torch.Generator(device=config.device).manual_seed(config.seed)
    except Exception:
        generator = None

    # Build conditioning frames list from poses (repeat/crop to num_frames)
    cond_frames = []
    for idx in range(config.num_frames):
        pose = poses[idx % len(poses)]
        pose_image_path = pose.get("pose_image_path")
        if not pose_image_path or not Path(pose_image_path).exists():
            cond_frames.append(None)
            continue
        with Image.open(pose_image_path) as pose_img:
            cond_frames.append(pose_img.convert("RGB").resize((config.frame_size, config.frame_size)))

    # If AnimateDiff supports directly passing num_frames, try once; fallback to per-frame.
    try:
        extra_kwargs = {
            "prompt": config.prompt,
            "negative_prompt": config.negative_prompt,
            "guidance_scale": config.guidance_scale,
            "num_inference_steps": config.num_inference_steps,
            "generator": generator,
            "conditioning_frames": cond_frames,
            "num_frames": config.num_frames,
        }
        if config.motion_bucket_id is not None:
            extra_kwargs["motion_bucket_id"] = config.motion_bucket_id
        if config.fps_id is not None:
            extra_kwargs["fps_id"] = config.fps_id
        if config.interp_steps is not None:
            extra_kwargs["interpolation_steps"] = config.interp_steps
        if config.cond_aug is not None:
            extra_kwargs["cond_aug"] = config.cond_aug

        result = pipe(**extra_kwargs)
        frames = []
        if hasattr(result, "frames") and result.frames:
            frames = result.frames
        elif hasattr(result, "images") and result.images:
            frames = result.images
        else:
            frames = []
    except Exception as exc:
        logger.warning("AnimateDiff failed (%s); falling back to stub frames.", exc)
        frames = []

    out_paths: List[Path] = []
    for idx in range(config.num_frames):
        frame_path = Path(config.output_dir) / f"frame_{idx:04d}.png"
        frame = frames[idx] if idx < len(frames) else None
        if frame is None:
            _write_stub_frame(frame_path, config.frame_size, text=f"animatediff-fallback {idx}", rng=random.Random(config.seed + idx))
        else:
            frame_path.parent.mkdir(parents=True, exist_ok=True)
            frame.save(frame_path)
        out_paths.append(frame_path)
    return out_paths
