"""
Stubbed inference for LoRA + ControlNet pose.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from anime_pipeline.utils.logging_utils import setup_logging
from anime_pipeline.utils.path_utils import ensure_dir


@dataclass
class InferenceConfig:
    pose_metadata_path: str = "metadata/poses.parquet"
    output_dir: str = "outputs/inference"
    output_metadata_path: str = "outputs/inference/metadata.parquet"
    image_size: int = 512
    num_samples: int = 4
    seed: int = 42
    prompt: str = "1girl, best quality, anime style"
    negative_prompt: str = ""
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    sd_model_path: Optional[str] = None  # e.g., /mnt/c/ai_models/stable-diffusion/sd15.safetensors
    controlnet_path: Optional[str] = None  # e.g., /mnt/c/ai_models/controlnet/pose
    lora_path: Optional[str] = None  # e.g., /mnt/c/ai_models/lora/character.safetensors
    scheduler: str = "dpmpp_2m"  # dpmpp_2m / euler / ddim ...
    device: str = "cpu"
    dtype: str = "fp16"  # fp16/bf16/fp32
    use_stub: bool = True
    log_dir: Optional[str] = "logs"


def run_inference(config: InferenceConfig, logger=None) -> List[Dict]:
    """
    Generate images conditioned on pose metadata. Falls back to stub if models are unavailable.
    """
    logger = logger or setup_logging("infer_lora_controlnet_pose", config.log_dir)
    poses = _load_rows(config.pose_metadata_path, logger)
    if not poses:
        logger.warning("No poses found at %s; generating empty output.", config.pose_metadata_path)
        return []

    out_dir = ensure_dir(config.output_dir)
    rng = random.Random(config.seed)
    records: List[Dict] = []

    pipeline = None
    if not config.use_stub:
        pipeline = _load_diffusers_pipeline(config, logger)
        if pipeline is None:
            logger.warning("Pipeline unavailable; falling back to stub.")

    for idx, pose in enumerate(poses[: config.num_samples]):
        img_path = out_dir / f"sample_{idx:04d}.png"
        if pipeline:
            image = _run_diffusers(pipeline, pose, config, seed=config.seed + idx, out_path=img_path, logger=logger)
        else:
            _write_stub_image(img_path, config.image_size, text=f"pose {pose.get('pose_id')}", rng=rng)
        records.append(
            {
                "sample_id": f"sample_{idx:04d}",
                "pose_id": pose.get("pose_id"),
                "pose_image_path": pose.get("pose_image_path"),
                "output_image_path": str(img_path),
            }
        )

    meta_path = _write_records(records, config.output_metadata_path, logger)
    logger.info("Generated %d samples -> %s", len(records), meta_path)
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


def _load_diffusers_pipeline(config: InferenceConfig, logger):
    try:
        import torch
        from diffusers import (  # type: ignore
            ControlNetModel,
            DPMSolverMultistepScheduler,
            EulerAncestralDiscreteScheduler,
            EulerDiscreteScheduler,
            StableDiffusionControlNetPipeline,
            StableDiffusionPipeline,
        )
    except Exception as exc:
        logger.warning("diffusers/torch not available: %s", exc)
        return None

    torch_dtype = _select_dtype(config.dtype, torch)
    if config.controlnet_path:
        controlnet = ControlNetModel.from_pretrained(config.controlnet_path, torch_dtype=torch_dtype)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            config.sd_model_path or config.controlnet_path,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
    else:
        if not config.sd_model_path:
            logger.warning("sd_model_path missing; cannot build pipeline.")
            return None
        pipe = StableDiffusionPipeline.from_pretrained(
            config.sd_model_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )

    scheduler_name = (config.scheduler or "dpmpp_2m").lower()
    if scheduler_name.startswith("dpm"):
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name.startswith("euler_a"):
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif scheduler_name.startswith("euler"):
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe.to(config.device)

    if config.lora_path:
        try:
            pipe.load_lora_weights(config.lora_path)
        except Exception as exc:
            logger.warning("Failed to load LoRA weights (%s); continuing without.", exc)
    return pipe


def _select_dtype(dtype_str: str, torch_module):
    dtype_str = dtype_str.lower()
    if dtype_str == "fp16":
        return torch_module.float16
    if dtype_str == "bf16":
        return torch_module.bfloat16
    return torch_module.float32


def _run_diffusers(pipe, pose: Dict, config: InferenceConfig, seed: int, out_path: Path, logger):
    generator = None
    try:
        import torch
        generator = torch.Generator(device=config.device).manual_seed(seed)
    except Exception:
        generator = None

    pose_image_path = pose.get("pose_image_path")
    if not pose_image_path or not Path(pose_image_path).exists():
        _write_stub_image(out_path, config.image_size, text="pose-missing", rng=random.Random(seed))
        return None

    with Image.open(pose_image_path) as pose_img:
        pose_img = pose_img.convert("RGB")
        result = pipe(
            prompt=config.prompt,
            image=pose_img,
            negative_prompt=config.negative_prompt,
            num_inference_steps=config.num_inference_steps,
            guidance_scale=config.guidance_scale,
            generator=generator,
        )
        image = result.images[0]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        image.save(out_path)
        return image


def _write_stub_image(path: Path, size: int, text: str, rng: random.Random) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        path.write_text(f"stub image: {text}", encoding="utf-8")
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
