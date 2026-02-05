#!/usr/bin/env python3
"""
Smoke test for a merged P3D multi-character SDXL checkpoint.

Loads a single SDXL checkpoint (.safetensors) and generates a small set of images
from a prompt list, saving outputs for quick visual validation.
"""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List

import torch

logger = logging.getLogger(__name__)


def load_prompts(path: Path, limit: int) -> List[str]:
    prompts: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        prompts.append(s)
        if limit > 0 and len(prompts) >= limit:
            break
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Path to merged SDXL checkpoint (.safetensors).")
    ap.add_argument(
        "--prompts",
        default="prompts/p3d/p3d_multichar_validation_prompts.txt",
        help="Prompt file (one per line).",
    )
    ap.add_argument("--out-dir", default="", help="Output directory (default: outputs/p3d_checkpoint_smoke_<ts>).")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=12, help="Max prompts to run (0 = all).")
    ap.add_argument(
        "--negative",
        default="multiple people, extra limbs, blurry, low quality, worst quality, watermark, text",
    )
    ap.add_argument("--device", default="cuda", help="Generation device (cuda/cpu).")
    ap.add_argument(
        "--offload",
        action="store_true",
        help="Enable Diffusers CPU offload (slower but lower VRAM).",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"p3d_checkpoint_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path, limit=args.limit)
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_path}")

    logger.info("Loading SDXL checkpoint: %s", ckpt)
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(ckpt),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.set_progress_bar_config(disable=True)

    if args.offload and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        device = "cuda"
    else:
        device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
        pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)

    for i, prompt in enumerate(prompts):
        logger.info("Generating %d/%d", i + 1, len(prompts))
        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            height=1024,
            width=1024,
            generator=generator,
        ).images[0]
        out_path = out_dir / f"{i:03d}.png"
        image.save(out_path)
        (out_dir / f"{i:03d}.txt").write_text(prompt + "\n", encoding="utf-8")

    logger.info("Done. Outputs: %s", out_dir)


if __name__ == "__main__":
    main()
