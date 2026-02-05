#!/usr/bin/env python3
"""
Smoke test a P3D multi-character SDXL LoRA (without merging) using Diffusers.

This is useful because Kohya training-time samples can look grainy depending on
sampler/steps; this script lets you quickly check:
  - different LoRA strengths (e.g. 0.7 / 0.8 / 0.9)
  - different samplers/steps (DPM++ 2M Karras recommended)
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List

import torch


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
    ap.add_argument("--base", default="/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors")
    ap.add_argument("--lora", required=True)
    ap.add_argument("--prompts", default="prompts/p3d/p3d_multichar_validation_prompts.txt")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument(
        "--negative",
        default="multiple people, extra limbs, blurry, low quality, worst quality, watermark, text, jpeg artifacts",
    )
    args = ap.parse_args()

    base = Path(args.base)
    lora = Path(args.lora)
    prompts_path = Path(args.prompts)
    if not base.exists():
        raise FileNotFoundError(base)
    if not lora.exists():
        raise FileNotFoundError(lora)
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / f"p3d_lora_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(prompts_path, limit=args.limit)

    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(base),
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe.load_lora_weights(str(lora))

    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    for i, prompt in enumerate(prompts):
        image = pipe(
            prompt=prompt,
            negative_prompt=args.negative,
            num_inference_steps=args.steps,
            guidance_scale=args.cfg,
            height=1024,
            width=1024,
            generator=generator,
            cross_attention_kwargs={"scale": float(args.lora_scale)},
        ).images[0]
        out = out_dir / f"{i:03d}_scale{args.lora_scale:.2f}.png"
        image.save(out)
        (out_dir / f"{i:03d}.txt").write_text(prompt + "\n", encoding="utf-8")

    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()

