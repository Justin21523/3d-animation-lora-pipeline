#!/usr/bin/env python3
"""
Generate a "style generalization" dataset (no character tokens) for P3D.

Purpose:
  - Teach the model to produce new characters in the same Pixar/3D style
    using only the global style token (p3d_style), without any p3d_<character>.

Output (Kohya-friendly):
  out_dir/
    images/
      sample_000000.png
      sample_000000.txt
      ...
    prompts.json
    generation_meta.json

Captions:
  p3d_style, solo, a 3d animated character, <camera/pose/expression/scene/style...>

Notes:
  - Uses a merged SDXL checkpoint (base+LoRA) as the generator model.
  - Avoids overly "studio glossy / overexposed" look via negative prompt and
    conservative lighting descriptors.
  - For speed, default behavior keeps the model on GPU (no CPU offload) and
    supports batched generation. If you hit VRAM limits, enable --offload or
    reduce --batch-size.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import torch


STYLE_TOKEN = "p3d_style"


CAMERA = [
    "full body",
    "medium shot",
    "close-up portrait",
    "three-quarter view",
    "wide shot",
]

POSE = [
    "standing pose",
    "walking pose",
    "sitting pose",
    "running pose",
    "jumping pose",
    "dynamic action pose",
]

EXPR = [
    "neutral expression",
    "gentle smile",
    "happy expression",
    "surprised expression",
    "determined expression",
]

SCENE = [
    "clean background",
    "simple gradient background",
    "outdoors",
    "street scene",
    "park",
    "interior room",
]

LIGHTING = [
    "soft diffuse lighting",
    "balanced exposure",
    "cinematic lighting",
    "natural soft key light",
    "gentle rim light",
]

QUALITY = [
    "pixar style",
    "high quality 3d render",
    "smooth shading",
    "clean materials",
    "no harsh specular highlights",
]


DEFAULT_NEG = (
    "photograph, photo, realistic, photorealistic, real person, live action, "
    "anime, 2d, illustration, lineart, sketch, manga, "
    "overexposed, blown highlights, harsh specular, glossy plastic, metallic shine, "
    "blurry, out of focus, low quality, worst quality, noisy, grainy, jpeg artifacts, watermark, text, signature, "
    "multiple people, two people, group, crowd, extra limbs, extra fingers, deformed, bad anatomy"
)


@dataclass(frozen=True)
class GenMeta:
    checkpoint: str
    num_prompts: int
    images_per_prompt: int
    steps: int
    cfg: float
    height: int
    width: int
    seed: int
    created_at: str


def build_prompts(num_prompts: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    prompts: List[str] = []
    for _ in range(num_prompts):
        p = (
            f"{STYLE_TOKEN}, solo, a 3d animated character, "
            f"{rng.choice(CAMERA)}, {rng.choice(POSE)}, {rng.choice(EXPR)}, "
            f"{rng.choice(SCENE)}, {rng.choice(LIGHTING)}, {rng.choice(QUALITY)}"
        )
        prompts.append(p)
    return prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Merged SDXL checkpoint (.safetensors).")
    ap.add_argument("--out-dir", default="", help="Output directory (default: /mnt/data/.../style_generalization_<ts>).")
    ap.add_argument("--num-prompts", type=int, default=1200)
    ap.add_argument("--images-per-prompt", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=2, help="Images generated per pipeline call.")
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--cfg", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--device", default="cuda", help="Generation device (default: cuda).")
    ap.add_argument(
        "--offload",
        action="store_true",
        help="Enable Diffusers CPU offload (slower but lower VRAM).",
    )
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    default_root = Path("/mnt/data/ai_data/synthetic_lora_data")
    out_dir = Path(args.out_dir) if args.out_dir else default_root / f"style_generalization_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    prompts = build_prompts(num_prompts=args.num_prompts, seed=args.seed)
    (out_dir / "prompts.json").write_text(json.dumps({"prompts": prompts}, indent=2), encoding="utf-8")

    meta = GenMeta(
        checkpoint=str(ckpt),
        num_prompts=args.num_prompts,
        images_per_prompt=args.images_per_prompt,
        steps=args.steps,
        cfg=args.cfg,
        height=args.height,
        width=args.width,
        seed=args.seed,
        created_at=datetime.now().isoformat(),
    )
    (out_dir / "generation_meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(ckpt),
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.set_progress_bar_config(disable=True)

    if args.offload:
        pipe.enable_model_cpu_offload()
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        pipe.to(device)

    # Generate
    idx = 0
    base_seed = args.seed
    for p_i, prompt in enumerate(prompts):
        remaining = args.images_per_prompt
        while remaining > 0:
            bs = min(args.batch_size, remaining)
            while True:
                try:
                    generators = [torch.Generator(device=device).manual_seed(base_seed + idx + j) for j in range(bs)]
                    out = pipe(
                        prompt=[prompt] * bs,
                        negative_prompt=[args.negative] * bs,
                        num_inference_steps=args.steps,
                        guidance_scale=args.cfg,
                        height=args.height,
                        width=args.width,
                        generator=generators,
                    )
                    images = out.images
                    break
                except torch.cuda.OutOfMemoryError:
                    if args.offload:
                        raise
                    if bs == 1:
                        raise
                    torch.cuda.empty_cache()
                    bs = 1

            for j, image in enumerate(images):
                name = f"sample_{idx:06d}"
                out_png = images_dir / f"{name}.png"
                out_txt = images_dir / f"{name}.txt"
                image.save(out_png)
                out_txt.write_text(prompt + "\n", encoding="utf-8")
                idx += 1
                remaining -= 1

    print(f"Done: {out_dir}")


if __name__ == "__main__":
    main()
