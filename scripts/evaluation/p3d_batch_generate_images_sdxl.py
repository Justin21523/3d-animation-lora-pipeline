#!/usr/bin/env python3
"""
Batch-generate SDXL images from a prompt list (one prompt per line).

Designed for long-running, restartable QA runs:
  - Skips existing outputs by index
  - Records per-image metadata (prompt/seed/params) to JSONL
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import torch


def read_prompts(path: Path) -> List[str]:
    prompts: List[str] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        prompts.append(line)
    return prompts


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


@dataclass(frozen=True)
class Job:
    index: int
    prompt: str
    seed: int


@dataclass(frozen=True)
class Record:
    index: int
    seed: int
    steps: int
    cfg: float
    width: int
    height: int
    checkpoint: str
    sampler: str
    negative_prompt: str
    prompt: str
    image_path: str
    prompt_path: str
    created_at: str


def iter_jobs(prompts: List[str], *, seed: int, start_index: int) -> Iterable[Job]:
    rng = random.Random(seed if seed >= 0 else None)
    for offset, prompt in enumerate(prompts):
        idx = start_index + offset
        if seed >= 0:
            job_seed = seed + idx  # stable, deterministic per index
        else:
            job_seed = rng.randint(1, 2**31 - 1)
        yield Job(index=idx, prompt=prompt, seed=job_seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="SDXL checkpoint (.safetensors).")
    ap.add_argument("--prompts", required=True, help="Prompt list file (one per line).")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    ap.add_argument("--negative", default="", help="Negative prompt string.")
    ap.add_argument("--negative-file", default="", help="Read negative prompt from file.")
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=-1, help="-1=random; >=0 deterministic.")
    ap.add_argument("--start-index", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="0=all")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--offload", action="store_true", help="Enable Diffusers CPU offload (slower, lower VRAM).")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if img_NNNNN.png exists.")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)
    prompts_path = Path(args.prompts)
    if not prompts_path.exists():
        raise FileNotFoundError(prompts_path)

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = (out_dir / "images").resolve()
    images_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_prompts(prompts_path)
    if args.limit and args.limit > 0:
        prompts = prompts[: args.limit]
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_path}")

    negative = args.negative.strip()
    if args.negative_file:
        negative = read_text(Path(args.negative_file))

    # Diffusers pipeline
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(ckpt),
        torch_dtype=torch.bfloat16 if (torch.cuda.is_available() and args.device == "cuda") else torch.float32,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    sampler_name = "dpmpp_2m_karras"
    pipe.set_progress_bar_config(disable=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    if args.offload and device == "cuda":
        pipe.enable_model_cpu_offload()
        gen_device = "cuda"
    else:
        pipe.to(device)
        gen_device = device
        if gen_device == "cuda":
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()

    meta_path = out_dir / "meta.json"
    meta = {
        "created_at": datetime.now().isoformat(),
        "checkpoint": str(ckpt),
        "prompts": str(prompts_path),
        "num_prompts": len(prompts),
        "steps": args.steps,
        "cfg": args.cfg,
        "width": args.width,
        "height": args.height,
        "seed_mode": args.seed,
        "sampler": sampler_name,
        "negative_prompt": negative,
        "offload": bool(args.offload),
        "device": device,
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    jsonl_path = out_dir / "records.jsonl"
    with jsonl_path.open("a", encoding="utf-8") as f_jsonl:
        for job in iter_jobs(prompts, seed=args.seed, start_index=args.start_index):
            img_path = images_dir / f"img_{job.index:05d}.png"
            txt_path = images_dir / f"img_{job.index:05d}.txt"
            if args.skip_existing and img_path.exists() and txt_path.exists():
                continue

            generator = torch.Generator(device=gen_device).manual_seed(int(job.seed))
            image = pipe(
                prompt=job.prompt,
                negative_prompt=negative if negative else None,
                num_inference_steps=args.steps,
                guidance_scale=args.cfg,
                height=args.height,
                width=args.width,
                generator=generator,
            ).images[0]

            # Be robust against external cleanup / cwd surprises in long nohup runs.
            img_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(img_path)
            txt_path.parent.mkdir(parents=True, exist_ok=True)
            txt_path.write_text(job.prompt + "\n", encoding="utf-8")

            rec = Record(
                index=job.index,
                seed=job.seed,
                steps=args.steps,
                cfg=float(args.cfg),
                width=args.width,
                height=args.height,
                checkpoint=str(ckpt),
                sampler=sampler_name,
                negative_prompt=negative,
                prompt=job.prompt,
                image_path=str(img_path),
                prompt_path=str(txt_path),
                created_at=datetime.now().isoformat(),
            )
            f_jsonl.write(json.dumps(asdict(rec), ensure_ascii=False) + "\n")
            f_jsonl.flush()


if __name__ == "__main__":
    main()
