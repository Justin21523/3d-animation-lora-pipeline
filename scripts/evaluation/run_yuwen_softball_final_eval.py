#!/usr/bin/env python3
"""
Evaluate final yuwen SDXL identity LoRA checkpoint with a softball throwing action suite.

Generates baseline (no LoRA) and LoRA images for each prompt, then writes:
  /mnt/data/training/lora/win-or-lose/yuwen_identity/eval/<run_tag>/
    - report.md
    - summary.json
    - promptXX_baseline.png
    - promptXX_lora.png
    - comparison_grid.png

Run:
  conda run -n kohya_ss python -m scripts.evaluation.run_yuwen_softball_final_eval
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]

BASE_MODEL = Path("/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors")
VAE_PATH = Path("/mnt/c/ai_models/stable-diffusion/vae/sdxl_vae.safetensors")
LORA_PATH = Path("/mnt/data/training/lora/win-or-lose/yuwen_identity/yuwen_identity_lora_sdxl.safetensors")
PROMPT_FILE = REPO_ROOT / "prompts" / "win_or_lose" / "yuwen_softball_throw_eval.txt"
OUTPUT_ROOT = Path("/mnt/data/training/lora/win-or-lose/yuwen_identity/eval")


@dataclass(frozen=True)
class PromptSpec:
    prompt: str
    negative: str
    width: int
    height: int
    seed: int
    steps: int
    guidance: float


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _parse_prompt_line(line: str) -> Optional[PromptSpec]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    def _take_arg(flag: str) -> Optional[str]:
        m = re.search(rf"\\s{re.escape(flag)}\\s+([^\\s].*?)(?=\\s--[a-zA-Z]\\b|$)", stripped)
        return m.group(1).strip() if m else None

    prompt_text = re.split(r"\\s--[a-zA-Z]\\b", stripped, maxsplit=1)[0].strip()

    width = int(_take_arg("--w") or "1024")
    height = int(_take_arg("--h") or "1024")
    seed = int(_take_arg("--d") or "42")
    steps = int(_take_arg("--s") or "35")
    guidance = float(_take_arg("--l") or "7")
    negative = _take_arg("--n") or "worst quality, low quality, blurry, text, watermark, logo, bad anatomy"

    return PromptSpec(
        prompt=prompt_text,
        negative=negative,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        guidance=guidance,
    )


def _load_prompts(path: Path) -> List[PromptSpec]:
    specs: List[PromptSpec] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        spec = _parse_prompt_line(line)
        if spec:
            specs.append(spec)
    if not specs:
        raise ValueError(f"No prompts parsed from {path}")
    return specs


def _font(size: int = 28) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def _label(img: Image.Image, text: str) -> Image.Image:
    out = img.copy()
    draw = ImageDraw.Draw(out)
    font = _font(28)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = 18, 18
    pad = 12
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0, 200))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return out


def _grid(rows: List[Tuple[Image.Image, Image.Image]], left_label: str, right_label: str) -> Image.Image:
    w, h = rows[0][0].size
    grid = Image.new("RGB", (w * 2, h * len(rows)), color=(255, 255, 255))
    for r, (left, right) in enumerate(rows):
        grid.paste(_label(left, left_label), (0, r * h))
        grid.paste(_label(right, right_label), (w, r * h))
    return grid


def _load_pipe(device: str, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    vae = AutoencoderKL.from_single_file(str(VAE_PATH), torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(BASE_MODEL),
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass
    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass
    try:
        pipe.enable_vae_tiling()
    except Exception:
        pass
    return pipe


def _generate(pipe: StableDiffusionXLPipeline, spec: PromptSpec, lora_scale: Optional[float]) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(spec.seed)
    extra = {}
    if lora_scale is not None:
        extra["cross_attention_kwargs"] = {"scale": float(lora_scale)}
    result = pipe(
        prompt=spec.prompt,
        negative_prompt=spec.negative,
        num_inference_steps=spec.steps,
        guidance_scale=spec.guidance,
        width=spec.width,
        height=spec.height,
        generator=gen,
        **extra,
    )
    return result.images[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    ap.add_argument("--prompt-file", type=Path, default=PROMPT_FILE)
    ap.add_argument("--lora", type=Path, default=LORA_PATH)
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="fp16")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if not args.prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {args.prompt_file}")
    if not args.lora.exists():
        raise FileNotFoundError(f"Final LoRA not found: {args.lora}")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    run_tag = _now_tag()
    run_dir = args.output_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    prompts = _load_prompts(args.prompt_file)
    pipe = _load_pipe(args.device, dtype=dtype)

    # Baseline
    baseline_imgs: List[Image.Image] = []
    for i, spec in enumerate(prompts, 1):
        img = _generate(pipe, spec, lora_scale=None)
        out = run_dir / f"prompt{i:02d}_baseline.png"
        img.save(out)
        baseline_imgs.append(img)

    # LoRA
    pipe.load_lora_weights(str(args.lora))
    lora_imgs: List[Image.Image] = []
    try:
        for i, spec in enumerate(prompts, 1):
            img = _generate(pipe, spec, lora_scale=args.lora_scale)
            out = run_dir / f"prompt{i:02d}_lora.png"
            img.save(out)
            lora_imgs.append(img)
    finally:
        try:
            pipe.unload_lora_weights()
        except Exception:
            pass

    grid = _grid(list(zip(baseline_imgs, lora_imgs)), "Baseline", f"LoRA x{args.lora_scale:g}")
    grid_path = run_dir / "comparison_grid.png"
    grid.save(grid_path)

    summary = {
        "run_tag": run_tag,
        "created_at": datetime.now().isoformat(),
        "base_model": str(BASE_MODEL),
        "vae": str(VAE_PATH),
        "lora": str(args.lora),
        "lora_scale": args.lora_scale,
        "prompt_file": str(args.prompt_file),
        "dtype": args.dtype,
        "device": args.device,
        "num_prompts": len(prompts),
        "comparison_grid": str(grid_path),
        "artifacts": [
            {
                "index": i,
                "prompt": prompts[i - 1].prompt,
                "negative": prompts[i - 1].negative,
                "seed": prompts[i - 1].seed,
                "steps": prompts[i - 1].steps,
                "guidance": prompts[i - 1].guidance,
                "baseline": str(run_dir / f"prompt{i:02d}_baseline.png"),
                "lora": str(run_dir / f"prompt{i:02d}_lora.png"),
            }
            for i in range(1, len(prompts) + 1)
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report_lines = [
        "# yuwen — Softball Throwing Evaluation (Final checkpoint)",
        "",
        f"- Run: `{run_tag}`",
        f"- LoRA: `{args.lora}`",
        f"- LoRA scale: `{args.lora_scale}`",
        f"- Prompt file: `{args.prompt_file}`",
        f"- Grid: `{grid_path}`",
        "",
    ]
    (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
