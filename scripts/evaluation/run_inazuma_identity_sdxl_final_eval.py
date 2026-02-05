#!/usr/bin/env python3
"""
Evaluate Inazuma Eleven SDXL identity LoRAs using the FINAL (last-epoch) weights.

Outputs:
  /mnt/data/training/lora/inazuma_eleven/eval/<run_tag>/
    - summary.json
    - report.md
    - <character_id>/
        - prompt01_baseline.png
        - prompt01_lora.png
        - ...
        - comparison_grid.png

Run (recommended):
  conda run -n kohya_ss python -m scripts.evaluation.run_inazuma_identity_sdxl_final_eval
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from diffusers import AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE_MODEL = Path("/mnt/c/ai_models/stable-diffusion/checkpoints/sd_xl_base_1.0.safetensors")
DEFAULT_VAE = Path("/mnt/c/ai_models/stable-diffusion/vae/sdxl_vae.safetensors")
DEFAULT_TRAIN_ROOT = Path("/mnt/data/training/lora/inazuma_eleven")
DEFAULT_PROMPT_DIR = REPO_ROOT / "prompts" / "inazuma"

CHARACTER_IDS: List[str] = [
    "endou_mamoru",
    "fudou_akio",
    "gouenji_shuuya",
    "inamori_asuto",
    "matsukaze_tenma",
    "nosaka_yuuma",
    "utsunomiya_toramaru",
]


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


def _read_last_avr_loss(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore")
    matches = re.findall(r"avr_loss=([0-9.]+)", text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _find_latest_train_log(log_dir: Path, character_id: str) -> Optional[Path]:
    if not log_dir.exists():
        return None
    candidates = sorted(log_dir.glob(f"train_{character_id}_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _parse_prompt_line(line: str) -> Optional[PromptSpec]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None

    def _take_arg(flag: str) -> Optional[str]:
        m = re.search(rf"\\s{re.escape(flag)}\\s+([^\\s].*?)(?=\\s--[a-zA-Z]\\b|$)", stripped)
        return m.group(1).strip() if m else None

    # Remove all "--x ..." chunks to get the raw prompt text
    prompt_text = re.split(r"\\s--[a-zA-Z]\\b", stripped, maxsplit=1)[0].strip()

    width = int(_take_arg("--w") or "1024")
    height = int(_take_arg("--h") or "1024")
    seed = int(_take_arg("--d") or "42")
    steps = int(_take_arg("--s") or "30")
    guidance = float(_take_arg("--l") or "7")
    negative = _take_arg("--n") or "worst quality, low quality, blurry, out of focus, text, watermark, logo"

    return PromptSpec(
        prompt=prompt_text,
        negative=negative,
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        guidance=guidance,
    )


def _load_prompts(prompt_file: Path) -> List[PromptSpec]:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    prompts: List[PromptSpec] = []
    for line in prompt_file.read_text(encoding="utf-8", errors="ignore").splitlines():
        spec = _parse_prompt_line(line)
        if spec:
            prompts.append(spec)
    if not prompts:
        raise ValueError(f"No prompts parsed from: {prompt_file}")
    return prompts


def _load_pipeline(device: str, base_model: Path, vae_path: Path, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    if not base_model.exists():
        raise FileNotFoundError(f"Base model not found: {base_model}")
    if not vae_path.exists():
        raise FileNotFoundError(f"VAE not found: {vae_path}")

    vae = AutoencoderKL.from_single_file(str(vae_path), torch_dtype=dtype)
    pipe = StableDiffusionXLPipeline.from_single_file(
        str(base_model),
        vae=vae,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    pipe.to(device)

    # VRAM helpers
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
    x = 20
    y = 20
    pad = 12
    draw.rectangle([x - pad, y - pad, x + tw + pad, y + th + pad], fill=(0, 0, 0, 200))
    draw.text((x, y), text, fill=(255, 255, 255), font=font)
    return out


def _grid(rows: List[Tuple[Image.Image, Image.Image]], left_label: str, right_label: str) -> Image.Image:
    # Each row: (baseline, lora)
    w, h = rows[0][0].size
    cols = 2
    grid = Image.new("RGB", (w * cols, h * len(rows)), color=(255, 255, 255))

    for r, (left, right) in enumerate(rows):
        grid.paste(_label(left, left_label), (0, r * h))
        grid.paste(_label(right, right_label), (w, r * h))

    return grid


def _generate(
    pipe: StableDiffusionXLPipeline,
    spec: PromptSpec,
    lora_scale: Optional[float],
) -> Image.Image:
    gen = torch.Generator(device=pipe.device).manual_seed(spec.seed)
    kwargs = {}
    if lora_scale is not None:
        kwargs["cross_attention_kwargs"] = {"scale": float(lora_scale)}

    result = pipe(
        prompt=spec.prompt,
        negative_prompt=spec.negative,
        num_inference_steps=spec.steps,
        guidance_scale=spec.guidance,
        width=spec.width,
        height=spec.height,
        generator=gen,
        **kwargs,
    )
    return result.images[0]


def _iter_characters(selected: Optional[Iterable[str]]) -> List[str]:
    if selected is None:
        return list(CHARACTER_IDS)
    wanted = [c.strip() for c in selected if c.strip()]
    unknown = [c for c in wanted if c not in CHARACTER_IDS]
    if unknown:
        raise ValueError(f"Unknown character ids: {unknown}")
    return wanted


def _write_report(run_dir: Path, summary: Dict) -> None:
    lines: List[str] = []
    lines.append("# Inazuma Eleven — SDXL Identity LoRA Evaluation (Final epoch)")
    lines.append("")
    lines.append(f"- Run: `{summary['run_tag']}`")
    lines.append(f"- Base model: `{summary['base_model']}`")
    lines.append(f"- VAE: `{summary['vae']}`")
    lines.append(f"- LoRA scale: `{summary['lora_scale']}`")
    lines.append("")

    for char in summary["characters"]:
        lines.append(f"## {char['character_id']}")
        lines.append(f"- Final LoRA: `{char['lora_path']}`")
        if char.get("final_avr_loss") is not None:
            lines.append(f"- Final avr_loss (from log): `{char['final_avr_loss']}`")
        lines.append(f"- Prompt file: `{char['prompt_file']}`")
        lines.append(f"- Grid: `{char['comparison_grid']}`")
        lines.append("")

    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    ap.add_argument("--prompt-dir", type=Path, default=DEFAULT_PROMPT_DIR)
    ap.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    ap.add_argument("--output-root", type=Path, default=DEFAULT_TRAIN_ROOT / "eval")
    ap.add_argument("--characters", type=str, default=None, help="Comma-separated character ids")
    ap.add_argument("--lora-scale", type=float, default=0.8)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "bf16"])
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    selected = None
    if args.characters:
        selected = [x.strip() for x in args.characters.split(",") if x.strip()]

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    run_tag = _now_tag()
    run_dir = args.output_root / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    pipe = _load_pipeline(args.device, args.base_model, args.vae, dtype=dtype)

    summary: Dict[str, object] = {
        "run_tag": run_tag,
        "created_at": datetime.now().isoformat(),
        "base_model": str(args.base_model),
        "vae": str(args.vae),
        "lora_scale": float(args.lora_scale),
        "dtype": args.dtype,
        "device": args.device,
        "characters": [],
    }

    for character_id in _iter_characters(selected):
        char_dir = run_dir / character_id
        char_dir.mkdir(parents=True, exist_ok=True)

        lora_path = args.train_root / character_id / f"inazuma_{character_id}_identity_sdxl_lora.safetensors"
        if not lora_path.exists():
            raise FileNotFoundError(f"Final LoRA not found for {character_id}: {lora_path}")

        prompt_file = args.prompt_dir / f"inazuma_{character_id}_sample_prompts.txt"
        prompts = _load_prompts(prompt_file)

        # Parse last-epoch loss from training log (if present)
        log_path = _find_latest_train_log(args.train_root / character_id / "logs", character_id)
        final_avr_loss = _read_last_avr_loss(log_path) if log_path else None

        # Baseline first (no LoRA)
        baseline_images: List[Image.Image] = []
        for i, spec in enumerate(prompts, 1):
            img = _generate(pipe, spec, lora_scale=None)
            out_path = char_dir / f"prompt{i:02d}_baseline.png"
            img.save(out_path)
            baseline_images.append(img)

        # LoRA images
        pipe.load_lora_weights(str(lora_path))
        lora_images: List[Image.Image] = []
        try:
            for i, spec in enumerate(prompts, 1):
                img = _generate(pipe, spec, lora_scale=args.lora_scale)
                out_path = char_dir / f"prompt{i:02d}_lora.png"
                img.save(out_path)
                lora_images.append(img)
        finally:
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

        rows = list(zip(baseline_images, lora_images))
        grid = _grid(rows, left_label="Baseline", right_label=f"LoRA x{args.lora_scale:g}")
        grid_path = char_dir / "comparison_grid.png"
        grid.save(grid_path)

        char_entry = {
            "character_id": character_id,
            "lora_path": str(lora_path),
            "prompt_file": str(prompt_file),
            "train_log": str(log_path) if log_path else None,
            "final_avr_loss": final_avr_loss,
            "comparison_grid": str(grid_path),
            "artifacts": {
                "baseline": [str(char_dir / f"prompt{i:02d}_baseline.png") for i in range(1, len(prompts) + 1)],
                "lora": [str(char_dir / f"prompt{i:02d}_lora.png") for i in range(1, len(prompts) + 1)],
            },
        }
        summary["characters"].append(char_entry)

        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        _write_report(run_dir, summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

