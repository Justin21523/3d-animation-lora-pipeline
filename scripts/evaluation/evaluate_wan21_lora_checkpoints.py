#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# DiffSynth-Studio (external dependency used by this repo's Wan2.1 workflow)
from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
from diffsynth.utils.data import save_video


DEFAULT_NEGATIVE_PROMPT = (
    "overexposed, low quality, worst quality, jpeg artifacts, watermark, subtitle, text, logo, "
    "blurry, static, deformed, disfigured, bad hands, extra fingers, fused fingers, "
    "noisy background, crowd, wrong anatomy"
)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    prompt: str
    reference_video: str | None


def _load_samples_from_metadata(metadata_jsonl: Path) -> list[Sample]:
    samples: list[Sample] = []
    with metadata_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                raise ValueError(f"Invalid JSONL at {metadata_jsonl}:{idx}")
            prompt = (obj.get("prompt") or "").strip()
            if not prompt:
                continue
            file_name = (obj.get("file_name") or obj.get("video") or f"sample_{idx}").strip()
            sample_id = Path(file_name).stem
            ref = obj.get("video") or obj.get("file_name")
            samples.append(Sample(sample_id=sample_id, prompt=prompt, reference_video=ref))
    if not samples:
        raise ValueError(f"No usable samples found in: {metadata_jsonl}")
    return samples


def _discover_checkpoints(lora_dir: Path) -> list[Path]:
    ckpts = sorted(lora_dir.glob("epoch-*.safetensors"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found under: {lora_dir} (expected epoch-*.safetensors)")
    return ckpts


def _init_pipe(model_root: Path, tokenizer_path: Path, vram_limit_gb: float | None) -> WanVideoPipeline:
    vram_config: dict[str, Any] = {
        "offload_dtype": "disk",
        "offload_device": "disk",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cpu",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }

    model_configs = [
        ModelConfig(path=str(model_root / "diffusion_pytorch_model.safetensors"), **vram_config),
        ModelConfig(path=str(model_root / "models_t5_umt5-xxl-enc-bf16.pth"), **vram_config),
        ModelConfig(path=str(model_root / "Wan2.1_VAE.pth"), **vram_config),
    ]
    tok_cfg = ModelConfig(path=str(tokenizer_path))

    if vram_limit_gb is None:
        # Keep a small headroom to avoid fragmentation spikes.
        total_gb = torch.cuda.mem_get_info("cuda")[1] / (1024**3)
        vram_limit_gb = max(1.0, total_gb - 2.0)

    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=tok_cfg,
        vram_limit=vram_limit_gb,
    )


def _write_index_html(out_dir: Path, runs: list[dict[str, Any]]) -> None:
    rows = []
    for r in runs:
        ckpt = r["checkpoint"]
        sample_id = r["sample_id"]
        seed = r["seed"]
        prompt = r["prompt"]
        rel = r["rel_path"]
        rows.append(
            f"<tr><td><code>{ckpt}</code></td><td><code>{sample_id}</code></td>"
            f"<td>{seed}</td><td style='max-width:900px;white-space:pre-wrap'>{prompt}</td>"
            f"<td><video src='{rel}' controls loop muted style='max-width:480px'></video></td></tr>"
        )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'/>"
        "<title>Wan2.1 LoRA Evaluation</title>"
        "<style>body{font-family:system-ui,Segoe UI,Roboto,Arial;margin:24px}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}"
        "th{background:#f6f6f6;position:sticky;top:0}</style></head><body>"
        "<h1>Wan2.1 LoRA Evaluation</h1>"
        "<p>Generated videos for baseline and checkpoints.</p>"
        "<table><thead><tr><th>Checkpoint</th><th>Sample</th><th>Seed</th><th>Prompt</th><th>Video</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table></body></html>"
    )
    (out_dir / "index.html").write_text(html, encoding="utf-8")


def _video_to_pil_frames(pipe: WanVideoPipeline, video: Any):
    if isinstance(video, list):
        return video
    if isinstance(video, torch.Tensor):
        if not video.is_floating_point():
            # Some VAE decode paths return already-quantized uint8/int tensors.
            # Convert to float so einops reduce(mean) works, and keep identity scaling.
            video = video.to(device=pipe.device, dtype=torch.float32)
            return pipe.vae_output_to_video(video, min_value=0, max_value=255)
        # Some einops backends don’t support reduce(mean) for bf16, so cast up.
        if video.dtype in (torch.bfloat16, torch.float16):
            video = video.to(dtype=torch.float32)
        return pipe.vae_output_to_video(video)
    raise TypeError(f"Unsupported video type: {type(video)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Qualitative evaluation for Wan2.1 LoRA checkpoints (base vs epoch-*.safetensors).")
    parser.add_argument("--model-root", type=Path, required=True, help="Base Wan2.1 model directory (contains diffusion_pytorch_model.safetensors, models_t5..., Wan2.1_VAE.pth).")
    parser.add_argument("--tokenizer-path", type=Path, required=True, help="Tokenizer directory (umt5-xxl).")
    parser.add_argument("--lora-dir", type=Path, required=True, help="Directory containing epoch-*.safetensors.")
    parser.add_argument("--metadata-jsonl", type=Path, required=True, help="Dataset metadata.jsonl to sample prompts from.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for generated videos + index.html.")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for sampling prompts.")
    parser.add_argument("--base-seed", type=int, default=0, help="Seed used for generation (kept constant across checkpoints).")
    parser.add_argument("--include-base", action="store_true", help="Also generate baseline (no LoRA).")
    parser.add_argument("--alpha", type=float, default=1.0, help="LoRA alpha.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=448)
    parser.add_argument("--num-frames", type=int, default=9)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--quality", type=int, default=5, help="Video encoding quality (DiffSynth save_video).")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--tiled", action="store_true", help="Enable VAE tiling to reduce VRAM.")
    parser.add_argument("--tile-size", type=str, default="30,52", help="Tile size as 'H,W'.")
    parser.add_argument("--tile-stride", type=str, default="15,26", help="Tile stride as 'H,W'.")
    parser.add_argument(
        "--output-type",
        type=str,
        default="floatpoint",
        choices=("floatpoint", "quantized"),
        help="Pipeline output type. Use 'floatpoint' to avoid quantized reduce_mean issues.",
    )
    parser.add_argument("--vram-limit-gb", type=float, default=None, help="Override VRAM limit passed to WanVideoPipeline.")
    parser.add_argument("--max-checkpoints", type=int, default=0, help="If >0, only evaluate the newest N checkpoints.")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Keep tokenizer seq len low to match training cache (and avoid attention OOM).
    os.environ.setdefault("WAN21_TOKENIZER_SEQ_LEN", "32")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    samples_all = _load_samples_from_metadata(args.metadata_jsonl)
    rng = random.Random(args.seed)
    if args.num_samples >= len(samples_all):
        samples = samples_all
    else:
        samples = rng.sample(samples_all, args.num_samples)

    ckpts = _discover_checkpoints(args.lora_dir)
    if args.max_checkpoints and args.max_checkpoints > 0:
        ckpts = sorted(ckpts, key=lambda p: p.stat().st_mtime)[-args.max_checkpoints :]

    tile_size = tuple(int(x) for x in args.tile_size.split(","))
    tile_stride = tuple(int(x) for x in args.tile_stride.split(","))

    pipe = _init_pipe(args.model_root, args.tokenizer_path, args.vram_limit_gb)

    manifest: dict[str, Any] = {
        "model_root": str(args.model_root),
        "tokenizer_path": str(args.tokenizer_path),
        "lora_dir": str(args.lora_dir),
        "metadata_jsonl": str(args.metadata_jsonl),
        "num_samples": len(samples),
        "height": args.height,
        "width": args.width,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "cfg_scale": args.cfg_scale,
        "alpha": args.alpha,
        "checkpoints": [p.name for p in ckpts],
        "include_base": bool(args.include_base),
        "output_type": args.output_type,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    runs: list[dict[str, Any]] = []

    def run_one(checkpoint_name: str, lora_path: Path | None, sample: Sample, seed: int) -> None:
        if lora_path is None:
            pipe.clear_lora()
        else:
            pipe.clear_lora()
            pipe.load_lora(pipe.dit, str(lora_path), alpha=args.alpha, hotload=True)

        video = pipe(
            prompt=sample.prompt,
            negative_prompt=args.negative_prompt,
            seed=seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.steps,
            tiled=bool(args.tiled),
            tile_size=tile_size,
            tile_stride=tile_stride,
            output_type=args.output_type,
        )
        video = _video_to_pil_frames(pipe, video)
        ckpt_dir = args.out_dir / checkpoint_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        out_path = ckpt_dir / f"{sample.sample_id}_seed{seed}.mp4"
        save_video(video, str(out_path), fps=args.fps, quality=args.quality)
        runs.append(
            {
                "checkpoint": checkpoint_name,
                "sample_id": sample.sample_id,
                "seed": seed,
                "prompt": sample.prompt,
                "rel_path": str(out_path.relative_to(args.out_dir)).replace("\\", "/"),
            }
        )

    # Baseline first (optional)
    if args.include_base:
        for s in samples:
            run_one("base", None, s, args.base_seed)

    # Checkpoints
    for ckpt in ckpts:
        ckpt_name = ckpt.stem  # epoch-4
        for s in samples:
            run_one(ckpt_name, ckpt, s, args.base_seed)

    (args.out_dir / "runs.json").write_text(json.dumps(runs, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_index_html(args.out_dir, runs)
    print(f"✅ Evaluation complete: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
