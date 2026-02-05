#!/usr/bin/env python3
"""
Generate a large pair-interaction dataset using an existing merged checkpoint.

Intent:
  - Use a strong identity checkpoint (e.g. trial1 merged) to synthesize many
    two-character interaction images.
  - Human then deletes low-quality / collage-like samples.
  - Later we train a small "pair-consistency" LoRA on the curated set.

Output layout:
  out_dir/
    meta.json
    index.csv
    pairs/
      p3d_a__p3d_b/
        img_000000.png
        img_000000.txt
        ...

Notes:
  - Prompts are kept short so key tokens are not truncated.
  - Designed to be restartable: existing images are skipped.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch


CHAR_TOKENS_DEFAULT = [
    "p3d_luca",
    "p3d_giulia",
    "p3d_alberto",
    "p3d_alberto_seamonster",
    "p3d_luca_seamonster",
    "p3d_miguel",
    "p3d_ian_lightfoot",
    "p3d_barley_lightfoot",
    "p3d_elio",
    "p3d_bryce",
    "p3d_caleb",
    "p3d_orion",
    "p3d_russell",
    "p3d_tyler",
]


INTERACTIONS = [
    "walking side by side, talking",
    "laughing together",
    "high-five, hands touching",
    "handshake, eye contact",
    "hugging, close contact",
    "passing a small object hand-to-hand, hands touching",
    "sitting at a table facing each other, talking",
    "running together, synchronized motion",
    "dancing together, coordinated pose",
    "back-to-back pose, confident",
]


DEFAULT_NEG = (
    "collage, cutout, pasted, sticker, composited, "
    "inconsistent lighting, mismatched shadows, different color temperature, "
    "harsh specular, blown highlights, overexposed, "
    "extra person, three people, crowd, group, "
    "extra limbs, extra fingers, blurry, low quality, worst quality, watermark, text"
)


def read_lines(path: Path) -> List[str]:
    items: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        items.append(s)
    return items


def all_pairs(tokens: Sequence[str]) -> List[Tuple[str, str]]:
    return list(combinations(tokens, 2))


def sanitize_pair_dir(a: str, b: str) -> str:
    return f"{a}__{b}".replace("/", "_")


def build_prompt(a: str, b: str, interaction: str) -> str:
    # Keep this intentionally short to reduce SDXL/CLIP truncation.
    return (
        f"{a}, {b}, p3d_style, two characters, {interaction}, "
        "both characters visible, single coherent scene, shared environment, "
        "consistent lighting, consistent shadows, matching color grading, "
        "same material response, medium wide shot, high quality 3d render"
    )


@dataclass(frozen=True)
class Meta:
    checkpoint: str
    out_dir: str
    characters: List[str]
    images_per_pair: int
    steps: int
    cfg: float
    seed: int
    width: int
    height: int
    created_at: str


def iter_jobs(pairs: Sequence[Tuple[str, str]], images_per_pair: int) -> Iterable[Tuple[str, str, int]]:
    for a, b in pairs:
        for k in range(images_per_pair):
            yield a, b, k


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Merged SDXL checkpoint (.safetensors), e.g. trial1 merged.",
    )
    ap.add_argument(
        "--out-dir",
        default="",
        help="Output directory (default: /mnt/data/.../pairs_generated_<ts>).",
    )
    ap.add_argument("--images-per-pair", type=int, default=50)
    ap.add_argument("--steps", type=int, default=45)
    ap.add_argument("--cfg", type=float, default=5.5)
    ap.add_argument("--seed", type=int, default=0, help="0 = random seed.")
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--negative", default=DEFAULT_NEG)
    ap.add_argument("--device", default="cuda", help="cuda/cpu.")
    ap.add_argument("--offload", action="store_true", help="Enable CPU offload (slower, lower VRAM).")
    ap.add_argument(
        "--characters-file",
        default="",
        help="Optional txt file of p3d_* tokens (one per line, # comments ok).",
    )
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise FileNotFoundError(ckpt)

    if args.images_per_pair < 1:
        raise ValueError("--images-per-pair must be >= 1")

    if args.seed == 0:
        args.seed = random.randint(1, 2**31 - 1)

    if args.characters_file:
        char_tokens = read_lines(Path(args.characters_file))
    else:
        char_tokens = CHAR_TOKENS_DEFAULT

    pairs = all_pairs(char_tokens)

    default_root = Path("/mnt/data/ai_data/synthetic_lora_data")
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else default_root / f"pairs_generated_trial1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    pairs_root = out_dir / "pairs"
    pairs_root.mkdir(parents=True, exist_ok=True)

    meta = Meta(
        checkpoint=str(ckpt),
        out_dir=str(out_dir),
        characters=list(char_tokens),
        images_per_pair=args.images_per_pair,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
        width=args.width,
        height=args.height,
        created_at=datetime.now().isoformat(),
    )
    (out_dir / "meta.json").write_text(json.dumps(asdict(meta), indent=2), encoding="utf-8")

    index_path = out_dir / "index.csv"
    index_exists = index_path.exists()

    # Diffusers pipeline
    from diffusers import DPMSolverMultistepScheduler, StableDiffusionXLPipeline

    if args.device == "cuda" and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    pipe = StableDiffusionXLPipeline.from_single_file(
        str(ckpt),
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        use_safetensors=True,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config,
        algorithm_type="dpmsolver++",
        use_karras_sigmas=True,
    )
    pipe.set_progress_bar_config(disable=True)

    device = args.device
    if args.offload and torch.cuda.is_available():
        pipe.enable_model_cpu_offload()
        device = "cuda"
    else:
        # Try to fit on GPU; if this OOMs later we will fall back to offload per-call.
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()

    rng = random.Random(args.seed)

    # Open index for append
    with index_path.open("a", newline="", encoding="utf-8") as f_index:
        writer = csv.DictWriter(
            f_index,
            fieldnames=[
                "rel_path",
                "pair",
                "a",
                "b",
                "k",
                "seed",
                "steps",
                "cfg",
                "prompt",
                "negative",
            ],
        )
        if not index_exists:
            writer.writeheader()

        total = len(pairs) * args.images_per_pair
        done = 0

        for a, b, k in iter_jobs(pairs, args.images_per_pair):
            pair_dir = pairs_root / sanitize_pair_dir(a, b)
            pair_dir.mkdir(parents=True, exist_ok=True)
            name = f"img_{k:06d}"
            out_png = pair_dir / f"{name}.png"
            out_txt = pair_dir / f"{name}.txt"

            # Restartable: skip if already exists
            if out_png.exists() and out_txt.exists():
                done += 1
                continue

            interaction = INTERACTIONS[(hash((a, b, k)) + k) % len(INTERACTIONS)]
            prompt = build_prompt(a, b, interaction=interaction)

            # Per-image seed derived from base seed + stable offsets
            seed = (args.seed + (hash((a, b)) & 0x7FFFFFFF) + k) & 0x7FFFFFFF
            gen = torch.Generator(device=device).manual_seed(seed)

            # Generate with OOM fallback to offload
            while True:
                try:
                    image = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative,
                        num_inference_steps=args.steps,
                        guidance_scale=args.cfg,
                        height=args.height,
                        width=args.width,
                        generator=gen,
                    ).images[0]
                    break
                except torch.cuda.OutOfMemoryError:
                    if args.offload:
                        raise
                    torch.cuda.empty_cache()
                    pipe.enable_model_cpu_offload()
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    gen = torch.Generator(device=device).manual_seed(seed)
                    args.offload = True

            image.save(out_png)
            out_txt.write_text(prompt + "\n", encoding="utf-8")

            writer.writerow(
                {
                    "rel_path": str(out_png.relative_to(out_dir)),
                    "pair": sanitize_pair_dir(a, b),
                    "a": a,
                    "b": b,
                    "k": k,
                    "seed": seed,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "prompt": prompt,
                    "negative": args.negative,
                }
            )
            f_index.flush()

            done += 1
            if done % 25 == 0:
                print(f"[gen] {done}/{total} ({done/total:.1%})", flush=True)

    print(f"Done: {out_dir}")
    print(f"Seed: {args.seed}")


if __name__ == "__main__":
    main()

