#!/usr/bin/env python3
"""
Interaction Single (Green Screen) Orchestrator
==============================================

Automates building a pair-friendly single-character image bank for a set of characters:
  1) Generate prompts (green-screen, full/3-4 body, consistent camera)
  2) Generate images using SDXL + each character's identity LoRA

This is the recommended prep step for training a universal two-character interaction LoRA.

Resilience:
  - Orchestrator checkpoint tracks completed characters.
  - Image generation uses batch_image_generator.py which has its own checkpointing in output_dir/checkpoints.
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    base_model: Path
    identity_loras_dir: Path
    lora_filename_pattern: str
    conda_env: str

    characters: List[str]

    prompts_out_root: Path
    prompts_per_character: int
    prompts_seed: int
    prompt_weights: Dict[str, float]

    images_out_root: Path
    images_per_prompt: int
    num_inference_steps: int
    guidance_scale: float
    height: int
    width: int
    lora_weight: float
    device: str

    checkpoint_path: Path
    max_retries: int
    retry_delay_seconds: int

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            base_model=Path(raw["models"]["base_model"]),
            identity_loras_dir=Path(raw["models"]["identity_loras_dir"]),
            lora_filename_pattern=str(raw["models"].get("lora_filename_pattern", "BEST_{character}_lora_sdxl.safetensors")),
            conda_env=str(raw.get("conda", {}).get("env", "ai_env")),
            characters=[str(x) for x in raw["characters"]],
            prompts_out_root=Path(raw["prompts"]["out_root"]),
            prompts_per_character=int(raw["prompts"]["num_prompts"]),
            prompts_seed=int(raw["prompts"].get("seed", 1234)),
            prompt_weights=dict(raw["prompts"].get("weights", {})),
            images_out_root=Path(raw["images"]["out_root"]),
            images_per_prompt=int(raw["images"]["num_images_per_prompt"]),
            num_inference_steps=int(raw["images"]["num_inference_steps"]),
            guidance_scale=float(raw["images"]["guidance_scale"]),
            height=int(raw["images"]["height"]),
            width=int(raw["images"]["width"]),
            lora_weight=float(raw["images"].get("lora_weight", 1.0)),
            device=str(raw["images"].get("device", "cuda")),
            checkpoint_path=Path(raw["resilience"]["checkpoint_path"]),
            max_retries=int(raw["resilience"].get("max_retries", 2)),
            retry_delay_seconds=int(raw["resilience"].get("retry_delay_seconds", 60)),
        )


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"completed": {}, "created_at": time.time()}


def _save_checkpoint(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(cmd: List[str], env: Dict[str, str], max_retries: int, retry_delay: int, label: str) -> None:
    for attempt in range(1, max_retries + 2):
        try:
            LOGGER.info("[%s] %s", label, " ".join(cmd))
            subprocess.run(cmd, check=True, env=env)
            return
        except subprocess.CalledProcessError as e:
            if attempt >= max_retries + 1:
                raise
            LOGGER.warning("[%s] failed (attempt %d/%d): %s", label, attempt, max_retries + 1, e)
            time.sleep(retry_delay)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    cfg = Config.from_yaml(args.config)
    checkpoint = _load_checkpoint(cfg.checkpoint_path)

    import os
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cfg.prompts_out_root.mkdir(parents=True, exist_ok=True)
    cfg.images_out_root.mkdir(parents=True, exist_ok=True)

    for idx, character in enumerate(cfg.characters, start=1):
        if checkpoint.get("completed", {}).get(character) == "done":
            LOGGER.info("[%d/%d] %s already completed, skipping.", idx, len(cfg.characters), character)
            continue

        LOGGER.info("[%d/%d] Processing %s", idx, len(cfg.characters), character)

        # 1) Generate prompts
        prompts_dir = cfg.prompts_out_root / character
        prompt_seed = (cfg.prompts_seed ^ (hash(character) & 0xFFFFFFFF)) & 0x7FFFFFFF
        weights = cfg.prompt_weights
        cmd_prompts = [
            sys.executable,
            str(PROJECT_ROOT / "scripts/generic/training/interaction/generate_interaction_single_prompts.py"),
            "--character",
            character,
            "--num-prompts",
            str(cfg.prompts_per_character),
            "--seed",
            str(prompt_seed),
            "--out-root",
            str(cfg.prompts_out_root),
            "--w-style",
            str(weights.get("style", 1.35)),
            "--w-pose",
            str(weights.get("pose", 1.60)),
            "--w-camera",
            str(weights.get("camera", 1.25)),
            "--w-framing",
            str(weights.get("framing", 1.20)),
            "--w-bg",
            str(weights.get("background", 1.25)),
            "--w-light",
            str(weights.get("lighting", 1.15)),
        ]
        _run(cmd_prompts, env=env, max_retries=0, retry_delay=0, label=f"{character}:prompts")

        prompts_file = prompts_dir / "prompts_converted.json"
        if not prompts_file.exists():
            raise FileNotFoundError(f"Missing prompts file: {prompts_file}")

        # 2) Generate images (uses internal checkpoint in output-dir)
        lora_path = cfg.identity_loras_dir / cfg.lora_filename_pattern.format(character=character)
        if not lora_path.exists():
            raise FileNotFoundError(f"Missing LoRA file: {lora_path}")

        out_dir = cfg.images_out_root / character / "images"
        out_dir.mkdir(parents=True, exist_ok=True)

        # NOTE: prefer running with the current interpreter to avoid nested `conda run`
        # which can stall/hide output in long background jobs.
        cmd_images = [
            sys.executable,
            "-u",
            str(PROJECT_ROOT / "scripts/generic/training/batch_image_generator.py"),
            "--base-model",
            str(cfg.base_model),
            "--lora-paths",
            str(lora_path),
            "--lora-scales",
            str(cfg.lora_weight),
            "--prompts-file",
            str(prompts_file),
            "--output-dir",
            str(out_dir),
            "--num-images-per-prompt",
            str(cfg.images_per_prompt),
            "--steps",
            str(cfg.num_inference_steps),
            "--guidance-scale",
            str(cfg.guidance_scale),
            "--height",
            str(cfg.height),
            "--width",
            str(cfg.width),
            "--device",
            cfg.device,
        ]
        _run(
            cmd_images,
            env=env,
            max_retries=cfg.max_retries,
            retry_delay=cfg.retry_delay_seconds,
            label=f"{character}:images",
        )

        checkpoint.setdefault("completed", {})[character] = "done"
        checkpoint["updated_at"] = time.time()
        _save_checkpoint(cfg.checkpoint_path, checkpoint)
        LOGGER.info("[%d/%d] %s done.", idx, len(cfg.characters), character)

    LOGGER.info("All characters completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
