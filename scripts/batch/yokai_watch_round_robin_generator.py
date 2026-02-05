#!/usr/bin/env python3
"""
Yokai-Watch Round-Robin Synthetic Image Generator
=================================================

Goal:
  - For each round, generate exactly 1 image per (character × type).
  - Characters: 52 (Yokai-Watch identity LoRAs)
  - Types: pose / action / expression (3)
  - Prompts per type: 50
  - Variants per prompt: 3

So the total target images are:
  52 * 3 * 50 * 3

Implementation:
  - Per (character/type), we track:
      prompt_index: 0..(N-1)
      variant_index: 0..(V-1)
    Each round generates one image for each (character/type), then advances
    variant_index; after V variants, increments prompt_index.
  - Uses a checkpoint JSON to resume exactly.

Notes:
  - This script expects prompt packs generated under:
      prompts_root/{Character}/{type}/prompts_converted.json
    (created by generate_yokai_watch_prompt_packs.py)
  - Identity LoRAs are expected under:
      lora_root/{Character}/{Character}_sdxl_lora.safetensors
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


LOGGER = logging.getLogger(__name__)


DEFAULT_TYPES = ["pose", "action", "expression"]


@dataclass
class JobConfig:
    base_model: Path
    lora_root: Path
    prompts_root: Path
    output_root: Path
    checkpoint_path: Path
    device: str
    dtype: str
    height: int
    width: int
    guidance_scale: float
    num_inference_steps: int
    lora_scale: float
    variants_per_prompt: int
    seed_base: int
    types: List[str]
    negative_prompt: Optional[str]


def _stable_int_hash(text: str) -> int:
    h = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _load_yaml(path: Path) -> Dict[str, Any]:
    import yaml

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _list_characters(lora_root: Path, explicit: Optional[List[str]] = None) -> List[str]:
    if explicit:
        return list(explicit)
    return sorted([p.name for p in lora_root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _lora_path_for(character: str, lora_root: Path, pattern: str) -> Path:
    return lora_root / character / pattern.format(character=character)


def _prompt_file_for(character: str, ptype: str, prompts_root: Path) -> Path:
    return prompts_root / character / ptype / "prompts_converted.json"


def _load_prompts(prompt_file: Path) -> Tuple[List[str], Optional[str]]:
    data = json.loads(prompt_file.read_text(encoding="utf-8"))
    prompts = data.get("prompts")
    if not isinstance(prompts, list) or not prompts:
        raise ValueError(f"Invalid prompts file: {prompt_file}")
    negative = data.get("negative_prompt")
    return [str(p) for p in prompts], (str(negative) if negative else None)


def _load_checkpoint(path: Path) -> Dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "current_round": 0,
        "current_char_index": 0,
        "current_type_index": 0,
        "progress": {},  # key -> {prompt_index, variant_index, generated}
        "total_generated": 0,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def _save_checkpoint(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _get_progress_entry(checkpoint: Dict[str, Any], key: str) -> Dict[str, int]:
    progress = checkpoint.setdefault("progress", {})
    entry = progress.get(key)
    if not isinstance(entry, dict):
        entry = {"prompt_index": 0, "variant_index": 0, "generated": 0}
        progress[key] = entry
    for k in ("prompt_index", "variant_index", "generated"):
        if k not in entry or not isinstance(entry[k], int) or entry[k] < 0:
            entry[k] = 0
    return entry


def _dtype_from_name(name: str) -> torch.dtype:
    if name.lower() in ("fp16", "float16"):
        return torch.float16
    if name.lower() in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def _ensure_diffusers_import_compat() -> None:
    """
    Work around known env mismatches where diffusers expects newer transformers
    symbols (e.g., SiglipImageProcessor) even if we don't use IP-Adapter.
    """
    try:
        import transformers  # type: ignore
    except Exception:
        return

    if not hasattr(transformers, "SiglipImageProcessor"):
        class SiglipImageProcessor:  # type: ignore
            pass

        transformers.SiglipImageProcessor = SiglipImageProcessor  # type: ignore[attr-defined]

    if not hasattr(transformers, "SiglipVisionModel"):
        class SiglipVisionModel:  # type: ignore
            pass

        transformers.SiglipVisionModel = SiglipVisionModel  # type: ignore[attr-defined]


def run(config: JobConfig, character_list: List[str], lora_filename_pattern: str, max_rounds: Optional[int]) -> None:
    LOGGER.info("Characters: %d", len(character_list))
    LOGGER.info("Types: %s", ", ".join(config.types))
    LOGGER.info("Prompts root: %s", config.prompts_root)
    LOGGER.info("LoRA root: %s", config.lora_root)
    LOGGER.info("Output root: %s", config.output_root)

    checkpoint = _load_checkpoint(config.checkpoint_path)
    start_round = int(checkpoint.get("current_round", 0))
    LOGGER.info("Resuming at round %d", start_round + 1)

    _ensure_diffusers_import_compat()
    from diffusers import StableDiffusionXLPipeline  # noqa: WPS433

    dtype = _dtype_from_name(config.dtype)
    pipeline = StableDiffusionXLPipeline.from_single_file(
        str(config.base_model),
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None,
    ).to(config.device)

    try:
        pipeline.enable_xformers_memory_efficient_attention()
        LOGGER.info("✓ Enabled xformers memory efficient attention")
    except Exception as e:
        LOGGER.warning("Could not enable xformers: %s", e)

    gen_params = {
        "height": config.height,
        "width": config.width,
        "guidance_scale": config.guidance_scale,
        "num_inference_steps": config.num_inference_steps,
    }

    # Cache prompt lists to avoid re-reading JSON every round
    prompt_cache: Dict[str, Dict[str, Any]] = {}

    def get_prompt_bundle(char: str, ptype: str) -> Dict[str, Any]:
        cache_key = f"{char}/{ptype}"
        if cache_key in prompt_cache:
            return prompt_cache[cache_key]
        pf = _prompt_file_for(char, ptype, config.prompts_root)
        prompts, neg_from_file = _load_prompts(pf)
        bundle = {"prompts": prompts, "negative_prompt": neg_from_file}
        prompt_cache[cache_key] = bundle
        return bundle

    # Precompute per-key targets (prompt_count * variants)
    targets: Dict[str, int] = {}
    for char in character_list:
        for ptype in config.types:
            pf = _prompt_file_for(char, ptype, config.prompts_root)
            if not pf.exists():
                raise FileNotFoundError(f"Missing prompts file: {pf}")
            prompts, _ = _load_prompts(pf)
            targets[f"{char}/{ptype}"] = len(prompts) * config.variants_per_prompt

    total_target = sum(targets.values())
    LOGGER.info("Target images: %d", total_target)

    current_lora_path: Optional[Path] = None

    def is_done() -> bool:
        for key, target in targets.items():
            entry = _get_progress_entry(checkpoint, key)
            if entry["generated"] < target:
                return False
        return True

    round_num = start_round
    while True:
        if max_rounds is not None and round_num >= max_rounds:
            LOGGER.info("Reached max_rounds=%d, stopping.", max_rounds)
            break
        if is_done():
            LOGGER.info("All images generated.")
            break

        LOGGER.info("=" * 72)
        LOGGER.info("ROUND %d", round_num + 1)
        LOGGER.info("=" * 72)

        char_start = int(checkpoint.get("current_char_index", 0))
        type_start = int(checkpoint.get("current_type_index", 0))

        for char_idx in range(char_start, len(character_list)):
            char = character_list[char_idx]

            lora_path = _lora_path_for(char, config.lora_root, lora_filename_pattern)
            if not lora_path.exists():
                raise FileNotFoundError(f"Missing LoRA file: {lora_path}")

            if current_lora_path != lora_path:
                if current_lora_path is not None:
                    try:
                        pipeline.unfuse_lora()
                        pipeline.unload_lora_weights()
                    except Exception:
                        pass
                LOGGER.info("Character %s: loading LoRA %s", char, lora_path.name)
                pipeline.load_lora_weights(str(lora_path))
                pipeline.fuse_lora(lora_scale=config.lora_scale)
                current_lora_path = lora_path

            for type_idx in range(type_start, len(config.types)):
                ptype = config.types[type_idx]
                key = f"{char}/{ptype}"

                target = targets[key]
                entry = _get_progress_entry(checkpoint, key)
                if entry["generated"] >= target:
                    continue

                bundle = get_prompt_bundle(char, ptype)
                prompts: List[str] = bundle["prompts"]

                prompt_index = entry["prompt_index"]
                variant_index = entry["variant_index"]
                if prompt_index >= len(prompts):
                    # Should not happen if generated==target, but keep safe.
                    entry["generated"] = target
                    continue

                prompt = prompts[prompt_index]
                negative_prompt = config.negative_prompt or bundle["negative_prompt"]

                # Deterministic variant seeds, stable across resumes.
                seed = (config.seed_base + _stable_int_hash(f"{char}|{ptype}|{prompt_index}|{variant_index}")) % 2_147_483_647
                generator = torch.Generator(device=config.device).manual_seed(seed)

                out_dir = config.output_root / char / ptype / "generated"
                out_dir.mkdir(parents=True, exist_ok=True)
                img_name = f"{ptype}_p{prompt_index:03d}_v{variant_index:02d}_r{round_num + 1:03d}_{seed}.png"
                out_path = out_dir / img_name

                LOGGER.info("%s: prompt %d/%d variant %d/%d", key, prompt_index + 1, len(prompts), variant_index + 1, config.variants_per_prompt)

                try:
                    output = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        generator=generator,
                        **gen_params,
                    )
                    image = output.images[0]
                    image.save(out_path, optimize=True)

                    entry["generated"] += 1
                    checkpoint["total_generated"] = int(checkpoint.get("total_generated", 0)) + 1

                    # Advance variant/prompt pointers
                    variant_index += 1
                    if variant_index >= config.variants_per_prompt:
                        variant_index = 0
                        prompt_index += 1

                    entry["prompt_index"] = prompt_index
                    entry["variant_index"] = variant_index

                    checkpoint["current_char_index"] = char_idx
                    checkpoint["current_type_index"] = type_idx + 1
                    _save_checkpoint(config.checkpoint_path, checkpoint)

                    del output
                    del image
                    if config.device.startswith("cuda"):
                        torch.cuda.empty_cache()

                except Exception as e:
                    LOGGER.error("Error generating %s: %s", key, e)
                    if config.device.startswith("cuda"):
                        torch.cuda.empty_cache()

            # next character starts at type index 0
            type_start = 0
            checkpoint["current_type_index"] = 0
            checkpoint["current_char_index"] = char_idx + 1
            _save_checkpoint(config.checkpoint_path, checkpoint)

        # round completed
        round_num += 1
        checkpoint["current_round"] = round_num
        checkpoint["current_char_index"] = 0
        checkpoint["current_type_index"] = 0
        checkpoint["updated_at"] = datetime.utcnow().isoformat() + "Z"
        _save_checkpoint(config.checkpoint_path, checkpoint)
        LOGGER.info("Round done. Total generated: %d / %d", checkpoint.get("total_generated", 0), total_target)


def _parse_config(path: Path) -> Tuple[JobConfig, List[str], str]:
    raw = _load_yaml(path)

    workspace_root = Path(raw["workspace"]["root"])
    prompts_root = Path(raw["workspace"]["prompts_root"])
    output_root = Path(raw["workspace"]["output_root"])
    checkpoint_path = Path(raw["workspace"]["checkpoint_path"])

    models = raw["models"]
    base_model = Path(models["base_model"])
    lora_root = Path(models["lora_root"])
    lora_pattern = models.get("lora_filename_pattern", "{character}_sdxl_lora.safetensors")

    gen = raw["generation"]
    types = raw.get("types", DEFAULT_TYPES)
    characters = raw.get("characters")  # optional explicit list

    cfg = JobConfig(
        base_model=base_model,
        lora_root=lora_root,
        prompts_root=prompts_root,
        output_root=output_root,
        checkpoint_path=checkpoint_path,
        device=str(gen.get("device", "cuda")),
        dtype=str(gen.get("dtype", "fp16")),
        height=int(gen.get("height", 1024)),
        width=int(gen.get("width", 1024)),
        guidance_scale=float(gen.get("guidance_scale", 7.5)),
        num_inference_steps=int(gen.get("num_inference_steps", 40)),
        lora_scale=float(gen.get("lora_scale", 1.0)),
        variants_per_prompt=int(gen.get("variants_per_prompt", 3)),
        seed_base=int(gen.get("seed_base", 1234)),
        types=[str(t) for t in types],
        negative_prompt=str(gen.get("negative_prompt")) if gen.get("negative_prompt") else None,
    )

    character_list = _list_characters(lora_root, explicit=characters)
    return cfg, character_list, lora_pattern


def main() -> int:
    parser = argparse.ArgumentParser(description="Yokai-Watch round-robin synthetic generator.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config path.")
    parser.add_argument("--max-rounds", type=int, default=None, help="Stop after N rounds (debug).")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))

    cfg, characters, lora_pattern = _parse_config(args.config)
    run(cfg, characters, lora_pattern, max_rounds=args.max_rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
