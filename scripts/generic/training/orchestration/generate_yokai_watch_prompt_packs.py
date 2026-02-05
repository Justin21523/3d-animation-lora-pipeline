#!/usr/bin/env python3
"""
Generate Yokai-Watch Prompt Packs (Batch)
========================================

Batch-generate pose/action/expression prompt packs for all Yokai-Watch identity LoRAs.

By default it:
  - Scans `/mnt/c/ai_models/lora_sdxl/yokai-watch/` for character folders (52)
  - Reads optional profile tags from `/mnt/data/datasets/general/yokai-watch/sdxl_lora_datasets/{char}/meta/character_profile.json`
  - Writes output to `prompts/yokai-watch/generated/{char}/{type}/prompts*.json`

This does NOT generate images; it only generates prompt JSONs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import sys
import importlib.util

# Ensure repo root is on sys.path when running as a script
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


LOGGER = logging.getLogger(__name__)


def _read_profile_tags(profile_path: Path) -> Optional[str]:
    if not profile_path.exists():
        return None
    try:
        data = json.loads(profile_path.read_text(encoding="utf-8"))
    except Exception as e:
        LOGGER.warning("Failed reading %s: %s", profile_path, e)
        return None

    tags = data.get("tags")
    if not isinstance(tags, list):
        return None
    cleaned = [str(t).strip() for t in tags if str(t).strip()]
    return ", ".join(cleaned) if cleaned else None


def _character_tokens_from_lora_dir(lora_root: Path) -> List[str]:
    if not lora_root.exists():
        raise FileNotFoundError(f"LoRA root not found: {lora_root}")
    return sorted([p.name for p in lora_root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _run_single(
    character_token: str,
    output_root: Path,
    vocab_root: Path,
    num_prompts: int,
    base_seed: int,
    negative_prompt: str,
    series_tags: str,
    emphasis_main: float,
    emphasis_style: float,
    emphasis_camera: float,
    emphasis_framing: float,
    emphasis_background: float,
    emphasis_lighting: float,
    generator_module: Any,
) -> None:
    import sys

    # Stable per-character seed to avoid identical packs across characters
    per_seed = (base_seed ^ (hash(character_token) & 0xFFFFFFFF)) & 0x7FFFFFFF

    argv = [
        "yokai_watch_prompt_generator.py",
        "--character-token",
        character_token,
        "--series-tags",
        series_tags,
        "--num-prompts",
        str(num_prompts),
        "--seed",
        str(per_seed),
        "--emphasis-main",
        str(emphasis_main),
        "--emphasis-style",
        str(emphasis_style),
        "--emphasis-camera",
        str(emphasis_camera),
        "--emphasis-framing",
        str(emphasis_framing),
        "--emphasis-background",
        str(emphasis_background),
        "--emphasis-lighting",
        str(emphasis_lighting),
        "--output-root",
        str(output_root),
        "--vocab-root",
        str(vocab_root),
        "--negative-prompt",
        negative_prompt,
        "--log-level",
        "ERROR",
    ]

    old_argv = sys.argv
    try:
        sys.argv = argv
        generator_module.main()
    finally:
        sys.argv = old_argv


def _load_generator_module(project_root: Path) -> Any:
    """
    Load the prompt generator module directly from its file path.

    This avoids importing `scripts.generic.training.orchestration` package `__init__`,
    which may import optional heavy deps not required for prompt generation.
    """
    module_path = project_root / "scripts" / "generic" / "training" / "orchestration" / "yokai_watch_prompt_generator.py"
    spec = importlib.util.spec_from_file_location("yokai_watch_prompt_generator", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec: {module_path}")
    module = importlib.util.module_from_spec(spec)
    # Required for dataclasses/type resolution during exec_module
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-generate Yokai-Watch prompt packs for all characters.")
    parser.add_argument(
        "--lora-root",
        type=Path,
        default=Path("/mnt/c/ai_models/lora_sdxl/yokai-watch"),
        help="Directory containing per-character identity LoRA folders.",
    )
    parser.add_argument(
        "--profiles-root",
        type=Path,
        default=Path("/mnt/data/datasets/general/yokai-watch/sdxl_lora_datasets"),
        help="Directory containing per-character meta/character_profile.json (optional).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prompts/yokai-watch/generated"),
        help="Output root directory.",
    )
    parser.add_argument(
        "--vocab-root",
        type=Path,
        default=Path("prompts/generation/vocabulary"),
        help="Vocabulary directory containing poses.yaml/actions.yaml/expressions.yaml.",
    )
    parser.add_argument("--num-prompts", type=int, default=50, help="Prompts per type.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed.")
    parser.add_argument("--emphasis-main", type=float, default=1.50, help="Weight for key pose/action/expression phrases.")
    parser.add_argument("--emphasis-style", type=float, default=1.35, help="Weight for anime style adherence phrases.")
    parser.add_argument("--emphasis-camera", type=float, default=1.25, help="Weight for camera angle phrases.")
    parser.add_argument("--emphasis-framing", type=float, default=1.20, help="Weight for framing phrases.")
    parser.add_argument("--emphasis-background", type=float, default=1.15, help="Weight for background phrases.")
    parser.add_argument("--emphasis-lighting", type=float, default=1.15, help="Weight for lighting phrases.")
    parser.add_argument(
        "--negative-prompt",
        default="",
        help="Negative prompt written into prompts_converted.json.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["pose", "action", "expression"],
        default=["pose", "action", "expression"],
        help="Which packs to generate per character.",
    )
    parser.add_argument(
        "--series-tags",
        default="anime, yokai watch style, clean lineart, cel shading",
        help="Fallback tags if no profile tags are found.",
    )
    parser.add_argument(
        "--prefer-profile-tags",
        action="store_true",
        help="If set, use tags from meta/character_profile.json as series-tags when available.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    gen_mod = _load_generator_module(PROJECT_ROOT)
    negative_prompt = args.negative_prompt or getattr(gen_mod, "NEGATIVE_PROMPT_ANIME")

    characters = _character_tokens_from_lora_dir(args.lora_root)
    LOGGER.info("Found %d characters under %s", len(characters), args.lora_root)

    for idx, token in enumerate(characters, start=1):
        if args.prefer_profile_tags:
            profile_tags = _read_profile_tags(args.profiles_root / token / "meta" / "character_profile.json")
        else:
            profile_tags = None

        series_tags = profile_tags or args.series_tags
        LOGGER.info("[%d/%d] Generating packs for %s", idx, len(characters), token)

        _run_single(
            character_token=token,
            output_root=args.output_root,
            vocab_root=args.vocab_root,
            num_prompts=args.num_prompts,
            base_seed=args.seed,
            negative_prompt=negative_prompt,
            series_tags=series_tags,
            emphasis_main=args.emphasis_main,
            emphasis_style=args.emphasis_style,
            emphasis_camera=args.emphasis_camera,
            emphasis_framing=args.emphasis_framing,
            emphasis_background=args.emphasis_background,
            emphasis_lighting=args.emphasis_lighting,
            generator_module=gen_mod,
        )

    LOGGER.info("Done. Output: %s", args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
