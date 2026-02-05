#!/usr/bin/env python3
"""
Yokai-Watch Anime Prompt Pack Generator
======================================

Generates prompt packs for synthetic Pose / Action / Expression data generation
using an existing SDXL identity LoRA for a Yokai-Watch character.

Outputs:
  - prompts.json (rich format with metadata; compatible with the synthetic pipeline)
  - prompts_converted.json (simple list + top-level negative_prompt; compatible with batch_image_generator.py)

Design goals:
  - Anime/TV screenshot style (clean lineart, cel shading)
  - Single character only (avoid crowd/duplicates)
  - Prompts stay identity-light (use the trigger token + series/style tags)
  - Good diversity across camera, framing, background, and emotion intensity
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml


LOGGER = logging.getLogger(__name__)


NEGATIVE_PROMPT_ANIME = (
    "photograph, photo, realistic, photorealistic, 3d render, pixar style, cgi, "
    "live action, real person, real human, hyperrealistic, "
    "multiple people, two people, group, crowd, duplicate character, clone, twin, "
    "extra limbs, extra arms, extra legs, extra hands, extra fingers, missing limbs, "
    "bad anatomy, wrong anatomy, anatomically incorrect, deformed, disfigured, distorted, "
    "blurry, out of focus, low quality, worst quality, low resolution, jpeg artifacts, "
    "watermark, text, signature, username, logo, border, frame, cropped, cut off"
)


STYLE_VARIATIONS_ANIME: List[str] = [
    "anime style, yokai watch style, clean lineart, cel shading, crisp outlines, high quality",
    "tv anime screenshot, yokai watch style, clean lineart, cel shading, flat colors, sharp lines",
    "official anime key visual, yokai watch style, clean lineart, cel shading, vibrant flat colors",
    "anime illustration, yokai watch style, clean lineart, cel shading, soft gradients, high quality",
]


QUALITY_TAGS: List[str] = [
    "sharp focus",
    "clean edges",
    "consistent line weight",
    "no motion blur",
    "high prompt adherence",
    "character centered",
    "single character, solo",
]


BACKGROUND_VARIATIONS: List[str] = [
    "plain white background",
    "simple gradient background",
    "simple flat-color background",
    "minimal indoor background",
    "minimal outdoor background",
    "school hallway background",
    "classroom background",
    "residential street background",
    "park background",
    "shrine background",
    "night street background",
    "sunset sky background",
    "convenience store interior background",
    "bedroom interior background",
    "living room interior background",
    "rooftop background",
    "forest path background",
    "riverbank background",
    "festival street background",
    "playground background",
    "train station background",
    "city sidewalk background",
    "empty stage background",
]


LIGHTING_VARIATIONS: List[str] = [
    "daytime lighting",
    "soft indoor lighting",
    "warm sunset lighting",
    "cool nighttime lighting",
    "dramatic lighting",
    "even studio lighting",
    "overcast daylight lighting",
    "bright noon sunlight lighting",
    "neon night lighting",
    "backlit rim lighting",
    "soft bounce lighting",
    "high contrast lighting",
]

COMPOSITION_VARIATIONS: List[str] = [
    "center composition",
    "rule of thirds composition",
    "symmetrical composition",
    "slight tilt composition",
    "dynamic diagonal composition",
]

SHOT_MOOD_VARIATIONS: List[str] = [
    "calm atmosphere",
    "energetic atmosphere",
    "comedic atmosphere",
    "dramatic atmosphere",
    "mysterious atmosphere",
]


@dataclass(frozen=True)
class PromptItem:
    prompt: str
    metadata: Dict[str, Any]


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _iter_action_entries(tree: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(tree, dict):
        for value in tree.values():
            yield from _iter_action_entries(value)
    elif isinstance(tree, list):
        for item in tree:
            if isinstance(item, dict) and "name" in item:
                yield item
            else:
                yield from _iter_action_entries(item)


def _flatten_pose_names(poses_vocab: Dict[str, Any]) -> List[str]:
    body_poses = poses_vocab.get("body_poses", {})
    names: List[str] = []
    for group in body_poses.values():
        if isinstance(group, list):
            for entry in group:
                if isinstance(entry, dict) and "name" in entry:
                    names.append(str(entry["name"]))
        elif isinstance(group, dict):
            for subgroup in group.values():
                if isinstance(subgroup, list):
                    for entry in subgroup:
                        if isinstance(entry, dict) and "name" in entry:
                            names.append(str(entry["name"]))
    # Fallback minimal set
    return names or ["standing", "sitting", "walking", "running", "jumping"]


def _flatten_camera_combos(poses_vocab: Dict[str, Any]) -> List[str]:
    angles = poses_vocab.get("camera_angles", {})
    horizontal: List[str] = []
    vertical: List[str] = []

    for entry in angles.get("horizontal", []):
        if isinstance(entry, dict) and "name" in entry:
            horizontal.append(str(entry["name"]))
    for entry in angles.get("vertical", []):
        if isinstance(entry, dict) and "name" in entry:
            vertical.append(str(entry["name"]))

    if not horizontal:
        horizontal = ["front view", "three-quarter view", "side view"]
    if not vertical:
        vertical = ["eye level", "low angle", "high angle"]

    combos: List[str] = []
    for h in horizontal:
        for v in vertical:
            combos.append(f"{h}, {v}")
    return combos


def _flatten_framings(poses_vocab: Dict[str, Any]) -> List[str]:
    framings: List[str] = []
    for entry in poses_vocab.get("framing", []):
        if isinstance(entry, dict) and "name" in entry:
            framings.append(str(entry["name"]))
    return framings or ["full body", "three-quarter body", "medium shot", "close-up"]


def _flatten_actions(actions_vocab: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for entry in _iter_action_entries(actions_vocab):
        name = entry.get("name")
        if name:
            names.append(str(name))
    return names or ["walking", "running", "jumping", "waving", "talking"]


def _flatten_expressions(expressions_vocab: Dict[str, Any]) -> List[Tuple[str, List[str]]]:
    expressions: List[Tuple[str, List[str]]] = []
    for key in ("basic_emotions", "complex_emotions", "neutral_states"):
        for entry in expressions_vocab.get(key, []):
            if not isinstance(entry, dict):
                continue
            name = str(entry.get("name", "")).strip()
            if not name:
                continue
            intensity_mods = entry.get("intensity_modifiers") or []
            if not isinstance(intensity_mods, list):
                intensity_mods = []
            intensity_mods = [str(x) for x in intensity_mods if str(x).strip()]
            expressions.append((name, intensity_mods))
    return expressions or [("happy", ["slightly", "moderately", "very"]), ("sad", ["slightly", "moderately", "very"])]


def _choose(rng: random.Random, items: Sequence[str]) -> str:
    return items[rng.randrange(0, len(items))]

def _emph(text: str, weight: float) -> str:
    """
    SD-style emphasis using (text:weight).
    Keeps weight neutral at 1.0.
    """
    if weight is None or weight <= 1.0:
        return text
    # Avoid nested emphasis markers
    cleaned = text.strip().strip("()")
    return f"({cleaned}:{weight:.2f})"


def _build_character_prefix(
    trigger_token: str,
    series_tags: Optional[str],
) -> str:
    # Keep identity-light: trigger token + global series/style tags only.
    base = trigger_token.strip()
    if series_tags:
        return f"{base}, {series_tags.strip()}"
    return base


def _generate_pose_prompt(
    rng: random.Random,
    character_prefix: str,
    pose_names: Sequence[str],
    camera_combos: Sequence[str],
    framings: Sequence[str],
    weights: Dict[str, float],
) -> PromptItem:
    style = _choose(rng, STYLE_VARIATIONS_ANIME)
    pose = _choose(rng, pose_names)
    camera = _choose(rng, camera_combos)
    body_framings = [f for f in framings if f not in ("close-up", "extreme close-up")]
    framing = _choose(rng, body_framings or list(framings))
    background = _choose(rng, BACKGROUND_VARIATIONS)
    lighting = _choose(rng, LIGHTING_VARIATIONS)
    quality = _choose(rng, QUALITY_TAGS)
    composition = _choose(rng, COMPOSITION_VARIATIONS)
    mood = _choose(rng, SHOT_MOOD_VARIATIONS)

    style_e = _emph(style, weights["style"])
    pose_e = _emph(f"{pose} pose", weights["main"])
    camera_e = _emph(camera, weights["camera"])
    framing_e = _emph(framing, weights["framing"])
    background_e = _emph(background, weights["background"])
    lighting_e = _emph(lighting, weights["lighting"])

    prompt = (
        f"{character_prefix}, {style_e}, {pose_e}, {camera_e}, {framing_e}, "
        f"{background_e}, {lighting_e}, {composition}, {mood}, {quality}"
    )
    return PromptItem(
        prompt=prompt,
        metadata={
            "category": "pose",
            "style": style,
            "pose": pose,
            "camera": camera,
            "framing": framing,
            "background": background,
            "lighting": lighting,
            "composition": composition,
            "mood": mood,
            "quality": quality,
            "weights": weights,
        },
    )


def _generate_action_prompt(
    rng: random.Random,
    character_prefix: str,
    action_names: Sequence[str],
    camera_combos: Sequence[str],
    framings: Sequence[str],
    weights: Dict[str, float],
) -> PromptItem:
    style = _choose(rng, STYLE_VARIATIONS_ANIME)
    action = _choose(rng, action_names)
    camera = _choose(rng, camera_combos)
    body_framings = [f for f in framings if f not in ("close-up", "extreme close-up")]
    framing = _choose(rng, body_framings or list(framings))
    background = _choose(rng, BACKGROUND_VARIATIONS)
    lighting = _choose(rng, LIGHTING_VARIATIONS)
    quality = _choose(rng, QUALITY_TAGS)
    composition = _choose(rng, COMPOSITION_VARIATIONS)
    mood = _choose(rng, SHOT_MOOD_VARIATIONS)

    action_lower = action.lower()
    dynamic_actions = (
        "run",
        "jump",
        "kick",
        "throw",
        "catch",
        "dance",
        "climb",
        "crawl",
        "sprint",
        "leap",
        "fall",
        "fight",
        "punch",
        "push",
        "pull",
        "celebrat",
    )
    is_dynamic = any(key in action_lower for key in dynamic_actions)
    motion_tag = "dynamic action pose" if is_dynamic else "clear action pose"

    style_e = _emph(style, weights["style"])
    action_e = _emph(action, weights["main"])
    motion_e = _emph(motion_tag, weights["main"])
    camera_e = _emph(camera, weights["camera"])
    framing_e = _emph(framing, weights["framing"])
    background_e = _emph(background, weights["background"])
    lighting_e = _emph(lighting, weights["lighting"])

    prompt = (
        f"{character_prefix}, {style_e}, {action_e}, {motion_e}, {camera_e}, {framing_e}, "
        f"{background_e}, {lighting_e}, {composition}, {mood}, {quality}"
    )
    return PromptItem(
        prompt=prompt,
        metadata={
            "category": "action",
            "style": style,
            "action": action,
            "motion_tag": motion_tag,
            "camera": camera,
            "framing": framing,
            "background": background,
            "lighting": lighting,
            "composition": composition,
            "mood": mood,
            "quality": quality,
            "weights": weights,
        },
    )


def _generate_expression_prompt(
    rng: random.Random,
    character_prefix: str,
    expressions: Sequence[Tuple[str, List[str]]],
    weights: Dict[str, float],
) -> PromptItem:
    style = _choose(rng, STYLE_VARIATIONS_ANIME)
    expr_name, intensity_mods = expressions[rng.randrange(0, len(expressions))]
    intensity = _choose(rng, intensity_mods) if intensity_mods else _choose(rng, ["slightly", "moderately", "very"])

    expr_lower = expr_name.lower()
    if any(k in expr_lower for k in ("happy", "joy", "cheer", "delight", "excited", "proud")):
        eye_states = ["sparkling eyes", "wide eyes", "soft eyes"]
        mouth_states = ["smiling", "grinning", "open mouth"]
        brow_states = ["raised eyebrows", "relaxed brows"]
    elif any(k in expr_lower for k in ("sad", "melanchol", "downcast", "gloom", "cry")):
        eye_states = ["teary eyes", "eyes looking down", "soft eyes"]
        mouth_states = ["frowning", "closed mouth", "slightly open mouth"]
        brow_states = ["furrowed brows", "knitted brows"]
    elif any(k in expr_lower for k in ("angry", "furious", "mad", "irritat", "enrag")):
        eye_states = ["narrowed eyes", "side glance"]
        mouth_states = ["closed mouth", "slightly open mouth", "open mouth"]
        brow_states = ["furrowed brows", "knitted brows"]
    elif any(k in expr_lower for k in ("fear", "scared", "terrif", "alarm", "nervous", "anx")):
        eye_states = ["wide eyes", "side glance"]
        mouth_states = ["slightly open mouth", "open mouth", "closed mouth"]
        brow_states = ["raised eyebrows", "knitted brows"]
    elif any(k in expr_lower for k in ("surpris", "shock", "amaze", "startl")):
        eye_states = ["wide eyes", "sparkling eyes"]
        mouth_states = ["open mouth", "slightly open mouth"]
        brow_states = ["raised eyebrows"]
    elif any(k in expr_lower for k in ("disgust", "repuls", "nauseat")):
        eye_states = ["narrowed eyes", "squinting"]
        mouth_states = ["closed mouth", "pursed lips"]
        brow_states = ["furrowed brows"]
    elif any(k in expr_lower for k in ("neutral", "calm", "serene", "composed", "serious", "solemn", "stern")):
        eye_states = ["soft eyes", "side glance"]
        mouth_states = ["closed mouth"]
        brow_states = ["relaxed brows"]
    else:
        eye_states = ["wide eyes", "narrowed eyes", "soft eyes", "teary eyes", "sparkling eyes", "side glance"]
        mouth_states = ["closed mouth", "slightly open mouth", "open mouth", "smiling", "grinning", "frowning"]
        brow_states = ["raised eyebrows", "furrowed brows", "relaxed brows", "one raised eyebrow"]

    eyes = _choose(rng, eye_states)
    mouth = _choose(rng, mouth_states)
    brows = _choose(rng, brow_states)
    background = _choose(rng, ["plain white background", "simple gradient background", "simple flat-color background"])
    lighting = _choose(rng, LIGHTING_VARIATIONS)
    quality = _choose(rng, QUALITY_TAGS)
    mood = _choose(rng, SHOT_MOOD_VARIATIONS)

    style_e = _emph(style, weights["style"])
    expr_e = _emph(f"{intensity} {expr_name} expression", weights["main"])
    face_e = _emph(f"{eyes}, {mouth}, {brows}", weights["main"])
    background_e = _emph(background, weights["background"])
    lighting_e = _emph(lighting, weights["lighting"])

    prompt = (
        f"{character_prefix}, {style_e}, close-up portrait, {expr_e}, {face_e}, "
        f"{background_e}, {lighting_e}, {mood}, {quality}"
    )
    return PromptItem(
        prompt=prompt,
        metadata={
            "category": "expression",
            "style": style,
            "expression": expr_name,
            "intensity": intensity,
            "eyes": eyes,
            "mouth": mouth,
            "brows": brows,
            "background": background,
            "lighting": lighting,
            "mood": mood,
            "quality": quality,
            "weights": weights,
        },
    )


def _generate_unique(
    maker,
    target: int,
    rng: random.Random,
) -> List[PromptItem]:
    out: List[PromptItem] = []
    seen: set[str] = set()
    max_tries = target * 50

    for _ in range(max_tries):
        item: PromptItem = maker()
        if item.prompt in seen:
            continue
        seen.add(item.prompt)
        out.append(item)
        if len(out) >= target:
            break

    if len(out) < target:
        LOGGER.warning("Only generated %d/%d unique prompts", len(out), target)
    return out


def _write_prompt_files(
    output_dir: Path,
    character_token: str,
    lora_type: str,
    prompts: List[PromptItem],
    generation_method: str,
    negative_prompt: str,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts_json = {
        "character": character_token,
        "lora_type": lora_type,
        "num_prompts": len(prompts),
        "generation_method": generation_method,
        "prompts": [{"prompt": p.prompt, "metadata": p.metadata} for p in prompts],
    }

    prompts_converted = {
        "prompts": [p.prompt for p in prompts],
        "negative_prompt": negative_prompt,
    }

    prompts_path = output_dir / "prompts.json"
    converted_path = output_dir / "prompts_converted.json"

    prompts_path.write_text(json.dumps(prompts_json, ensure_ascii=False, indent=2), encoding="utf-8")
    converted_path.write_text(json.dumps(prompts_converted, ensure_ascii=False, indent=2), encoding="utf-8")

    return prompts_path, converted_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Yokai-Watch anime prompt packs (pose/action/expression).")
    parser.add_argument("--character-token", required=True, help="Trigger token used by the identity LoRA (e.g. Ash).")
    parser.add_argument(
        "--series-tags",
        default="anime, yokai watch style, clean lineart, cel shading",
        help="Global series/style tags to append after trigger token.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=100,
        help="Number of prompts per type.",
    )
    parser.add_argument(
        "--emphasis-main",
        type=float,
        default=1.50,
        help="Weight for key pose/action/expression phrases (>=1.0).",
    )
    parser.add_argument(
        "--emphasis-style",
        type=float,
        default=1.35,
        help="Weight for anime style adherence phrases (>=1.0).",
    )
    parser.add_argument(
        "--emphasis-camera",
        type=float,
        default=1.25,
        help="Weight for camera angle phrases (>=1.0).",
    )
    parser.add_argument(
        "--emphasis-framing",
        type=float,
        default=1.20,
        help="Weight for framing phrases (>=1.0).",
    )
    parser.add_argument(
        "--emphasis-background",
        type=float,
        default=1.15,
        help="Weight for background phrases (>=1.0).",
    )
    parser.add_argument(
        "--emphasis-lighting",
        type=float,
        default=1.15,
        help="Weight for lighting phrases (>=1.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducible prompt sets.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("prompts/yokai-watch/generated"),
        help="Output root directory (will create {character}/{type}/).",
    )
    parser.add_argument(
        "--vocab-root",
        type=Path,
        default=Path("prompts/generation/vocabulary"),
        help="Vocabulary directory containing poses.yaml/actions.yaml/expressions.yaml.",
    )
    parser.add_argument(
        "--negative-prompt",
        default=NEGATIVE_PROMPT_ANIME,
        help="Negative prompt written into prompts_converted.json.",
    )
    parser.add_argument(
        "--types",
        nargs="+",
        choices=["pose", "action", "expression"],
        default=["pose", "action", "expression"],
        help="Which packs to generate.",
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    poses_vocab = _load_yaml(args.vocab_root / "poses.yaml")
    actions_vocab = _load_yaml(args.vocab_root / "actions.yaml")
    expressions_vocab = _load_yaml(args.vocab_root / "expressions.yaml")

    pose_names = _flatten_pose_names(poses_vocab)
    camera_combos = _flatten_camera_combos(poses_vocab)
    framings = _flatten_framings(poses_vocab)
    action_names = _flatten_actions(actions_vocab)
    expressions = _flatten_expressions(expressions_vocab)

    rng = random.Random(args.seed)
    character_prefix = _build_character_prefix(args.character_token, args.series_tags)

    generation_method = "yokai_watch_anime_v1"
    weights = {
        "main": float(args.emphasis_main),
        "style": float(args.emphasis_style),
        "camera": float(args.emphasis_camera),
        "framing": float(args.emphasis_framing),
        "background": float(args.emphasis_background),
        "lighting": float(args.emphasis_lighting),
    }

    for lora_type in args.types:
        if lora_type == "pose":
            prompts = _generate_unique(
                maker=lambda: _generate_pose_prompt(
                    rng=rng,
                    character_prefix=character_prefix,
                    pose_names=pose_names,
                    camera_combos=camera_combos,
                    framings=framings,
                    weights=weights,
                ),
                target=args.num_prompts,
                rng=rng,
            )
        elif lora_type == "action":
            prompts = _generate_unique(
                maker=lambda: _generate_action_prompt(
                    rng=rng,
                    character_prefix=character_prefix,
                    action_names=action_names,
                    camera_combos=camera_combos,
                    framings=framings,
                    weights=weights,
                ),
                target=args.num_prompts,
                rng=rng,
            )
        else:
            prompts = _generate_unique(
                maker=lambda: _generate_expression_prompt(
                    rng=rng,
                    character_prefix=character_prefix,
                    expressions=expressions,
                    weights=weights,
                ),
                target=args.num_prompts,
                rng=rng,
            )

        out_dir = args.output_root / args.character_token / lora_type
        prompts_path, converted_path = _write_prompt_files(
            output_dir=out_dir,
            character_token=args.character_token,
            lora_type=lora_type,
            prompts=prompts,
            generation_method=generation_method,
            negative_prompt=args.negative_prompt,
        )

        LOGGER.info("Wrote %s (%d prompts)", prompts_path, len(prompts))
        LOGGER.info("Wrote %s", converted_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
