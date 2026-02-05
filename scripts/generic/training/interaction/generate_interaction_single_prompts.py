#!/usr/bin/env python3
"""
Generate Pair-Friendly Single-Character Prompts (Reference-Aligned)
===================================================================

This generator is used to create a pair-friendly single-character image bank.

IMPORTANT: It aligns prompt/negative_prompt style with the existing synthetic
datasets under:
  /mnt/data/ai_data/synthetic_lora_data/generated_data/{character}/pose/

Why:
  - Those prompts have proven identity stability (face consistency)
  - The comprehensive negative prompt strongly suppresses multi-person outputs

Outputs (per character):
  prompts/interaction_single/{character}/prompts.json
  prompts/interaction_single/{character}/prompts_converted.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


FALLBACK_NEGATIVE_PROMPT = (
    "multiple people, two people, group, crowd, duplicate character, duplicate person, clone, second person, extra character, "
    "photograph, photo, realistic, photorealistic, real person, real life, live action, hyperrealistic, real human, candid photo, "
    "portrait photo, stock photo, adult, elderly, old person, mature, grown-up, old man, baby, toddler, infant, young child, girl, "
    "female, woman, extra limbs, extra arms, extra legs, extra hands, extra fingers, missing limbs, missing arms, missing legs, missing hands, "
    "missing fingers, deformed, disfigured, distorted, malformed, mutated, mutation, bad anatomy, wrong anatomy, anatomically incorrect, blurry, "
    "out of focus, unfocused, fuzzy, hazy, soft focus, low quality, bad quality, worst quality, low resolution, low res, jpeg artifacts, "
    "compression artifacts, pixelated, grainy, noisy, watermark, text, signature, username, artist name, bad proportions, gross proportions, "
    "unnatural proportions"
)


DEFAULT_STYLE = "3d animation, pixar style, high quality, detailed"


BACKGROUND_SIMPLE = [
    # Keep it easy to segment, but still aligned with existing 3D prompt wording.
    "professional studio environment with seamless backdrop",
    "plain white background",
    "simple gradient background",
    "clean studio backdrop",
    "minimal studio background",
]


CAMERA = [
    "front view, eye level",
    "three-quarter view, eye level",
    "left side view, profile, eye level",
    "right side view, profile, eye level",
]


FRAMING = [
    "full body",
    "three-quarter body",
]


LIGHTING = [
    "three-point lighting setup with soft key light, balanced fill light, subtle rim light",
    "even professional studio lighting with balanced key and fill lights",
    "soft diffused studio lighting with gentle shadows",
]


COMPOSITION = [
    "center composition",
    "symmetrical composition",
]


PAIR_FRIENDLY_POSES = [
    # Neutral / cooperative
    "standing upright, neutral stance, arms relaxed at sides",
    "standing with hands on hips, confident posture",
    "standing with arms crossed, relaxed posture",
    "standing casually, weight shifted to one leg",
    "walking, mid-stride, natural gait",
    "sitting on bench, relaxed seated pose",
    "talking, conversational pose, subtle hand gesture",
    "pointing, one arm extended, clear pointing gesture",
    "waving, friendly greeting gesture",
    "celebrating, arms raised, cheerful body language",
]

SUFFIX = "pixar style 3d animated character"


def _emph(text: str, weight: float) -> str:
    if weight is None or weight <= 1.0:
        return text
    cleaned = text.strip().strip("()")
    return f"({cleaned}:{weight:.2f})"


@dataclass(frozen=True)
class PromptItem:
    prompt: str
    metadata: Dict[str, Any]


def _write_prompt_files(
    out_dir: Path,
    character: str,
    prompts: List[PromptItem],
    negative_prompt: str,
    generation_method: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts_json = {
        "character": character,
        "lora_type": "interaction_single",
        "num_prompts": len(prompts),
        "generation_method": generation_method,
        "prompts": [{"prompt": p.prompt, "metadata": p.metadata} for p in prompts],
    }
    converted = {"prompts": [p.prompt for p in prompts], "negative_prompt": negative_prompt}

    (out_dir / "prompts.json").write_text(json.dumps(prompts_json, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "prompts_converted.json").write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--character", required=True, help="Identity trigger token (e.g. alberto)")
    ap.add_argument("--num-prompts", type=int, default=40, help="Prompts per character (pair-friendly)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out-root", type=Path, default=Path("prompts/interaction_single"))
    ap.add_argument(
        "--reference-root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/generated_data"),
        help="Existing synthetic generated_data root used to pull style/prefix/negative_prompt.",
    )
    ap.add_argument(
        "--negative-prompt",
        default="",
        help="Override negative prompt (leave empty to use reference).",
    )
    ap.add_argument("--w-style", type=float, default=1.35)
    ap.add_argument("--w-pose", type=float, default=1.60)
    ap.add_argument("--w-camera", type=float, default=1.25)
    ap.add_argument("--w-framing", type=float, default=1.20)
    ap.add_argument("--w-bg", type=float, default=1.15)
    ap.add_argument("--w-light", type=float, default=1.15)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prompts: List[PromptItem] = []
    seen = set()

    # Reference-aligned prefix/style/negative prompt
    ref_pose_dir = args.reference_root / args.character / "pose"
    ref_prompts_json = ref_pose_dir / "prompts.json"
    ref_prompts_converted = ref_pose_dir / "prompts_converted.json"

    character_prefix = args.character
    style = DEFAULT_STYLE

    if ref_prompts_json.exists():
        try:
            data = json.loads(ref_prompts_json.read_text(encoding="utf-8"))
            first = data.get("prompts", [])[0]
            if isinstance(first, dict) and isinstance(first.get("metadata"), dict):
                style = str(first["metadata"].get("style") or style)
            if isinstance(first, dict) and isinstance(first.get("prompt"), str) and style:
                token = f", {style},"
                if token in first["prompt"]:
                    character_prefix = first["prompt"].split(token, 1)[0].strip()
        except Exception:
            pass

    negative_prompt = args.negative_prompt
    if not negative_prompt and ref_prompts_converted.exists():
        try:
            d = json.loads(ref_prompts_converted.read_text(encoding="utf-8"))
            negative_prompt = str(d.get("negative_prompt") or "")
        except Exception:
            negative_prompt = ""
    if not negative_prompt:
        negative_prompt = FALLBACK_NEGATIVE_PROMPT

    for _ in range(args.num_prompts * 50):
        pose = rng.choice(PAIR_FRIENDLY_POSES)
        bg = rng.choice(BACKGROUND_SIMPLE)
        cam = rng.choice(CAMERA)
        framing = rng.choice(FRAMING)
        light = rng.choice(LIGHTING)
        comp = rng.choice(COMPOSITION)

        prompt = ", ".join(
            [
                character_prefix,
                _emph(style, args.w_style),
                _emph(pose, args.w_pose),
                _emph(cam, args.w_camera),
                _emph(framing, args.w_framing),
                _emph(bg, args.w_bg),
                _emph(light, args.w_light),
                comp,
                "single character, solo",
                SUFFIX,
            ]
        )

        if prompt in seen:
            continue
        seen.add(prompt)
        prompts.append(
            PromptItem(
                prompt=prompt,
                metadata={
                    "pose": pose,
                    "camera": cam,
                    "framing": framing,
                    "background": bg,
                    "lighting": light,
                    "composition": comp,
                    "weights": {
                        "style": args.w_style,
                        "pose": args.w_pose,
                        "camera": args.w_camera,
                        "framing": args.w_framing,
                        "background": args.w_bg,
                        "lighting": args.w_light,
                    },
                },
            )
        )
        if len(prompts) >= args.num_prompts:
            break

    generation_method = "interaction_single_v2_reference_aligned"
    _write_prompt_files(args.out_root / args.character, args.character, prompts, negative_prompt, generation_method)
    print(f"Wrote {args.out_root / args.character / 'prompts_converted.json'} ({len(prompts)} prompts)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
