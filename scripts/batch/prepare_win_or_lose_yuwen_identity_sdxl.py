#!/usr/bin/env python3
"""
Prepare Win or Lose 'yuwen' (Pixar-style 3D) identity LoRA dataset for SDXL.

Phases:
  1) Preprocess: remove alpha halos via inpaint, composite onto solid gray,
     and letterbox to 1024x1024.
  2) Augment: generate 3D-safe geometric/light augmentations to reach a target count.
  3) Caption: per-image captions via OpenAI Vision (gpt-4o-mini by default),
     outputting comma-separated tags; prefixes with the trigger token first.
  4) Assemble: Kohya format under training_data_sdxl/<character>_identity/<repeats>_<token>/

Default input dir (provided by user):
  /mnt/data/datasets/general/win-or-lose/lora_data/characters/yuwen/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))


DEFAULT_INPUT_DIR = Path("/mnt/data/datasets/general/win-or-lose/lora_data/characters/yuwen")
DEFAULT_LORA_DATA_ROOT = Path("/mnt/data/datasets/general/win-or-lose/lora_data")


def _now_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _list_images(input_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    images = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(images)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _read_rgba(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        bgr = img
        alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
        img = np.concatenate([bgr, alpha], axis=2)
    if img.shape[2] != 4:
        raise ValueError(f"Unexpected channels={img.shape[2]} for {path}")
    return img  # BGRA uint8


def _inpaint_transparent_halo(bgra: np.ndarray, alpha_threshold: int = 4, radius: int = 3) -> np.ndarray:
    """
    Inpaint fully/mostly-transparent regions to reduce black/dirty fringes when resizing.
    """
    bgr = bgra[:, :, :3].copy()
    alpha = bgra[:, :, 3]
    mask = (alpha <= alpha_threshold).astype(np.uint8) * 255
    if mask.mean() < 0.1:
        return bgra
    inpainted = cv2.inpaint(bgr, mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)
    out = bgra.copy()
    out[:, :, :3] = inpainted
    return out


def _letterbox_composite_to_gray(
    bgra: np.ndarray,
    out_size: int,
    gray: int = 128,
    alpha_feather: int = 2,
) -> np.ndarray:
    """
    Composite RGBA onto solid gray and letterbox to out_size x out_size.

    Returns RGB uint8.
    """
    bgr = bgra[:, :, :3].astype(np.float32)
    alpha = bgra[:, :, 3].astype(np.float32) / 255.0  # HxW

    if alpha_feather > 0:
        k = alpha_feather * 2 + 1
        alpha = cv2.GaussianBlur(alpha, (k, k), sigmaX=0)

    h, w = alpha.shape
    scale = min(out_size / w, out_size / h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    bgr_resized = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    alpha_resized = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((out_size, out_size, 3), float(gray), dtype=np.float32)
    x0 = (out_size - new_w) // 2
    y0 = (out_size - new_h) // 2

    a3 = alpha_resized[:, :, None]
    canvas[y0 : y0 + new_h, x0 : x0 + new_w, :] = (
        bgr_resized * a3 + canvas[y0 : y0 + new_h, x0 : x0 + new_w, :] * (1.0 - a3)
    )

    rgb = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_BGR2RGB)
    return rgb


def _apply_3d_safe_augmentation(img: Image.Image, rng: random.Random) -> Image.Image:
    """
    Light augmentations intended to preserve 3D/PBR look.
    Assumes input is already 1024x1024 RGB on a uniform background.
    """
    aug = img.copy()
    w, h = aug.size

    # Small random crop (simulate framing variance)
    if rng.random() < 0.8:
        scale = rng.uniform(0.88, 1.0)
        new_w, new_h = int(w * scale), int(h * scale)
        left = rng.randint(0, w - new_w) if w > new_w else 0
        top = rng.randint(0, h - new_h) if h > new_h else 0
        aug = aug.crop((left, top, left + new_w, top + new_h)).resize((w, h), Image.LANCZOS)

    # Tiny rotation
    if rng.random() < 0.5:
        angle = rng.uniform(-3.0, 3.0)
        aug = aug.rotate(angle, resample=Image.BICUBIC, expand=False, fillcolor=(128, 128, 128))

    # Minimal brightness/contrast (safe range)
    if rng.random() < 0.6:
        aug = ImageEnhance.Brightness(aug).enhance(rng.uniform(0.93, 1.07))
    if rng.random() < 0.4:
        aug = ImageEnhance.Contrast(aug).enhance(rng.uniform(0.95, 1.05))

    return aug


def _write_json(path: Path, data: Dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass(frozen=True)
class PreparedPaths:
    prepared_dir: Path
    augmented_dir: Path
    captions_dir: Path
    training_dir: Path


def _resolve_paths(lora_data_root: Path, character_id: str) -> PreparedPaths:
    return PreparedPaths(
        prepared_dir=lora_data_root / "characters_prepared" / character_id / "1024_gray",
        augmented_dir=lora_data_root / "characters_augmented" / character_id,
        captions_dir=lora_data_root / "captions_output" / character_id,
        training_dir=lora_data_root / "training_data_sdxl" / f"{character_id}_identity",
    )

def _archive_move(path: Path, archive_root: Path, label: str) -> Optional[Path]:
    if not path.exists():
        return None
    archive_root.mkdir(parents=True, exist_ok=True)
    dst = archive_root / f"{label}_{_now_tag()}"
    shutil.move(str(path), str(dst))
    return dst


def preprocess_images(
    images: List[Path],
    out_dir: Path,
    out_size: int,
    gray: int,
    max_files: Optional[int],
) -> List[Path]:
    _ensure_dir(out_dir)
    out_paths: List[Path] = []

    subset = images[: max_files] if max_files else images
    for p in subset:
        bgra = _read_rgba(p)
        bgra = _inpaint_transparent_halo(bgra)
        rgb = _letterbox_composite_to_gray(bgra, out_size=out_size, gray=gray)

        out_path = out_dir / f"{p.stem}.png"
        Image.fromarray(rgb).save(out_path)
        out_paths.append(out_path)

    return out_paths


def augment_to_target(
    prepared_images: List[Path],
    out_dir: Path,
    target_count: int,
    seed: int,
) -> Dict[str, int]:
    _ensure_dir(out_dir)

    # Copy originals first
    originals = 0
    for p in prepared_images:
        shutil.copy2(p, out_dir / p.name)
        originals += 1

    needed = max(0, target_count - originals)
    if needed == 0:
        return {"original": originals, "augmented": 0, "total": originals}

    rng = random.Random(seed)
    augmented = 0

    # Deterministic selection: augment the first N images (stable across runs)
    for i in range(needed):
        src = prepared_images[i % len(prepared_images)]
        img = Image.open(src).convert("RGB")
        aug = _apply_3d_safe_augmentation(img, rng)
        out_name = f"{src.stem}_aug{i+1:04d}.png"
        aug.save(out_dir / out_name)
        augmented += 1

    return {"original": originals, "augmented": augmented, "total": originals + augmented}


def generate_captions_openai(
    images_dir: Path,
    captions_out_dir: Path,
    character_token: str,
    subject_tags: str,
    model: str,
    max_files: Optional[int],
    skip_existing: bool = True,
    overwrite_existing: bool = False,
) -> Dict[str, object]:
    """
    Uses OpenAI Vision to produce comma-separated SDXL-friendly tags.
    Requires OPENAI_API_KEY in env.
    """
    from generic.training.caption_engines.openai_api_engine import OpenAIAPICaptionEngine

    _ensure_dir(captions_out_dir)
    image_paths = sorted(images_dir.glob("*.png"))
    if max_files:
        image_paths = image_paths[:max_files]

    engine = OpenAIAPICaptionEngine(
        {
            "model_name": model,
            "max_tokens": 260,
            "schema_mode": False,
            "detail": "auto",
        }
    )

    banned_phrases = {
        "pixar style",
        "pixar-style",
        "pixar",
        "3d animated character",
        "3d animation",
        "3d character",
        character_token.lower(),
    }

    prompt = (
        "Generate a comma-separated caption for SDXL LoRA training.\n"
        "Rules:\n"
        "- Output ONLY comma-separated tags/phrases, no sentences.\n"
        "- Do NOT include any character names.\n"
        "- The subject is a 12-year-old boy character; use child proportions and avoid adult traits.\n"
        "- Focus on: hair, eyes, skin, facial features, clothing/accessories (cap, gloves, etc.), pose/action, expression,\n"
        "  camera framing (close-up/medium/full-body), and lighting.\n"
        "- Do NOT include these phrases (they will be added separately): 'pixar style', '3d animated character', the trigger token.\n"
        "- Avoid speculation. Be factual.\n"
        "- Keep it detailed (roughly 60-140 tokens).\n"
        "Example format:\n"
        "child boy, round face, big expressive eyes, short dark hair, baseball cap, yellow sports shirt, "
        "baseball glove, dynamic running pose, focused expression, medium shot, three-quarter view, soft studio lighting\n"
    )

    stats = {"total": 0, "ok": 0, "failed": 0, "model": model}

    prefix = ", ".join([character_token, subject_tags.strip(), "pixar style"]).strip().strip(",")

    for img_path in image_paths:
        stats["total"] += 1
        out_txt = captions_out_dir / f"{img_path.stem}.txt"
        if (not overwrite_existing) and skip_existing and out_txt.exists():
            try:
                if out_txt.read_text(encoding="utf-8").strip():
                    stats["ok"] += 1
                    continue
            except Exception:
                pass
        try:
            base = engine.generate_single(img_path, prompt=prompt)
            base = base.replace("\n", " ").strip().strip(",")
            # Post-process: split, strip, drop banned phrases, dedupe
            tags: List[str] = []
            seen = set()
            for t in [x.strip() for x in base.split(",") if x.strip()]:
                tl = t.lower()
                if tl in banned_phrases:
                    continue
                if tl.startswith(character_token.lower()):
                    continue
                if tl in seen:
                    continue
                seen.add(tl)
                tags.append(t)

            caption = prefix
            if tags:
                caption = caption + ", " + ", ".join(tags)
            out_txt.write_text(caption + "\n", encoding="utf-8")
            stats["ok"] += 1
        except Exception as e:
            out_txt.write_text(prefix + "\n", encoding="utf-8")
            stats["failed"] += 1
            (captions_out_dir / "caption_errors.log").open("a", encoding="utf-8").write(
                f"{img_path.name}\t{repr(e)}\n"
            )

    return stats


def assemble_kohya_training_dir(
    augmented_dir: Path,
    captions_dir: Path,
    training_dir: Path,
    repeats: int,
    token: str,
) -> Path:
    out = training_dir / f"{repeats}_{token}"
    if out.exists():
        shutil.rmtree(out)
    _ensure_dir(out)

    images = sorted(augmented_dir.glob("*.png"))
    for img_path in images:
        # For augmented variants, reuse the original caption:
        #   foo.png -> foo.txt
        #   foo_aug0001.png -> foo.txt
        stem = img_path.stem
        base_stem = stem.split("_aug", 1)[0]
        cap_path = captions_dir / f"{base_stem}.txt"
        if not cap_path.exists():
            raise FileNotFoundError(f"Missing base caption for {img_path.name}: {cap_path}")
        shutil.copy2(img_path, out / img_path.name)
        # Write caption matching the image stem (Kohya expects same basename).
        caption_text = cap_path.read_text(encoding="utf-8", errors="ignore").strip()
        (out / f"{stem}.txt").write_text(caption_text + "\n", encoding="utf-8")

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    ap.add_argument("--lora-data-root", type=Path, default=DEFAULT_LORA_DATA_ROOT)
    ap.add_argument("--character-id", type=str, default="yuwen")
    ap.add_argument("--character-token", type=str, default="yuwen")
    ap.add_argument("--out-size", type=int, default=1024)
    ap.add_argument("--gray", type=int, default=128)
    ap.add_argument("--target-count", type=int, default=220)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--repeats", type=int, default=8)
    ap.add_argument("--max-files", type=int, default=None, help="Limit images for quick testing")
    ap.add_argument("--clean", action="store_true", help="Archive previous prepared/augmented/training dirs before running")

    ap.add_argument("--run-captions", action="store_true", help="Call OpenAI Vision to generate captions")
    ap.add_argument("--openai-model", type=str, default="gpt-4o-mini")
    ap.add_argument(
        "--caption-subject-tags",
        type=str,
        default="12-year-old boy, child, youth softball player",
        help="Comma-separated tags inserted after the trigger token (must start with yuwen).",
    )
    ap.add_argument("--caption-skip-existing", action="store_true", help="Skip OpenAI calls if caption txt exists")
    ap.add_argument("--caption-overwrite-existing", action="store_true", help="Overwrite existing captions (force recaption)")
    ap.add_argument("--skip-assemble", action="store_true", help="Skip Kohya training dir assembly")

    args = ap.parse_args()

    images = _list_images(args.input_dir)
    if not images:
        raise SystemExit(f"No images found in {args.input_dir}")

    paths = _resolve_paths(args.lora_data_root, args.character_id)

    archive_root = Path("/mnt/data/_archive_deleted/win-or-lose/yuwen")
    archived: Dict[str, Optional[str]] = {"prepared": None, "augmented": None, "training": None}
    if args.clean:
        archived["prepared"] = str(_archive_move(paths.prepared_dir, archive_root, "prepared") or "")
        archived["augmented"] = str(_archive_move(paths.augmented_dir, archive_root, "augmented") or "")
        archived["training"] = str(_archive_move(paths.training_dir, archive_root, "training") or "")

    _ensure_dir(paths.prepared_dir)
    _ensure_dir(paths.augmented_dir)
    _ensure_dir(paths.captions_dir)
    _ensure_dir(paths.training_dir)

    run_dir = paths.captions_dir / f"run_{_now_tag()}"
    _ensure_dir(run_dir)

    # Phase 1: preprocess
    prepared = preprocess_images(
        images,
        out_dir=paths.prepared_dir,
        out_size=args.out_size,
        gray=args.gray,
        max_files=args.max_files,
    )

    # Phase 2: captions (on ORIGINALS only, to save resources; augmented variants reuse captions)
    caption_stats: Optional[Dict[str, object]] = None
    if args.run_captions:
        if not os.getenv("OPENAI_API_KEY"):
            raise SystemExit("OPENAI_API_KEY is not set in environment (source ~/.bashrc first).")
        caption_stats = generate_captions_openai(
            images_dir=paths.prepared_dir,
            captions_out_dir=paths.captions_dir,
            character_token=args.character_token,
            subject_tags=args.caption_subject_tags,
            model=args.openai_model,
            max_files=args.max_files,
            skip_existing=args.caption_skip_existing,
            overwrite_existing=args.caption_overwrite_existing,
        )

    # Phase 3: augment (after captions)
    aug_stats = augment_to_target(prepared, out_dir=paths.augmented_dir, target_count=args.target_count, seed=args.seed)

    # Phase 4: assemble kohya training dir (requires captions for originals; auto-links for augmented)
    kohya_dir: Optional[Path] = None
    if not args.skip_assemble:
        kohya_dir = assemble_kohya_training_dir(
            augmented_dir=paths.augmented_dir,
            captions_dir=paths.captions_dir,
            training_dir=paths.training_dir,
            repeats=args.repeats,
            token=args.character_token,
        )

    report = {
        "character_id": args.character_id,
        "input_dir": str(args.input_dir),
        "prepared_dir": str(paths.prepared_dir),
        "augmented_dir": str(paths.augmented_dir),
        "captions_dir": str(paths.captions_dir),
        "training_dir": str(paths.training_dir),
        "kohya_dir": str(kohya_dir) if kohya_dir else None,
        "input_images": len(images),
        "prepared_images": len(prepared),
        "archived": archived,
        "augmentation": aug_stats,
        "captions": caption_stats,
        "settings": {
            "out_size": args.out_size,
            "gray": args.gray,
            "target_count": args.target_count,
            "seed": args.seed,
            "repeats": args.repeats,
            "caption_before_augment": True,
            "caption_skip_existing": bool(args.caption_skip_existing),
            "caption_overwrite_existing": bool(args.caption_overwrite_existing),
        },
        "timestamp": datetime.now().isoformat(),
    }

    _write_json(run_dir / "FINAL_REPORT.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
