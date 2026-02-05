#!/usr/bin/env python3
"""
Synthetic Generated Data Report (Single-Character)
==================================================

Scans the synthetic single-character generated datasets under:
  /mnt/data/ai_data/synthetic_lora_data/generated_data/{character}/{type}/images/*.png

Produces:
  - A per-character/type summary (counts, missing per prompt, size stats)
  - Simple heuristics to find "pair-composition friendly" frames:
      - high edge energy (sharpness proxy)
      - low corner variance (background simplicity proxy)
      - framing preference inferred from prompt text (full/3/4 > medium > close-up)

This is intended to support building 2-character composite datasets later.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageFilter, ImageStat


IMG_RE = re.compile(r"prompt_(\d+)_img_(\d+)\.png$")


def _safe_int(x: str) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None


def _edge_energy(img: Image.Image) -> float:
    gray = img.convert("L")
    edges = gray.filter(ImageFilter.FIND_EDGES)
    s = ImageStat.Stat(edges)
    return float(s.mean[0])


def _corner_std(img: Image.Image, patch: int = 32) -> float:
    rgb = img.convert("RGB")
    w, h = rgb.size
    patch = min(patch, w // 4, h // 4)
    if patch <= 0:
        return 0.0

    def crop(x0: int, y0: int) -> Image.Image:
        return rgb.crop((x0, y0, x0 + patch, y0 + patch))

    corners = [
        crop(0, 0),
        crop(w - patch, 0),
        crop(0, h - patch),
        crop(w - patch, h - patch),
    ]
    vals: List[float] = []
    for c in corners:
        st = ImageStat.Stat(c)
        # mean of per-channel stddevs
        vals.append(float(sum(st.stddev) / len(st.stddev)))
    return float(sum(vals) / len(vals))


def _infer_framing(prompt: str) -> str:
    p = prompt.lower()
    if "full body" in p or "full-body" in p:
        return "full_body"
    if "three-quarter body" in p or "3/4 body" in p or "three quarter body" in p:
        return "three_quarter_body"
    if "medium shot" in p or "waist up" in p:
        return "medium"
    if "close-up" in p or "close up" in p:
        return "close_up"
    return "unknown"


def _framing_score(framing: str) -> float:
    # For pair composition, full/3/4 body are most useful.
    return {
        "full_body": 1.0,
        "three_quarter_body": 0.8,
        "medium": 0.4,
        "close_up": 0.1,
        "unknown": 0.3,
    }.get(framing, 0.3)


@dataclass
class ImageScore:
    path: str
    character: str
    lora_type: str
    prompt_index: int
    image_index: int
    width: int
    height: int
    file_size: int
    edge_energy: float
    corner_std: float
    framing: str
    score: float


def _score_image(path: Path, character: str, lora_type: str, prompt_text: Optional[str]) -> ImageScore:
    m = IMG_RE.search(path.name)
    if not m:
        raise ValueError(f"Unexpected filename: {path}")
    prompt_index = int(m.group(1))
    image_index = int(m.group(2))

    with Image.open(path) as im:
        im.load()
        w, h = im.size

        # Speed: score on a small thumbnail rather than full 1024x1024.
        thumb = im
        max_dim = 256
        if max(w, h) > max_dim:
            scale = max_dim / float(max(w, h))
            tw = max(1, int(w * scale))
            th = max(1, int(h * scale))
            thumb = im.resize((tw, th), Image.BILINEAR)

        e = _edge_energy(thumb)
        c = _corner_std(thumb)

    framing = _infer_framing(prompt_text or "")

    # Higher edge energy => sharper; lower corner std => simpler background
    # Normalize corner_std penalty with a soft clamp
    bg_penalty = min(c / 40.0, 1.5)  # ~0..1.5
    sharp_norm = min(e / 25.0, 2.0)  # ~0..2
    framing_w = _framing_score(framing)

    score = (sharp_norm * 1.0) + (framing_w * 1.5) - (bg_penalty * 1.0)

    return ImageScore(
        path=str(path),
        character=character,
        lora_type=lora_type,
        prompt_index=prompt_index,
        image_index=image_index,
        width=w,
        height=h,
        file_size=path.stat().st_size,
        edge_energy=e,
        corner_std=c,
        framing=framing,
        score=score,
    )


def _load_prompts_map(type_dir: Path) -> Dict[int, str]:
    prompts_path = type_dir / "prompts.json"
    if not prompts_path.exists():
        return {}
    data = json.loads(prompts_path.read_text(encoding="utf-8"))
    prompts = data.get("prompts", [])
    out: Dict[int, str] = {}
    if isinstance(prompts, list):
        for idx, item in enumerate(prompts):
            if isinstance(item, dict) and isinstance(item.get("prompt"), str):
                out[idx] = item["prompt"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("/mnt/data/ai_data/synthetic_lora_data/generated_data"),
        help="Synthetic generated_data root.",
    )
    ap.add_argument("--exclude", nargs="*", default=["yokai-watch"], help="Top-level dirs to exclude.")
    ap.add_argument("--types", nargs="*", default=["pose", "action", "expression"])
    ap.add_argument("--top-k", type=int, default=30, help="Top candidates per character/type.")
    ap.add_argument(
        "--per-prompt-sample",
        type=int,
        default=2,
        help="Max images to score per prompt index (speeds up large datasets).",
    )
    ap.add_argument("--out-json", type=Path, default=Path("outputs/reports/synthetic_single_character_report.json"))
    ap.add_argument("--out-md", type=Path, default=Path("outputs/reports/synthetic_single_character_report.md"))
    args = ap.parse_args()

    root = args.root
    exclude = set(args.exclude or [])
    types = list(args.types)

    characters = sorted([p for p in root.iterdir() if p.is_dir() and p.name not in exclude])

    report: Dict[str, Any] = {
        "root": str(root),
        "exclude": sorted(exclude),
        "types": types,
        "characters": [],
    }

    md_lines: List[str] = []
    md_lines.append("# Synthetic Single-Character Report")
    md_lines.append("")
    md_lines.append(f"- Root: `{root}`")
    md_lines.append(f"- Exclude: `{', '.join(sorted(exclude))}`")
    md_lines.append(f"- Types: `{', '.join(types)}`")
    md_lines.append("")

    for char_dir in characters:
        char_name = char_dir.name
        char_entry: Dict[str, Any] = {"character": char_name, "types": {}}

        md_lines.append(f"## {char_name}")

        for t in types:
            type_dir = char_dir / t
            images_dir = type_dir / "images"
            if not images_dir.exists():
                char_entry["types"][t] = {"exists": False, "count": 0}
                md_lines.append(f"- {t}: missing `{images_dir}`")
                continue

            prompts_map = _load_prompts_map(type_dir)

            all_imgs = sorted([p for p in images_dir.glob("*.png") if IMG_RE.search(p.name)])
            # Speed: stratified sampling per prompt index for scoring
            per_prompt = max(1, int(args.per_prompt_sample))
            bucket: Dict[int, List[Path]] = {}
            for p in all_imgs:
                m = IMG_RE.search(p.name)
                if not m:
                    continue
                pi = int(m.group(1))
                bucket.setdefault(pi, []).append(p)
            imgs: List[Path] = []
            for pi in sorted(bucket.keys()):
                imgs.extend(bucket[pi][:per_prompt])

            counts_by_prompt: Dict[int, int] = {}
            file_sizes = []

            scored: List[ImageScore] = []
            # Count using all images (accurate missing stats)
            for p in all_imgs:
                m = IMG_RE.search(p.name)
                if not m:
                    continue
                prompt_idx = int(m.group(1))
                counts_by_prompt[prompt_idx] = counts_by_prompt.get(prompt_idx, 0) + 1
                file_sizes.append(p.stat().st_size)
            # Score only sampled images
            for p in imgs:
                m = IMG_RE.search(p.name)
                if not m:
                    continue
                prompt_idx = int(m.group(1))
                prompt_text = prompts_map.get(prompt_idx)
                try:
                    scored.append(_score_image(p, char_name, t, prompt_text))
                except Exception:
                    continue

            count = len(all_imgs)
            expected_per_prompt = 10
            prompt_count = max(counts_by_prompt.keys(), default=-1) + 1 if counts_by_prompt else 0
            missing_prompts = [i for i in range(prompt_count) if counts_by_prompt.get(i, 0) < expected_per_prompt]

            top = sorted(scored, key=lambda x: x.score, reverse=True)[: args.top_k]

            char_entry["types"][t] = {
                "exists": True,
                "count": count,
                "prompt_count_observed": prompt_count,
                "missing_prompt_indices": missing_prompts,
                "file_size": {
                    "min": min(file_sizes) if file_sizes else None,
                    "median": int(statistics.median(file_sizes)) if file_sizes else None,
                    "max": max(file_sizes) if file_sizes else None,
                },
                "top_candidates": [top_item.__dict__ for top_item in top],
            }

            md_lines.append(
                f"- {t}: {count} images, prompts observed={prompt_count}, prompts with <{expected_per_prompt} imgs={len(missing_prompts)}"
            )
            if top:
                md_lines.append(f"  - top{min(args.top_k, len(top))} candidates (pair-friendly):")
                for item in top[: min(5, len(top))]:
                    md_lines.append(
                        f"    - `{item.path}` score={item.score:.2f} framing={item.framing} edge={item.edge_energy:.1f} corner_std={item.corner_std:.1f}"
                    )

        report["characters"].append(char_entry)
        md_lines.append("")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.out_md.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
