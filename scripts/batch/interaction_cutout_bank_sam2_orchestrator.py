#!/usr/bin/env python3
"""
SAM2 Cutout Bank Orchestrator (from existing synthetic images)
==============================================================

Builds a per-character cutout bank using SAM2, sourcing images from:
  /mnt/data/ai_data/synthetic_lora_data/generated_data/{character}/{type}/images/*.png

Selection modes:
  - per_prompt_sample (recommended): sample N images per prompt index from the
    existing synthetic folders.
  - target_count (recommended for filling): sample across all types/prompts until
    each character reaches a target number of valid cutouts.
  - report_topk (optional): use a report JSON (produced by
    synthetic_generated_data_report.py) to pick top candidate images.

Segmentation:
  - Runs scripts/generic/segmentation/instance_segmentation.py in transparent mode
    on a staged folder of selected images (symlinks).

Post-process:
  - Picks the best instance per source image using a score (area_ratio + quality + centeredness)
  - Filters likely multi-character frames using 2nd large instance + bbox separation
  - Outputs RGBA cutouts to:
      {out_root}/{character}/*.png
    with an optional crop bbox JSON alongside (for compose_interaction_dataset.py).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from PIL import Image
import numpy as np

LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    selection_mode: str
    report_json: Optional[Path]
    source_root: Path
    characters: List[str]
    types: List[str]
    per_type_top_k: int
    per_prompt_sample: int
    max_prompts_per_type: Optional[int]
    per_prompt_cap: int
    target_cutouts_per_character: int
    seed: int

    stage_root: Path
    sam2_output_root: Path
    out_root: Path
    clean_sam2_output: bool

    model_type: str
    device: str
    python_executable: Path
    min_size: int
    context_padding: int
    use_async_io: bool
    prefetch_size: int
    save_workers: int

    min_area_ratio: float
    max_area_ratio: float
    multi_second_area_ratio_min: float
    multi_second_area_ratio_max: float
    multi_bbox_iou_max: float

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        sam2 = raw.get("sam2", {})
        selection = raw.get("selection", {})
        io = raw.get("io", {})
        filtering = raw.get("filtering", {})

        report_json_raw = selection.get("report_json")
        report_json = Path(report_json_raw) if report_json_raw else None

        return cls(
            selection_mode=str(selection.get("mode", "per_prompt_sample")),
            report_json=report_json,
            source_root=Path(selection.get("source_root", "/mnt/data/ai_data/synthetic_lora_data/generated_data")),
            characters=[str(x) for x in selection["characters"]],
            types=[str(x) for x in selection.get("types", ["pose", "action", "expression"])],
            per_type_top_k=int(selection.get("per_type_top_k", 80)),
            per_prompt_sample=int(selection.get("per_prompt_sample", 1)),
            max_prompts_per_type=(int(selection["max_prompts_per_type"]) if "max_prompts_per_type" in selection else None),
            per_prompt_cap=int(selection.get("per_prompt_cap", 3)),
            target_cutouts_per_character=int(selection.get("target_cutouts_per_character", 320)),
            seed=int(selection.get("seed", 1234)),
            stage_root=Path(io["stage_root"]),
            sam2_output_root=Path(io["sam2_output_root"]),
            out_root=Path(io["out_root"]),
            clean_sam2_output=bool(io.get("clean_sam2_output", False)),
            model_type=str(sam2.get("model_type", "sam2_hiera_large")),
            device=str(sam2.get("device", "cuda")),
            python_executable=Path(sam2.get("python_executable", sys.executable)),
            min_size=int(sam2.get("min_size", 128 * 128)),
            context_padding=int(sam2.get("context_padding", 10)),
            use_async_io=bool(sam2.get("use_async_io", True)),
            prefetch_size=int(sam2.get("prefetch_size", 16)),
            save_workers=int(sam2.get("save_workers", 4)),
            min_area_ratio=float(filtering.get("min_area_ratio", 0.06)),
            max_area_ratio=float(filtering.get("max_area_ratio", 0.95)),
            multi_second_area_ratio_min=float(filtering.get("multi_second_area_ratio_min", 0.12)),
            multi_second_area_ratio_max=float(filtering.get("multi_second_area_ratio_max", 0.55)),
            multi_bbox_iou_max=float(filtering.get("multi_bbox_iou_max", 0.08)),
        )


def _safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(str(src), str(dst))


def _load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _select_paths_from_report(report: Dict[str, Any], character: str, types: List[str], per_type_top_k: int) -> List[Path]:
    for c in report.get("characters", []):
        if c.get("character") == character:
            out: List[Path] = []
            for t in types:
                tinfo = (c.get("types") or {}).get(t) or {}
                top = tinfo.get("top_candidates") or []
                for item in top[:per_type_top_k]:
                    p = item.get("path")
                    if p:
                        out.append(Path(p))
            # de-dup while preserving order
            seen = set()
            uniq: List[Path] = []
            for p in out:
                if p in seen:
                    continue
                seen.add(p)
                uniq.append(p)
            return uniq
    raise KeyError(f"Character not found in report: {character}")


def _extract_prompt_key(path: Path) -> str:
    """
    Best-effort prompt grouping key from filenames like:
      prompt_0003_img_05.png -> prompt_0003
    """
    name = path.name
    if not name.startswith("prompt_"):
        return "misc"
    parts = name.split("_")
    if len(parts) >= 2 and parts[0] == "prompt":
        return f"{parts[0]}_{parts[1]}"
    return "misc"


def _stable_seed(base_seed: int, *parts: str) -> int:
    h = hashlib.md5(("|".join(parts)).encode("utf-8")).hexdigest()
    return (base_seed ^ int(h[:8], 16)) & 0xFFFFFFFF


def _select_paths_per_prompt(
    source_root: Path,
    character: str,
    lora_type: str,
    per_prompt_sample: int,
    max_prompts_per_type: Optional[int],
    seed: int,
) -> List[Path]:
    img_dir = source_root / character / lora_type / "images"
    if not img_dir.exists():
        return []
    images = sorted(img_dir.glob("*.png"))
    if not images:
        return []

    grouped: Dict[str, List[Path]] = defaultdict(list)
    for p in images:
        grouped[_extract_prompt_key(p)].append(p)

    keys = sorted(grouped.keys())
    import random

    rng = random.Random(_stable_seed(seed, character, lora_type))
    rng.shuffle(keys)
    if max_prompts_per_type is not None:
        keys = keys[: max(0, int(max_prompts_per_type))]

    selected: List[Path] = []
    for k in keys:
        group = grouped[k]
        group_sorted = sorted(group, key=lambda x: x.name)
        if per_prompt_sample >= len(group_sorted):
            selected.extend(group_sorted)
        else:
            rng.shuffle(group_sorted)
            selected.extend(group_sorted[:per_prompt_sample])
    return selected


def _list_source_images(source_root: Path, character: str, types: List[str]) -> List[Path]:
    out: List[Path] = []
    for t in types:
        img_dir = source_root / character / t / "images"
        if not img_dir.exists():
            continue
        out.extend(sorted(img_dir.glob("*.png")))
    # de-dup while preserving order
    seen = set()
    uniq: List[Path] = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def _select_paths_target_count(
    source_root: Path,
    character: str,
    types: List[str],
    needed: int,
    per_prompt_cap: int,
    max_prompts_per_type: Optional[int],
    seed: int,
) -> List[Path]:
    """
    Diversified selection across prompt groups until `needed` images are chosen.
    """
    if needed <= 0:
        return []

    import random

    # groups[(type, prompt_key)] = [paths...]
    groups: Dict[Tuple[str, str], List[Path]] = defaultdict(list)
    for t in types:
        img_dir = source_root / character / t / "images"
        if not img_dir.exists():
            continue
        images = sorted(img_dir.glob("*.png"))
        if not images:
            continue
        by_prompt: Dict[str, List[Path]] = defaultdict(list)
        for p in images:
            by_prompt[_extract_prompt_key(p)].append(p)
        keys = sorted(by_prompt.keys())
        rng = random.Random(_stable_seed(seed, character, t))
        rng.shuffle(keys)
        if max_prompts_per_type is not None:
            keys = keys[: max(0, int(max_prompts_per_type))]
        for k in keys:
            groups[(t, k)].extend(sorted(by_prompt[k], key=lambda x: x.name))

    if not groups:
        return []

    group_keys = list(groups.keys())
    rng = random.Random(_stable_seed(seed, character, "target_count"))
    rng.shuffle(group_keys)

    # Track how many picked per group
    picked_per_group: Dict[Tuple[str, str], int] = defaultdict(int)
    selected: List[Path] = []

    # Round-robin across groups to maximize diversity
    progressed = True
    while len(selected) < needed and progressed:
        progressed = False
        for gk in group_keys:
            if len(selected) >= needed:
                break
            if picked_per_group[gk] >= per_prompt_cap:
                continue
            pool = groups.get(gk) or []
            if not pool:
                continue
            # Choose deterministically: shuffle per group once, then pop
            if picked_per_group[gk] == 0:
                rng.shuffle(pool)
            p = pool.pop()
            selected.append(p)
            picked_per_group[gk] += 1
            progressed = True

    return selected


def _run_sam2(input_dir: Path, output_dir: Path, cfg: Config) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(cfg.python_executable),
        str(PROJECT_ROOT / "scripts/generic/segmentation/instance_segmentation.py"),
        str(input_dir),
        "--output-dir",
        str(output_dir),
        "--model",
        cfg.model_type,
        "--device",
        cfg.device,
        "--min-size",
        str(cfg.min_size),
        "--context-mode",
        "transparent",
        "--context-padding",
        str(cfg.context_padding),
    ]
    if cfg.use_async_io:
        cmd += ["--use-async-io", "--prefetch-size", str(cfg.prefetch_size), "--save-workers", str(cfg.save_workers)]

    LOGGER.info("SAM2: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _bbox_xywh_to_xyxy(bbox_xywh: Any) -> Optional[Tuple[int, int, int, int]]:
    if not isinstance(bbox_xywh, list) or len(bbox_xywh) != 4:
        return None
    try:
        x, y, w, h = (float(bbox_xywh[0]), float(bbox_xywh[1]), float(bbox_xywh[2]), float(bbox_xywh[3]))
    except Exception:
        return None
    x1 = int(max(0, round(x)))
    y1 = int(max(0, round(y)))
    x2 = int(max(x1 + 1, round(x + w)))
    y2 = int(max(y1 + 1, round(y + h)))
    return (x1, y1, x2, y2)


def _bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _score_instance(
    meta: Dict[str, Any],
    image_size: Tuple[int, int],
    min_area_ratio: float,
    max_area_ratio: float,
) -> Optional[Tuple[float, float, Tuple[int, int, int, int]]]:
    w, h = image_size
    if w <= 0 or h <= 0:
        return None
    bbox_xyxy = _bbox_xywh_to_xyxy(meta.get("bbox"))
    if not bbox_xyxy:
        return None
    x1, y1, x2, y2 = bbox_xyxy
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    # Heuristic: reject "background-like" masks that cover almost the whole image
    if (bw / float(w)) >= 0.96 and (bh / float(h)) >= 0.96:
        return None
    area = float(meta.get("area", 0.0))
    area_ratio = float(area / float(w * h))
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return None

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    dx = (cx - (w / 2.0)) / float(w)
    dy = (cy - (h / 2.0)) / float(h)
    center_dist = (dx * dx + dy * dy) ** 0.5

    stability = float(meta.get("stability_score", 0.0) or 0.0)
    iou = float(meta.get("predicted_iou", 0.0) or 0.0)

    score = (area_ratio * 0.55) + (stability * 0.25) + (iou * 0.20) - (center_dist * 0.20)
    return (score, area_ratio, bbox_xyxy)


def _corner_alpha_mean(alpha: Image.Image) -> float:
    w, h = alpha.size
    pts = [(5, 5), (w - 6, 5), (5, h - 6), (w - 6, h - 6)]
    vals = [alpha.getpixel(p) for p in pts]
    return float(sum(vals) / len(vals))


def _fix_inverted_alpha(rgba: Image.Image) -> Tuple[Image.Image, bool]:
    """
    Some SAM2 outputs appear inverted (background opaque, subject transparent).
    Detect by checking corner alpha; if corners are mostly opaque, invert alpha.
    Also clears RGB in fully-transparent regions to reduce resize halos.
    """
    im = rgba.convert("RGBA")
    alpha = im.getchannel("A")
    inverted = _corner_alpha_mean(alpha) > 127.0
    if inverted:
        alpha = Image.eval(alpha, lambda a: 255 - a)
        im.putalpha(alpha)

    # Clear RGB in fully transparent pixels to reduce fringes when resizing
    arr = np.array(im, dtype=np.uint8)
    mask0 = arr[:, :, 3] == 0
    arr[mask0, 0:3] = 0
    return Image.fromarray(arr, mode="RGBA"), inverted


def _cutout_is_ok(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im = im.convert("RGBA")
            a = im.getchannel("A")
            # If corners are mostly transparent, it behaves like a normal cutout
            return _corner_alpha_mean(a) <= 127.0
    except Exception:
        return False


def _postprocess_instances(
    sam2_dir: Path,
    stage_dir: Path,
    out_dir: Path,
    cfg: Config,
) -> Dict[str, Any]:
    """
    Pick best instance per source frame, filter multi-person, export cutouts + bbox.
    """
    meta_path = sam2_dir / "instances_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing SAM2 metadata: {meta_path}")

    data = json.loads(meta_path.read_text(encoding="utf-8"))
    items = data.get("metadata") or []
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for m in items:
        src = m.get("source_frame")
        if src:
            by_source[str(src)].append(m)

    out_dir.mkdir(parents=True, exist_ok=True)

    kept = 0
    skipped_multi = 0
    skipped_missing = 0
    skipped_scoring = 0
    already_done = 0

    for source_frame, metas in by_source.items():
        src_name = Path(source_frame).name
        dst_png = out_dir / f"{Path(src_name).stem}.png"
        dst_json = out_dir / f"{Path(src_name).stem}.json"
        if dst_png.exists():
            if _cutout_is_ok(dst_png):
                already_done += 1
                continue

        input_frame_path = stage_dir / src_name
        if not input_frame_path.exists():
            skipped_scoring += 1
            continue

        try:
            with Image.open(input_frame_path) as im:
                im.load()
                image_size = im.size
        except Exception:
            skipped_scoring += 1
            continue

        scored: List[Tuple[float, float, Tuple[int, int, int, int], Dict[str, Any]]] = []
        for m in metas:
            s = _score_instance(
                m,
                image_size=image_size,
                min_area_ratio=cfg.min_area_ratio,
                max_area_ratio=cfg.max_area_ratio,
            )
            if not s:
                continue
            score, area_ratio, bbox_xyxy = s
            scored.append((score, area_ratio, bbox_xyxy, m))

        if not scored:
            skipped_scoring += 1
            continue

        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_area_ratio, best_bbox, best_meta = scored[0]

        # Multi-character heuristic: two large, separated candidates
        second = None
        for cand in scored[1:]:
            if cand[1] >= cfg.multi_second_area_ratio_min:
                second = cand
                break

        if second is not None:
            _, second_area_ratio, second_bbox, _ = second
            if (
                best_area_ratio > 0
                and (second_area_ratio / best_area_ratio) >= cfg.multi_second_area_ratio_max
                and _bbox_iou(best_bbox, second_bbox) <= cfg.multi_bbox_iou_max
            ):
                skipped_multi += 1
                continue

        inst_filename = best_meta.get("instance_filename")
        if not isinstance(inst_filename, str):
            skipped_missing += 1
            continue

        inst_path = sam2_dir / "instances" / inst_filename
        if not inst_path.exists():
            # Fallback: try inst0 name
            alt = sam2_dir / "instances" / f"{Path(src_name).stem}_inst0.png"
            if alt.exists():
                inst_path = alt
            else:
                skipped_missing += 1
                continue

        try:
            with Image.open(inst_path) as inst_im:
                inst_im.load()
                fixed, inverted = _fix_inverted_alpha(inst_im)
        except Exception:
            skipped_missing += 1
            continue

        alpha = fixed.getchannel("A")
        bbox_alpha = alpha.getbbox()
        if bbox_alpha is None:
            skipped_scoring += 1
            continue

        fixed.save(dst_png, optimize=True)
        payload = {
            "bbox": list(bbox_alpha),
            "meta": {
                "source_frame": source_frame,
                "instance_filename": inst_filename,
                "bbox_original_xywh": best_meta.get("bbox"),
                "score": best_score,
                "area_ratio": best_area_ratio,
                "predicted_iou": best_meta.get("predicted_iou"),
                "stability_score": best_meta.get("stability_score"),
                "num_instances": len(metas),
                "alpha_inverted_fix_applied": bool(inverted),
            },
        }
        dst_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        kept += 1

    return {
        "kept": kept,
        "already_done": already_done,
        "skipped_multi": skipped_multi,
        "skipped_missing": skipped_missing,
        "skipped_scoring": skipped_scoring,
        "total_frames": len(by_source),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level))
    cfg = Config.from_yaml(args.config)

    if cfg.selection_mode not in {"per_prompt_sample", "target_count", "report_topk"}:
        raise SystemExit(f"Unsupported selection.mode: {cfg.selection_mode}")

    report = _load_report(cfg.report_json) if (cfg.selection_mode == "report_topk" and cfg.report_json) else None
    if cfg.selection_mode == "report_topk" and report is None:
        raise SystemExit("selection.mode=report_topk requires selection.report_json")

    for idx, character in enumerate(cfg.characters, start=1):
        LOGGER.info("[%d/%d] %s: selecting images (mode=%s)", idx, len(cfg.characters), character, cfg.selection_mode)

        paths: List[Path] = []
        if cfg.selection_mode == "report_topk":
            assert report is not None
            paths = _select_paths_from_report(report, character, cfg.types, cfg.per_type_top_k)
        elif cfg.selection_mode == "per_prompt_sample":
            for t in cfg.types:
                paths.extend(
                    _select_paths_per_prompt(
                        source_root=cfg.source_root,
                        character=character,
                        lora_type=t,
                        per_prompt_sample=cfg.per_prompt_sample,
                        max_prompts_per_type=cfg.max_prompts_per_type,
                        seed=cfg.seed,
                    )
                )
        else:
            # target_count: decide how many more we need based on existing valid cutouts
            out_dir = cfg.out_root / character
            out_dir.mkdir(parents=True, exist_ok=True)
            existing_ok = 0
            for png in out_dir.glob("*.png"):
                if _cutout_is_ok(png):
                    existing_ok += 1
            needed = max(0, int(cfg.target_cutouts_per_character) - existing_ok)
            LOGGER.info(
                "[%d/%d] %s: existing_ok=%d target=%d needed=%d",
                idx,
                len(cfg.characters),
                character,
                existing_ok,
                cfg.target_cutouts_per_character,
                needed,
            )
            paths = _select_paths_target_count(
                source_root=cfg.source_root,
                character=character,
                types=cfg.types,
                needed=needed,
                per_prompt_cap=cfg.per_prompt_cap,
                max_prompts_per_type=cfg.max_prompts_per_type,
                seed=cfg.seed,
            )

        # de-dup while preserving order
        seen = set()
        uniq: List[Path] = []
        for p in paths:
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
        paths = uniq

        if not paths:
            LOGGER.warning("%s: no selected images, skipping", character)
            continue

        stage_dir = cfg.stage_root / character
        if stage_dir.exists():
            shutil.rmtree(stage_dir, ignore_errors=True)
        stage_dir.mkdir(parents=True, exist_ok=True)

        out_dir = cfg.out_root / character
        out_dir.mkdir(parents=True, exist_ok=True)

        staged = 0
        for p in paths:
            if not p.exists():
                continue
            existing = out_dir / f"{p.stem}.png"
            if existing.exists() and _cutout_is_ok(existing):
                continue
            _safe_symlink(p, stage_dir / p.name)
            staged += 1

        if staged == 0:
            LOGGER.info("[%d/%d] %s: nothing new to segment (all outputs exist)", idx, len(cfg.characters), character)
            continue

        sam2_dir = cfg.sam2_output_root / character
        if cfg.clean_sam2_output and sam2_dir.exists():
            shutil.rmtree(sam2_dir, ignore_errors=True)

        LOGGER.info("[%d/%d] %s: running SAM2 on %d staged images", idx, len(cfg.characters), character, staged)
        _run_sam2(stage_dir, sam2_dir, cfg)

        LOGGER.info("[%d/%d] %s: postprocess instances", idx, len(cfg.characters), character)
        stats = _postprocess_instances(sam2_dir=sam2_dir, stage_dir=stage_dir, out_dir=out_dir, cfg=cfg)
        (out_dir / "cutouts_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
        LOGGER.info("[%d/%d] %s: %s", idx, len(cfg.characters), character, stats)

    LOGGER.info("Done. Cutouts at %s", cfg.out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
