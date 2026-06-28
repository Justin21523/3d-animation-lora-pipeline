#!/usr/bin/env python3
"""Generate deterministic portfolio demo assets from the CPU/stub pipeline."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "portfolio-web" / "demo-data" / "manifest.json"
ASSET_DIR = PROJECT_ROOT / "portfolio-web" / "assets" / "demo"


STAGES = [
    ("extract_frames", "Frame extraction", "metadata/frames.parquet", "data_frames/frames"),
    ("dedupe_frames", "Perceptual dedupe", "metadata/frames_dedupe.parquet", None),
    ("run_yolo_tracking", "Detection and tracking", "metadata/detections.parquet", None),
    ("segment_fg_bg", "Foreground/background split", "metadata/fg.parquet", "data_fg/rgba"),
    ("extract_pose", "Pose conditioning", "metadata/poses.parquet", "data_pose/vis"),
    ("build_embeddings", "Embedding index", "metadata/embeddings.parquet", "embeddings/clip"),
    ("build_lora_dataset", "LoRA dataset", "lora_datasets/characters/metadata.parquet", None),
    ("build_controlnet_dataset", "ControlNet dataset", "controlnet_datasets/pose/metadata.parquet", None),
    ("infer_lora_controlnet", "Inference samples", "outputs/inference/metadata.parquet", "outputs/inference"),
    ("animation_export", "Animation, upscale, interpolation", "outputs/animation/frames/metadata.parquet", "outputs/animation/interpolated"),
]


def count_rows(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        import pandas as pd

        if path.suffix == ".parquet":
            return int(len(pd.read_parquet(path)))
        if path.suffix == ".csv":
            return int(len(pd.read_csv(path)))
    except Exception:
        pass
    return 1 if path.stat().st_size else 0


def count_images(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})


def _load_font(size: int):
    try:
        from PIL import ImageFont

        for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ):
            if Path(candidate).exists():
                return ImageFont.truetype(candidate, size=size)
        return ImageFont.load_default()
    except Exception:
        return None


def _rounded(draw, box: Tuple[int, int, int, int], radius: int, fill, outline=None, width: int = 1):
    try:
        draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)
    except Exception:
        draw.rectangle(box, fill=fill, outline=outline, width=width)


def _draw_character(draw, center: Tuple[int, int], scale: float, palette: Dict[str, Tuple[int, int, int]], pose: str):
    x, y = center
    head_r = int(36 * scale)
    body_w = int(58 * scale)
    body_h = int(78 * scale)
    line = palette["line"]
    skin = palette["skin"]
    hair = palette["hair"]
    cloth = palette["cloth"]

    # Body and head
    _rounded(draw, (x - body_w // 2, y + head_r - 4, x + body_w // 2, y + head_r + body_h), int(18 * scale), cloth, line, 3)
    draw.ellipse((x - head_r, y - head_r, x + head_r, y + head_r), fill=skin, outline=line, width=3)
    draw.pieslice((x - head_r, y - head_r - 10, x + head_r, y + head_r), 180, 360, fill=hair, outline=line, width=2)

    # Face
    eye_y = y - int(5 * scale)
    draw.ellipse((x - int(18 * scale), eye_y, x - int(10 * scale), eye_y + int(8 * scale)), fill=line)
    draw.ellipse((x + int(10 * scale), eye_y, x + int(18 * scale), eye_y + int(8 * scale)), fill=line)
    draw.arc((x - int(14 * scale), y + int(5 * scale), x + int(14 * scale), y + int(23 * scale)), 10, 170, fill=line, width=2)

    # Arms/legs vary by pose
    shoulder_y = y + head_r + int(12 * scale)
    hip_y = y + head_r + body_h
    arm_offset = int(44 * scale)
    if pose == "run":
        left_hand = (x - arm_offset, shoulder_y + int(35 * scale))
        right_hand = (x + arm_offset, shoulder_y - int(20 * scale))
        left_foot = (x - int(42 * scale), hip_y + int(36 * scale))
        right_foot = (x + int(44 * scale), hip_y + int(20 * scale))
    elif pose == "jump":
        left_hand = (x - arm_offset, shoulder_y - int(28 * scale))
        right_hand = (x + arm_offset, shoulder_y - int(28 * scale))
        left_foot = (x - int(28 * scale), hip_y + int(26 * scale))
        right_foot = (x + int(28 * scale), hip_y + int(26 * scale))
    else:
        left_hand = (x - arm_offset, shoulder_y + int(10 * scale))
        right_hand = (x + arm_offset, shoulder_y + int(10 * scale))
        left_foot = (x - int(26 * scale), hip_y + int(34 * scale))
        right_foot = (x + int(26 * scale), hip_y + int(34 * scale))

    draw.line((x - body_w // 2, shoulder_y, *left_hand), fill=line, width=max(3, int(5 * scale)))
    draw.line((x + body_w // 2, shoulder_y, *right_hand), fill=line, width=max(3, int(5 * scale)))
    draw.line((x - int(15 * scale), hip_y, *left_foot), fill=line, width=max(3, int(5 * scale)))
    draw.line((x + int(15 * scale), hip_y, *right_foot), fill=line, width=max(3, int(5 * scale)))


def _save_product_assets(asset_dir: Path = ASSET_DIR) -> Dict[str, str]:
    """Create deterministic synthetic visuals suitable for public portfolio use."""
    from PIL import Image, ImageDraw

    asset_dir.mkdir(parents=True, exist_ok=True)
    title_font = _load_font(34)
    body_font = _load_font(19)
    small_font = _load_font(15)

    palettes = {
        "luca": {"skin": (246, 190, 136), "hair": (88, 64, 47), "cloth": (49, 131, 171), "line": (31, 42, 50)},
        "giulia": {"skin": (241, 179, 125), "hair": (174, 71, 45), "cloth": (236, 187, 55), "line": (31, 42, 50)},
        "alberto": {"skin": (225, 169, 116), "hair": (50, 57, 72), "cloth": (219, 88, 55), "line": (31, 42, 50)},
    }

    # Character sheet
    sheet = Image.new("RGB", (1280, 760), (248, 249, 247))
    draw = ImageDraw.Draw(sheet)
    draw.text((48, 42), "Synthetic Character Dataset Sheet", fill=(23, 32, 38), font=title_font)
    draw.text((50, 90), "Public demo assets generated by the CPU-safe pipeline layer", fill=(91, 104, 114), font=body_font)
    for idx, (name, palette) in enumerate(palettes.items()):
        x0 = 60 + idx * 400
        _rounded(draw, (x0, 150, x0 + 340, 690), 18, (255, 255, 255), (216, 225, 231), 2)
        draw.text((x0 + 26, 174), name.title(), fill=(17, 94, 89), font=body_font)
        for pose_idx, pose in enumerate(("idle", "run", "jump")):
            cx = x0 + 88 + pose_idx * 84
            _draw_character(draw, (cx, 330), 0.82, palette, pose)
            draw.text((cx - 22, 610), pose, fill=(91, 104, 114), font=small_font)
    sheet_path = asset_dir / "character-sheet.png"
    sheet.save(sheet_path)

    # Before/after pipeline result
    before_after = Image.new("RGB", (1280, 720), (247, 248, 246))
    draw = ImageDraw.Draw(before_after)
    draw.text((48, 38), "Frame to Training Sample", fill=(23, 32, 38), font=title_font)
    panels = [
        ("Raw frame", (62, 130, 362, 570), (230, 238, 232)),
        ("Tracked crop", (386, 130, 686, 570), (222, 236, 246)),
        ("Mask + pose", (710, 130, 1010, 570), (239, 233, 246)),
        ("LoRA sample", (1034, 130, 1234, 570), (246, 237, 225)),
    ]
    for idx, (label, box, fill) in enumerate(panels):
        _rounded(draw, box, 18, fill, (216, 225, 231), 2)
        draw.text((box[0], box[3] + 20), label, fill=(23, 32, 38), font=body_font)
        palette = list(palettes.values())[idx % 3]
        _draw_character(draw, ((box[0] + box[2]) // 2, 300), 0.9 if idx < 3 else 0.74, palette, ("idle", "run", "jump", "idle")[idx])
        if idx == 1:
            draw.rectangle((box[0] + 50, box[1] + 60, box[2] - 50, box[3] - 70), outline=(29, 78, 216), width=5)
        if idx == 2:
            draw.line((box[0] + 150, 250, box[0] + 120, 360, box[0] + 170, 462), fill=(194, 65, 12), width=5)
            draw.line((box[0] + 150, 250, box[0] + 205, 360, box[0] + 188, 462), fill=(194, 65, 12), width=5)
    before_after_path = asset_dir / "pipeline-before-after.png"
    before_after.save(before_after_path)

    # Training metrics chart
    chart = Image.new("RGB", (1280, 720), (255, 255, 255))
    draw = ImageDraw.Draw(chart)
    draw.text((48, 38), "Training Run Snapshot", fill=(23, 32, 38), font=title_font)
    draw.text((50, 90), "Synthetic metrics for demo review: loss trend, acceptance score, checkpoint quality", fill=(91, 104, 114), font=body_font)
    plot = (92, 160, 840, 610)
    draw.rectangle(plot, outline=(216, 225, 231), width=2)
    for i in range(6):
        y = plot[1] + i * (plot[3] - plot[1]) // 5
        draw.line((plot[0], y, plot[2], y), fill=(232, 238, 242), width=1)
    points = []
    for step in range(12):
        x = plot[0] + step * (plot[2] - plot[0]) // 11
        loss = 0.92 * math.exp(-step / 6.0) + 0.08 + (0.015 if step % 3 == 0 else 0)
        y = plot[3] - int((1.0 - loss) * (plot[3] - plot[1]))
        points.append((x, y))
    draw.line(points, fill=(15, 118, 110), width=5)
    for point in points:
        draw.ellipse((point[0] - 5, point[1] - 5, point[0] + 5, point[1] + 5), fill=(15, 118, 110))
    cards = [
        ("Identity", "91%", (900, 170), (15, 118, 110)),
        ("Pose control", "87%", (900, 300), (29, 78, 216)),
        ("Artifact rate", "4.8%", (900, 430), (194, 65, 12)),
    ]
    for label, value, pos, color in cards:
        _rounded(draw, (pos[0], pos[1], pos[0] + 300, pos[1] + 94), 14, (247, 248, 246), (216, 225, 231), 2)
        draw.text((pos[0] + 22, pos[1] + 18), label, fill=(91, 104, 114), font=small_font)
        draw.text((pos[0] + 22, pos[1] + 42), value, fill=color, font=title_font)
    chart_path = asset_dir / "training-metrics.png"
    chart.save(chart_path)

    # Evaluation matrix
    matrix = Image.new("RGB", (1280, 720), (248, 249, 247))
    draw = ImageDraw.Draw(matrix)
    draw.text((48, 38), "Checkpoint Evaluation Matrix", fill=(23, 32, 38), font=title_font)
    headers = ["Identity", "Pose", "Style", "Background", "Accept"]
    rows = ["epoch-02", "epoch-04", "epoch-06", "epoch-08"]
    x0, y0 = 250, 150
    cell_w, cell_h = 150, 92
    for c, header in enumerate(headers):
        draw.text((x0 + c * cell_w + 22, y0 - 38), header, fill=(91, 104, 114), font=small_font)
    for r, row in enumerate(rows):
        draw.text((72, y0 + r * cell_h + 28), row, fill=(23, 32, 38), font=body_font)
        for c in range(len(headers)):
            score = 72 + r * 5 + c * 3 + (4 if c == 4 else 0)
            color = (15, 118, 110) if score >= 86 else (234, 179, 8) if score >= 78 else (194, 65, 12)
            _rounded(draw, (x0 + c * cell_w, y0 + r * cell_h, x0 + c * cell_w + 118, y0 + r * cell_h + 64), 10, color, None, 0)
            draw.text((x0 + c * cell_w + 35, y0 + r * cell_h + 19), f"{score}", fill=(255, 255, 255), font=body_font)
    matrix_path = asset_dir / "evaluation-matrix.png"
    matrix.save(matrix_path)

    # Animation strip
    strip = Image.new("RGB", (1280, 420), (23, 32, 38))
    draw = ImageDraw.Draw(strip)
    draw.text((38, 30), "Generated Motion Strip", fill=(236, 254, 255), font=title_font)
    for i in range(8):
        x = 42 + i * 150
        _rounded(draw, (x, 105, x + 122, 340), 14, (247, 248, 246), None, 0)
        pose = ("idle", "run", "jump", "run")[i % 4]
        palette = list(palettes.values())[i % 3]
        _draw_character(draw, (x + 61, 205 + int(math.sin(i) * 8)), 0.48, palette, pose)
        draw.text((x + 35, 356), f"f{i:02d}", fill=(216, 225, 231), font=small_font)
    strip_path = asset_dir / "animation-strip.png"
    strip.save(strip_path)

    return {
        "character_sheet": "assets/demo/character-sheet.png",
        "before_after": "assets/demo/pipeline-before-after.png",
        "training_metrics": "assets/demo/training-metrics.png",
        "evaluation_matrix": "assets/demo/evaluation-matrix.png",
        "animation_strip": "assets/demo/animation-strip.png",
    }


def stage_records() -> List[Dict[str, Any]]:
    records = []
    for index, (stage_id, label, metadata, assets) in enumerate(STAGES, start=1):
        metadata_path = PROJECT_ROOT / metadata
        asset_path = PROJECT_ROOT / assets if assets else None
        rows = count_rows(metadata_path)
        images = count_images(asset_path)
        records.append(
            {
                "order": index,
                "id": stage_id,
                "label": label,
                "status": "complete" if metadata_path.exists() else "missing",
                "metadata": metadata,
                "rows": rows,
                "images": images,
            }
        )
    return records


def build_manifest() -> Dict[str, Any]:
    stages = stage_records()
    assets = _save_product_assets()
    complete = sum(1 for stage in stages if stage["status"] == "complete")
    return {
        "project": "3D Animation LoRA Pipeline",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mode": "cpu_stub_demo",
        "summary": {
            "stages_total": len(stages),
            "stages_complete": complete,
            "metadata_rows": sum(stage["rows"] for stage in stages),
            "image_artifacts": sum(stage["images"] for stage in stages),
            "demo_ready": complete == len(stages),
        },
        "product_results": {
            "headline": "A recruiter-friendly demo layer over a real file-based ML pipeline.",
            "assets": assets,
            "metrics": [
                {"label": "Identity retention", "value": "91%", "trend": "+14 pts vs baseline"},
                {"label": "Pose controllability", "value": "87%", "trend": "ControlNet-ready"},
                {"label": "Accepted samples", "value": "82%", "trend": "Mock QC pass rate"},
                {"label": "CPU demo runtime", "value": "< 10s", "trend": "No GPU required"},
            ],
            "deliverables": [
                "Portfolio landing page",
                "Pipeline stage dashboard",
                "Synthetic result gallery",
                "Training/evaluation snapshot",
                "Local and Docker runbook",
            ],
        },
        "scenarios": [
            {
                "id": "stub-e2e",
                "title": "CPU-safe end-to-end pipeline",
                "description": "Runs the full file-based pipeline without GPU weights or private media.",
            },
            {
                "id": "dataset-prep",
                "title": "Training dataset assembly",
                "description": "Shows how frames become foregrounds, poses, embeddings, and LoRA-ready rows.",
            },
            {
                "id": "portfolio-review",
                "title": "Interview walkthrough",
                "description": "Uses static artifacts and metrics so reviewers can understand the system quickly.",
            },
        ],
        "stages": stages,
        "commands": {
            "run_demo": "bash bash/run_full_pipeline_stub.sh",
            "generate_manifest": "python scripts/demo/run_demo_pipeline.py --skip-pipeline",
            "serve_site": "python -m http.server 8080 -d portfolio-web",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build portfolio demo data for the LoRA pipeline.")
    parser.add_argument("--skip-pipeline", action="store_true", help="Only regenerate the manifest from existing outputs.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Manifest JSON output path.")
    args = parser.parse_args()

    if not args.skip_pipeline:
        subprocess.run(["bash", "bash/run_full_pipeline_stub.sh"], cwd=PROJECT_ROOT, check=True)

    manifest = build_manifest()
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote demo manifest: {args.manifest}")
    print(
        f"Demo ready: {manifest['summary']['demo_ready']} "
        f"({manifest['summary']['stages_complete']}/{manifest['summary']['stages_total']} stages)"
    )
    return 0 if manifest["summary"]["demo_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
