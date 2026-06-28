#!/usr/bin/env python3
"""Generate a deterministic portfolio demo manifest from the CPU/stub pipeline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = PROJECT_ROOT / "portfolio-web" / "demo-data" / "manifest.json"


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
