#!/usr/bin/env python3
"""
Generate project configurations from existing dataset directories.

CPU-ONLY TOOL - No GPU required, uses 32-thread parallel processing.

This tool scans dataset directories and automatically generates
project configuration files following the AI_WAREHOUSE 3.0 structure.

Usage:
    python scripts/utils/generate_project_configs.py
    python scripts/utils/generate_project_configs.py --dataset-root /mnt/data/datasets/general
    python scripts/utils/generate_project_configs.py --project simpsons --force

Author: Justin Lu
Date: 2025-12-02
"""

import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Set, Any
from dataclasses import dataclass, field
import argparse
import json
import time

# CPU-only imports
import yaml

# Add parent directory for template access
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
CONFIG_ROOT = PROJECT_ROOT / "configs"
TEMPLATE_PATH = CONFIG_ROOT / "projects" / "_template.yaml"


@dataclass
class DetectedProject:
    """Information about a detected project."""
    name: str
    path: str
    has_frames: bool = False
    has_segmented: bool = False
    has_clustered: bool = False
    has_lora_data: bool = False
    frame_count: int = 0
    detected_characters: List[str] = field(default_factory=list)
    animation_type: str = "2d"  # Default to 2D


def detect_animation_type(project_dir: Path) -> str:
    """
    Detect animation type (2d or 3d) based on directory name or content.

    Returns:
        "2d" or "3d"
    """
    name_lower = project_dir.name.lower()

    # Known 3D projects (Pixar/Disney)
    known_3d = {
        "luca", "onward", "up", "coco", "turning-red", "elio", "orion",
        "toy-story", "finding-nemo", "frozen", "moana", "encanto",
        "brave", "inside-out", "soul", "ratatouille", "monsters"
    }

    # Known 2D projects
    known_2d = {
        "simpsons", "family-guy", "rick-and-morty", "adventure-time",
        "bojack", "archer", "futurama", "south-park", "king-of-the-hill"
    }

    if any(k in name_lower for k in known_3d):
        return "3d"
    if any(k in name_lower for k in known_2d):
        return "2d"

    # Default to 2D
    return "2d"


def count_images(directory: Path) -> int:
    """Count image files in directory (recursive)."""
    if not directory.exists():
        return 0

    count = 0
    for ext in [".png", ".jpg", ".jpeg", ".webp"]:
        count += len(list(directory.rglob(f"*{ext}")))
    return count


def list_character_dirs(directory: Path) -> List[str]:
    """List likely character directories."""
    if not directory.exists():
        return []

    chars = []
    for d in directory.iterdir():
        if d.is_dir() and not d.name.startswith((".", "noise", "unknown", "_")):
            chars.append(d.name)
    return chars


def detect_project(project_dir: Path) -> DetectedProject:
    """
    Detect project information from directory structure.

    Args:
        project_dir: Project directory path

    Returns:
        DetectedProject with detected information
    """
    result = DetectedProject(
        name=project_dir.name,
        path=str(project_dir)
    )

    # Detect animation type
    result.animation_type = detect_animation_type(project_dir)

    # Check frames
    frames_dir = project_dir / "frames"
    result.has_frames = frames_dir.exists()
    if result.has_frames:
        result.frame_count = count_images(frames_dir)

    # Check segmented
    for seg_name in ["segmented", "characters"]:
        seg_dir = project_dir / seg_name
        if seg_dir.exists():
            result.has_segmented = True
            break

    # Check clustered
    for clust_name in ["clustered", "character_clusters", "clustered_filtered"]:
        clust_dir = project_dir / clust_name
        if clust_dir.exists():
            result.has_clustered = True
            chars = list_character_dirs(clust_dir)
            if chars:
                result.detected_characters = chars
            break

    # Check lora_data
    lora_dir = project_dir / "lora_data"
    result.has_lora_data = lora_dir.exists()
    if result.has_lora_data and not result.detected_characters:
        # Try to get characters from lora_data structure
        chars_dir = lora_dir / "characters_inpainted"
        if chars_dir.exists():
            result.detected_characters = list_character_dirs(chars_dir)

        # Or from training_data
        train_dir = lora_dir / "training_data"
        if train_dir.exists() and not result.detected_characters:
            result.detected_characters = list_character_dirs(train_dir)

    return result


def generate_config_content(project: DetectedProject) -> Dict[str, Any]:
    """
    Generate configuration content for a project.

    Args:
        project: Detected project information

    Returns:
        Configuration dictionary
    """
    # Get appropriate defaults based on animation type
    if project.animation_type == "3d":
        alpha_threshold = 0.15
        blur_threshold = 80
        min_cluster_size = 12
        min_samples = 2
        target_size = 400
        caption_prefix = "a 3d animated character, pixar style, smooth shading, studio lighting"
    else:  # 2d
        alpha_threshold = 0.25
        blur_threshold = 100
        min_cluster_size = 20
        min_samples = 3
        target_size = 500
        caption_prefix = "a 2d animated character, western animation, cel shading"

    # Build character list
    characters = []
    for char_name in project.detected_characters[:10]:  # Limit to 10
        # Clean up character name for trigger word
        trigger = char_name.lower().replace("_", " ").replace("-", " ")
        characters.append({
            "name": char_name,
            "trigger_word": f"{trigger} character",
            "description": f"Character: {char_name}"
        })

    # If no characters detected, add placeholder
    if not characters:
        characters = [{
            "name": "character_1",
            "trigger_word": "character1_trigger",
            "description": "Add character description here"
        }]

    config = {
        "project": {
            "name": project.name,
            "animation_type": project.animation_type,
            "description": f"Auto-generated config for {project.name}",
            "characters": characters
        },
        "paths": {
            "base_dir": f"/mnt/data/datasets/general/{project.name}",
            "frames": "${paths.base_dir}/frames",
            "segmented": "${paths.base_dir}/segmented",
            "clustered": "${paths.base_dir}/clustered",
            "training_data": f"/mnt/data/training/lora/{'3d' if project.animation_type == '3d' else '2d'}_characters/{project.name}",
            "lora_output": f"/mnt/c/ai_models/lora/{project.name}",
            "lora_sdxl_output": f"/mnt/c/ai_models/lora_sdxl/{project.name}",
        },
        "multi_character_extraction": {
            "tracking": {
                "use_stub": False,
                "backend": "pytorch",
                "model": "yolov11n",
                "conf_threshold": 0.25,
                "min_track_length": 10
            },
            "segmentation": {
                "use_stub": False,
                "backend": "toonout" if project.animation_type == "2d" else "mobile_sam",
                "alpha_threshold": alpha_threshold,
                "blur_threshold": blur_threshold
            },
            "clustering": {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "similarity_threshold": 0.75 if project.animation_type == "2d" else 0.70,
                "use_face_detection": True,
                "device": "cuda"
            }
        },
        "dataset": {
            "target_size": target_size,
            "dedup_threshold": 0.92 if project.animation_type == "2d" else 0.95,
            "quality_min_score": 0.6,
            "generate_captions": True,
            "caption_prefix": caption_prefix
        },
        "training": {
            "base_model": "sdxl_base",
            "epochs": 10,
            "learning_rate": 0.0001,
            "text_encoder_lr": 0.00006,
            "batch_size": 2,
            "dropout": 0.0,
            "save_every_n_epochs": 2
        }
    }

    return config


def save_config(config: Dict, output_path: Path, force: bool = False) -> bool:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration dictionary
        output_path: Output file path
        force: Overwrite existing file

    Returns:
        True if saved, False if skipped
    """
    if output_path.exists() and not force:
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Add header comment
    yaml_content = f"""# Project Configuration for {config['project']['name']}
# Auto-generated by generate_project_configs.py
# Animation Type: {config['project']['animation_type']}
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}
#
# Review and customize this configuration before use.

"""
    yaml_content += yaml.dump(config, default_flow_style=False, sort_keys=False, allow_unicode=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    return True


def generate_all_configs(
    dataset_root: Path,
    config_output_dir: Path,
    num_threads: int = 32,
    force: bool = False,
    verbose: bool = False,
    project_filter: Optional[str] = None
) -> Dict:
    """
    Generate configurations for all detected projects.

    Args:
        dataset_root: Root directory containing project datasets
        config_output_dir: Directory to save generated configs
        num_threads: Number of threads for parallel processing
        force: Overwrite existing configs
        verbose: Print detailed output
        project_filter: Only process specific project

    Returns:
        Generation summary dictionary
    """
    # Find project directories
    if project_filter:
        project_dirs = [dataset_root / project_filter]
        if not project_dirs[0].exists():
            return {"error": f"Project not found: {project_filter}"}
    else:
        project_dirs = [
            d for d in dataset_root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ]

    print(f"Scanning {len(project_dirs)} project directories...")
    start_time = time.time()

    # Detect projects in parallel
    projects: List[DetectedProject] = []

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(detect_project, d): d for d in project_dirs}

        for future in as_completed(futures):
            project = future.result()
            projects.append(project)

            if verbose:
                chars = len(project.detected_characters)
                print(f"  Detected: {project.name} ({project.animation_type}, {project.frame_count} frames, {chars} chars)")

    # Generate configs
    print(f"\nGenerating configurations...")
    generated = 0
    skipped = 0
    errors = []

    for project in projects:
        try:
            config = generate_config_content(project)
            output_path = config_output_dir / f"{project.name}.yaml"

            if save_config(config, output_path, force):
                generated += 1
                if verbose:
                    print(f"  ✅ Generated: {project.name}.yaml")
            else:
                skipped += 1
                if verbose:
                    print(f"  ⏭️  Skipped (exists): {project.name}.yaml")

        except Exception as e:
            errors.append(f"{project.name}: {str(e)[:100]}")
            if verbose:
                print(f"  ❌ Error: {project.name}: {e}")

    elapsed = time.time() - start_time

    summary = {
        "total_projects": len(projects),
        "configs_generated": generated,
        "configs_skipped": skipped,
        "errors": len(errors),
        "elapsed_seconds": round(elapsed, 2),
        "projects": [
            {
                "name": p.name,
                "animation_type": p.animation_type,
                "has_frames": p.has_frames,
                "frame_count": p.frame_count,
                "characters": p.detected_characters
            }
            for p in sorted(projects, key=lambda x: x.name)
        ],
        "error_details": errors
    }

    return summary


def print_summary(summary: Dict) -> None:
    """Print generation summary."""
    if "error" in summary:
        print(f"\n❌ Error: {summary['error']}")
        return

    print("\n" + "=" * 70)
    print("Project Configuration Generation Summary")
    print("=" * 70)

    print(f"\nTotal projects scanned: {summary['total_projects']}")
    print(f"Configurations generated: {summary['configs_generated']} ✅")
    print(f"Configurations skipped: {summary['configs_skipped']} ⏭️")
    print(f"Errors: {summary['errors']} ❌")
    print(f"Time elapsed: {summary['elapsed_seconds']}s")

    # Project table
    print("\n" + "-" * 70)
    print(f"{'Project':<25} {'Type':<6} {'Frames':>8} {'Characters':<20}")
    print("-" * 70)

    for p in summary['projects'][:20]:  # Limit to 20
        chars = ", ".join(p['characters'][:3])
        if len(p['characters']) > 3:
            chars += f" (+{len(p['characters']) - 3})"
        print(f"{p['name']:<25} {p['animation_type']:<6} {p['frame_count']:>8} {chars:<20}")

    if len(summary['projects']) > 20:
        print(f"... and {len(summary['projects']) - 20} more projects")

    if summary['error_details']:
        print("\n" + "-" * 70)
        print("Errors:")
        for err in summary['error_details'][:5]:
            print(f"  ❌ {err}")

    print("\n" + "=" * 70)
    if summary['configs_generated'] > 0:
        print(f"✅ Generated {summary['configs_generated']} configuration file(s)")
        print(f"   Output directory: configs/projects/")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate project configurations from dataset directories"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="/mnt/data/datasets/general",
        help="Root directory containing project datasets"
    )
    parser.add_argument(
        "--config-output",
        type=str,
        default=None,
        help="Output directory for configs (default: configs/projects/)"
    )
    parser.add_argument(
        "--project",
        type=str,
        help="Generate config for specific project only"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=32,
        help="Number of threads for parallel processing (default: 32)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing configuration files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Save generation report to JSON file"
    )

    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    config_output = Path(args.config_output) if args.config_output else CONFIG_ROOT / "projects"

    if not dataset_root.exists():
        print(f"Error: Dataset root not found: {dataset_root}")
        sys.exit(1)

    print(f"Dataset root: {dataset_root}")
    print(f"Config output: {config_output}")
    print(f"Using {args.threads} threads")
    if args.force:
        print("Force mode: Will overwrite existing configs")

    # Generate configs
    summary = generate_all_configs(
        dataset_root,
        config_output,
        args.threads,
        args.force,
        args.verbose,
        args.project
    )

    # Print summary
    print_summary(summary)

    # Save report if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
