#!/usr/bin/env python
"""
CLI for stub-friendly pose extraction.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.pose.dwpose_wrapper import PoseExtractConfig, extract_poses
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract poses from detections (stub capable).")
    parser.add_argument("--config", type=str, default="configs/extract_pose.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Force stub mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path, PoseExtractConfig) if config_path.exists() else PoseExtractConfig()
    if args.use_stub:
        cfg.use_stub = True
    logger = setup_logging("extract_pose", cfg.log_dir)
    extract_poses(cfg, logger)


if __name__ == "__main__":
    main()

