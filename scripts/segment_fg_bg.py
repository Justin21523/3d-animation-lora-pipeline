#!/usr/bin/env python
"""
CLI for stub-friendly foreground/background segmentation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.segmentation.toonout_wrapper import SegmentConfig, segment_foreground_background
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment foreground/background using ToonOut stub.")
    parser.add_argument("--config", type=str, default="configs/segment_fg_bg.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Force stub mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path, SegmentConfig) if config_path.exists() else SegmentConfig()
    if args.use_stub:
        cfg.use_stub = True
    logger = setup_logging("segment_fg_bg", cfg.log_dir)
    segment_foreground_background(cfg, logger)


if __name__ == "__main__":
    main()

