#!/usr/bin/env python
"""
CLI for frame extraction (ffmpeg or stub mode).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.frames.sampling import ExtractFramesConfig, extract_frames
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract frames from videos or generate stub frames.")
    parser.add_argument("--config", type=str, default="configs/extract_frames.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Override to force stub mode.")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting existing frames.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if config_path.exists():
        cfg = load_config(config_path, ExtractFramesConfig)
    else:
        cfg = ExtractFramesConfig()

    if args.use_stub:
        cfg.use_stub = True
    if args.overwrite:
        cfg.overwrite = True

    logger = setup_logging("extract_frames", cfg.log_dir)
    extract_frames(cfg, logger)


if __name__ == "__main__":
    main()

