#!/usr/bin/env python
"""
CLI for frame deduplication via hashing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.frames.dedupe import DedupeConfig, dedupe_frames
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deduplicate extracted frames using hashes.")
    parser.add_argument("--config", type=str, default="configs/dedupe_frames.yaml", help="Path to YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)

    if config_path.exists():
        cfg = load_config(config_path, DedupeConfig)
    else:
        cfg = DedupeConfig()

    logger = setup_logging("dedupe_frames", cfg.log_dir)
    dedupe_frames(cfg, logger)


if __name__ == "__main__":
    main()

