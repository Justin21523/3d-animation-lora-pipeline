#!/usr/bin/env python
"""
CLI for stubbed RIFE interpolation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.interpolation.rife_wrapper import RIFEConfig, interpolate_frames
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate frames with RIFE (stub).")
    parser.add_argument("--config", type=str, default="configs/interpolate_rife.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Force stub interpolation.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path, RIFEConfig) if config_path.exists() else RIFEConfig()
    if args.use_stub:
        cfg.use_stub = True
    logger = setup_logging("interpolate_frames", cfg.log_dir)
    interpolate_frames(cfg, logger)


if __name__ == "__main__":
    main()

