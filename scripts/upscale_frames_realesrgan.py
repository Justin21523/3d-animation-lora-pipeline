#!/usr/bin/env python
"""
CLI for stubbed Real-ESRGAN upscaling.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.restoration.realesrgan_wrapper import RealESRGANConfig, upscale_frames
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upscale frames with Real-ESRGAN (stub).")
    parser.add_argument("--config", type=str, default="configs/upscale_realesrgan.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Force stub upscaling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path, RealESRGANConfig) if config_path.exists() else RealESRGANConfig()
    if args.use_stub:
        cfg.use_stub = True
    logger = setup_logging("upscale_frames", cfg.log_dir)
    upscale_frames(cfg, logger)


if __name__ == "__main__":
    main()

