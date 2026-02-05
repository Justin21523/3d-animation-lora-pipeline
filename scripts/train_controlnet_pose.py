#!/usr/bin/env python
"""
CLI for stubbed ControlNet pose training (CPU friendly).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.training.controlnet_trainer import ControlNetTrainingConfig, train_controlnet_pose
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ControlNet pose (stub).")
    parser.add_argument("--config", type=str, default="configs/train_controlnet_pose.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Force stub data/model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path, ControlNetTrainingConfig) if config_path.exists() else ControlNetTrainingConfig()
    if args.use_stub:
        cfg.use_stub = True
    logger = setup_logging("train_controlnet_pose", cfg.log_dir)
    result = train_controlnet_pose(cfg, logger)
    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()

