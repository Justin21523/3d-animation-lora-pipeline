#!/usr/bin/env python
"""
CLI for stub-friendly YOLO detection + tracking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.detection.yolo_detector import YoloTrackingConfig, run_yolo_tracking
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO detection + tracking (stub capable).")
    parser.add_argument("--config", type=str, default="configs/run_yolo_tracking.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Force stub outputs regardless of model availability.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path, YoloTrackingConfig) if config_path.exists() else YoloTrackingConfig()
    if args.use_stub:
        cfg.use_stub = True
    logger = setup_logging("run_yolo_tracking", cfg.log_dir)
    run_yolo_tracking(cfg, logger)


if __name__ == "__main__":
    main()

