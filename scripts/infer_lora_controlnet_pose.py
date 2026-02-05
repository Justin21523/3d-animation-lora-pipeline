#!/usr/bin/env python
"""
CLI for stubbed LoRA + ControlNet pose inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.inference.lora_controlnet import InferenceConfig, run_inference
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LoRA + ControlNet pose inference (stub).")
    parser.add_argument("--config", type=str, default="configs/infer_lora_controlnet_pose.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Force stub mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path, InferenceConfig) if config_path.exists() else InferenceConfig()
    if args.use_stub:
        cfg.use_stub = True
    logger = setup_logging("infer_lora_controlnet_pose", cfg.log_dir)
    run_inference(cfg, logger)


if __name__ == "__main__":
    main()

