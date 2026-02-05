#!/usr/bin/env python
"""
CLI for stubbed LoRA training (CPU friendly).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.training.lora_trainer_sd import LoRATrainingConfig, train_lora_sd
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LoRA (stub mode by default).")
    parser.add_argument("--config", type=str, default="configs/train_lora_sd.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Force stub data/model.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path, LoRATrainingConfig) if config_path.exists() else LoRATrainingConfig()
    if args.use_stub:
        cfg.use_stub = True
    logger = setup_logging("train_lora_sd", cfg.log_dir)
    result = train_lora_sd(cfg, logger)
    logger.info("Result: %s", result)


if __name__ == "__main__":
    main()

