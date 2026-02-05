#!/usr/bin/env python
"""
CLI for stub character LoRA dataset building.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.datasets.character_builder import CharacterDatasetConfig, build_character_dataset
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build character LoRA dataset (stub).")
    parser.add_argument("--config", type=str, default="configs/build_lora_dataset_characters.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Force stub mode.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path, CharacterDatasetConfig) if config_path.exists() else CharacterDatasetConfig()
    if args.use_stub:
        cfg.use_stub = True
    logger = setup_logging("build_lora_dataset_characters", cfg.log_dir)
    build_character_dataset(cfg, logger)


if __name__ == "__main__":
    main()

