#!/usr/bin/env python
"""
CLI for stub embedding generation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from anime_pipeline.config.loader import load_config
from anime_pipeline.embeddings.builders import EmbeddingConfig, build_embeddings
from anime_pipeline.utils.logging_utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate embeddings (stub).")
    parser.add_argument("--config", type=str, default="configs/build_embeddings.yaml", help="Path to YAML config.")
    parser.add_argument("--use-stub", action="store_true", help="Force stub vectors.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    cfg = load_config(config_path, EmbeddingConfig) if config_path.exists() else EmbeddingConfig()
    if args.use_stub:
        cfg.use_stub = True
    logger = setup_logging("build_embeddings", cfg.log_dir)
    build_embeddings(cfg, logger)


if __name__ == "__main__":
    main()

