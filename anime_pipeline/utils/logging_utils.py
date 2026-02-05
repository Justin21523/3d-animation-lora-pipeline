"""
Logging setup for CLI scripts and libraries.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from .path_utils import ensure_dir


def setup_logging(name: str, log_dir: Optional[str | Path] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Initialize a logger with console + optional file handlers.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_dir:
        log_dir_path = ensure_dir(log_dir)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        log_file = log_dir_path / f"{name}-{timestamp}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

