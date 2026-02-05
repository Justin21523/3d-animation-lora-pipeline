#!/usr/bin/env python3
"""
Logging Utility for Pipeline Scripts

Provides consistent logging setup across all pipeline components.
Supports both console and file logging with configurable formats and levels.

Features:
- Colored console output for different log levels
- File logging with rotation support
- Thread-safe logging operations
- Consistent timestamp formatting
- Configurable log levels and formats

Usage:
    from scripts.core.utils.logger import setup_logger

    # Basic usage (console only)
    logger = setup_logger("my_script")
    logger.info("Processing started")

    # With file logging
    logger = setup_logger(
        name="my_script",
        log_file="logs/my_script.log",
        level="DEBUG"
    )

    # With custom format
    logger = setup_logger(
        name="my_script",
        log_file="logs/my_script.log",
        level="INFO",
        console_format="simple"
    )

Author: AI Pipeline
Date: 2025-01-20
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime


# ANSI color codes for console output
class LogColors:
    """ANSI color codes for terminal output."""
    RESET = '\033[0m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'


class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to console output.

    Different log levels get different colors for better readability.
    """

    COLORS = {
        'DEBUG': LogColors.GRAY,
        'INFO': LogColors.GREEN,
        'WARNING': LogColors.YELLOW,
        'ERROR': LogColors.RED,
        'CRITICAL': LogColors.RED
    }

    def format(self, record):
        """Format log record with color codes."""
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{LogColors.RESET}"

        # Add color to message for errors and warnings
        if record.levelno >= logging.WARNING:
            record.msg = f"{self.COLORS[levelname.strip()]}{record.msg}{LogColors.RESET}"

        return super().format(record)


def setup_logger(
    name: str,
    log_file: Optional[Union[str, Path]] = None,
    level: Union[str, int] = "INFO",
    console_format: str = "detailed",
    file_format: str = "detailed",
    console_output: bool = True
) -> logging.Logger:
    """
    Set up logger with console and optional file output.

    Args:
        name: Logger name (usually script name or module name)
        log_file: Optional path to log file. If provided, logs will be written to file.
                 Parent directories will be created automatically.
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int
        console_format: Console output format:
                       - "simple": Just message
                       - "detailed": Timestamp + level + message (default)
                       - "full": Timestamp + level + name + message
        file_format: File output format (same options as console_format)
        console_output: Whether to output to console (default True)

    Returns:
        Configured logger instance

    Examples:
        # Basic console logging
        logger = setup_logger("my_script")

        # Console + file logging
        logger = setup_logger("my_script", log_file="logs/processing.log")

        # Debug level with full details
        logger = setup_logger(
            "my_script",
            log_file="logs/debug.log",
            level="DEBUG",
            console_format="full",
            file_format="full"
        )
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(_parse_level(level))

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Format strings for different detail levels
    format_templates = {
        "simple": "%(message)s",
        "detailed": "%(asctime)s | %(levelname)-8s | %(message)s",
        "full": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    }

    # Console handler (if enabled)
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(_parse_level(level))

        console_fmt = format_templates.get(console_format, format_templates["detailed"])
        console_formatter = ColoredFormatter(
            console_fmt,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)

    # File handler (if log_file provided)
    if log_file:
        log_path = Path(log_file)

        # Create parent directories
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(_parse_level(level))

        file_fmt = format_templates.get(file_format, format_templates["detailed"])
        file_formatter = logging.Formatter(
            file_fmt,
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # Log initial message to file
        logger.debug(f"Logging initialized - File: {log_path.absolute()}")

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def _parse_level(level: Union[str, int]) -> int:
    """
    Parse logging level from string or int.

    Args:
        level: Level as string (DEBUG, INFO, etc.) or int (10, 20, etc.)

    Returns:
        Logging level as integer
    """
    if isinstance(level, int):
        return level

    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    return level_map.get(level.upper(), logging.INFO)


def get_timestamped_log_path(base_dir: Union[str, Path], script_name: str) -> Path:
    """
    Generate timestamped log file path.

    Creates log path in format: base_dir/script_name_YYYYMMDD_HHMMSS.log

    Args:
        base_dir: Base directory for logs
        script_name: Name of the script (without extension)

    Returns:
        Path object for timestamped log file

    Example:
        log_path = get_timestamped_log_path("logs/clustering", "character_clustering")
        # Returns: logs/clustering/character_clustering_20250120_143022.log
    """
    base_dir = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{script_name}_{timestamp}.log"

    return base_dir / log_filename


def add_file_handler(
    logger: logging.Logger,
    log_file: Union[str, Path],
    level: Optional[Union[str, int]] = None,
    format_type: str = "detailed"
) -> None:
    """
    Add file handler to existing logger.

    Useful for adding file logging after logger creation.

    Args:
        logger: Existing logger instance
        log_file: Path to log file
        level: Optional logging level (inherits from logger if not specified)
        format_type: Format type (simple, detailed, full)

    Example:
        logger = setup_logger("my_script")
        # ... later ...
        add_file_handler(logger, "logs/additional.log")
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')

    if level:
        file_handler.setLevel(_parse_level(level))
    else:
        file_handler.setLevel(logger.level)

    format_templates = {
        "simple": "%(message)s",
        "detailed": "%(asctime)s | %(levelname)-8s | %(message)s",
        "full": "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    }

    file_fmt = format_templates.get(format_type, format_templates["detailed"])
    file_formatter = logging.Formatter(file_fmt, datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.addHandler(file_handler)


# Example usage
if __name__ == "__main__":
    # Demo different logging configurations

    print("=== Demo 1: Simple console logging ===")
    logger1 = setup_logger("demo1", console_format="simple")
    logger1.debug("Debug message (won't show, level is INFO)")
    logger1.info("Info message")
    logger1.warning("Warning message")
    logger1.error("Error message")

    print("\n=== Demo 2: Detailed console logging with DEBUG level ===")
    logger2 = setup_logger("demo2", level="DEBUG", console_format="detailed")
    logger2.debug("Debug message")
    logger2.info("Info message")
    logger2.warning("Warning message")
    logger2.error("Error message")

    print("\n=== Demo 3: Console + file logging ===")
    log_file = get_timestamped_log_path("logs/demo", "test_logger")
    logger3 = setup_logger("demo3", log_file=log_file, level="DEBUG")
    logger3.info(f"Logging to file: {log_file}")
    logger3.debug("Debug message")
    logger3.info("Info message")
    logger3.warning("Warning message")
    logger3.error("Error message")
    print(f"Log file created at: {log_file}")
