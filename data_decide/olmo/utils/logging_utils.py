"""Logging utilities for OLMo training."""

import logging
import os
import sys
from pathlib import Path
from typing import Optional


def setup_logging(output_dir: str, log_level: str = "INFO", log_file: str = "training.log") -> None:
    """
    Setup logging configuration for training.

    Args:
        output_dir: Directory to save log files
        log_level: Logging level (INFO, DEBUG, etc.)
        log_file: Name of the log file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Configure logging format
    log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # Get root logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(console_handler)

    # File handler
    log_path = Path(output_dir) / log_file
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(logging.Formatter(log_format, date_format))
    logger.addHandler(file_handler)

    # Log initial message
    logger.info(f"Logging initialized. Log file: {log_path}")


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Name of the logger (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)
