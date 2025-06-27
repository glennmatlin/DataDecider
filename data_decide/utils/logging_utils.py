"""Logging utilities for DataDecider."""

import logging
import sys
from typing import Optional


def get_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Get a logger with consistent formatting.

    Args:
        name: Logger name (defaults to module name)
        level: Logging level

    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)

    # Only configure if no handlers exist
    if not logger.handlers:
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)

        # Create formatter
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)

        # Add handler to logger
        logger.addHandler(console_handler)
        logger.setLevel(level)

    return logger


def set_verbosity(verbose: bool = False, quiet: bool = False):
    """Set global logging verbosity.

    Args:
        verbose: Enable verbose logging (DEBUG level)
        quiet: Suppress most logging (WARNING level)
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    # Set level for all DataDecider loggers
    logging.getLogger("data_decide").setLevel(level)
