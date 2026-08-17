"""Logging configuration and utilities."""

import sys

from loguru import logger

from .config import Settings


def configure_logging(settings: Settings) -> None:
    """Configure logging based on the provided settings."""
    logger.remove()  # Remove the default logger
    logger.add(
        sink=sys.stdout,
        level=settings.log_level,
        format=settings.log_format,
        diagnose=False,
    )
