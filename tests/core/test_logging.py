"""Tests for logging configuration."""

import pytest
from loguru import logger

from src.core.config import LogLevel, Settings
from src.core.logging import configure_logging


def test_configure_logging_writes_to_stdout(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(Settings(log_format="{level} | {message}"))

    logger.info("hello")

    assert "INFO | hello" in capsys.readouterr().out


def test_configure_logging_honours_the_configured_level(
    capsys: pytest.CaptureFixture[str],
) -> None:
    configure_logging(
        Settings(log_level=LogLevel.WARNING, log_format="{message}"),
    )

    logger.debug("below the threshold")
    logger.warning("above the threshold")

    captured = capsys.readouterr().out
    assert "below the threshold" not in captured
    assert "above the threshold" in captured
