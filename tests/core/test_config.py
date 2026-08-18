"""Tests for application settings."""

import pytest
from pydantic import ValidationError

from src.core.config import LogLevel, Settings, get_settings


def test_settings_fall_back_to_defaults() -> None:
    settings = Settings()

    assert settings.log_level is LogLevel.INFO
    assert "{message}" in settings.log_format


def test_settings_read_log_level_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    assert Settings().log_level is LogLevel.DEBUG


def test_settings_reject_unknown_options() -> None:
    with pytest.raises(ValidationError):
        Settings(unknown_option="value")  # type: ignore[call-arg]


def test_settings_build_the_item_url_from_the_scene_coordinates() -> None:
    settings = Settings(
        catalog_base_url="https://example.test/stac/",
        scene_collection="energy-mining",
        scene_id="scene-1",
    )

    assert settings.item_url == (
        "https://example.test/stac/energy-mining/scene-1/scene-1.json"
    )


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()

    assert get_settings() is get_settings()
