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
    )

    assert settings.item_url("scene-1") == (
        "https://example.test/stac/energy-mining/scene-1/scene-1.json"
    )


def test_settings_reject_an_empty_scene_list() -> None:
    with pytest.raises(ValidationError):
        Settings(scene_ids=())


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()

    assert get_settings() is get_settings()


def test_settings_list_every_species_in_group_order() -> None:
    settings = Settings(
        mineral_groups={"alunite": ("A",), "kaolinite_group": ("K", "D")},
    )

    assert settings.species == ("A", "K", "D")


def test_settings_report_the_group_a_species_belongs_to() -> None:
    settings = Settings(mineral_groups={"kaolinite_group": ("K", "D")})

    assert settings.group_of("D") == "kaolinite_group"
    with pytest.raises(KeyError):
        settings.group_of("A")


def test_settings_reject_a_species_filed_under_two_groups() -> None:
    """The groups are the whole mineral list, so a species has one home."""
    with pytest.raises(ValidationError):
        Settings(mineral_groups={"alunite": ("A",), "carbonate": ("A",)})


def test_settings_reject_a_group_with_no_species() -> None:
    with pytest.raises(ValidationError):
        Settings(mineral_groups={"alunite": ()})


def test_settings_reject_having_no_groups_at_all() -> None:
    with pytest.raises(ValidationError):
        Settings(mineral_groups={})


def test_settings_reject_a_match_range_that_ends_before_it_starts() -> None:
    with pytest.raises(ValidationError):
        Settings(match_range=(2400.0, 2080.0))
