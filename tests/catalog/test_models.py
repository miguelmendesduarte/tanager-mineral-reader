"""Tests for the STAC models."""

import pytest

from src.catalog.models import Asset, Item
from src.core.exceptions import AssetNotFoundError


def test_asset_derives_its_filename_from_the_href() -> None:
    asset = Asset(href="https://example.test/a/b/scene_ortho_visual.tif")

    assert asset.filename == "scene_ortho_visual.tif"


def test_asset_filename_ignores_the_query_string_of_a_signed_href() -> None:
    asset = Asset(href="https://example.test/cube.h5?X-Goog-Signature=abc123")

    assert asset.filename == "cube.h5"


def test_item_ignores_catalog_fields_the_pipeline_does_not_use() -> None:
    item = Item.model_validate(
        {"id": "scene", "stac_version": "1.1.0", "collection": "energy-mining"},
    )

    assert item.id == "scene"


def test_item_exposes_the_properties_the_report_needs() -> None:
    item = Item.model_validate(
        {
            "id": "scene",
            "properties": {
                "datetime": "2025-03-05T05:34:21.321891Z",
                "location_description": "Singrauli, Madhya Pradesh, India",
            },
        },
    )

    assert item.acquired_at == "2025-03-05T05:34:21.321891Z"
    assert item.location == "Singrauli, Madhya Pradesh, India"


def test_item_properties_are_none_when_the_catalog_omits_them() -> None:
    item = Item.model_validate({"id": "scene"})

    assert item.acquired_at is None
    assert item.location is None


def test_item_looks_up_an_asset_by_catalog_key() -> None:
    item = Item.model_validate(
        {
            "id": "scene",
            "assets": {"ortho_visual": {"href": "https://example.test/visual.tif"}},
        },
    )

    assert item.asset("ortho_visual").href == "https://example.test/visual.tif"


def test_item_lists_the_alternatives_when_an_asset_is_missing() -> None:
    item = Item.model_validate(
        {
            "id": "scene",
            "assets": {"ortho_visual": {"href": "https://example.test/visual.tif"}},
        },
    )

    with pytest.raises(AssetNotFoundError, match="ortho_visual"):
        item.asset("ortho_radiance_hdf5")
