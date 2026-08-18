"""Tests for catalog access."""

from collections.abc import Callable

import httpx
import pytest

from src.catalog.client import build_client, fetch_item
from src.core.config import Settings
from src.core.exceptions import ItemFetchError

ITEM_URL = "https://example.test/energy-mining/scene/scene.json"
UNREACHABLE = "the catalog is unreachable"
ITEM_DOCUMENT = {
    "id": "scene",
    "stac_version": "1.1.0",
    "bbox": [82.5, 24.0, 82.8, 24.2],
    "properties": {
        "datetime": "2025-03-05T05:34:21.321891Z",
        "location_description": "Singrauli, Madhya Pradesh, India",
    },
    "assets": {"ortho_visual": {"href": "https://example.test/visual.tif"}},
}


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """Build a client that answers from `handler` instead of the network."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_build_client_follows_the_redirect_to_the_storage_bucket() -> None:
    settings = Settings(request_timeout=12.0)

    with build_client(settings) as client:
        assert client.follow_redirects is True
        assert client.timeout.read == 12.0


def test_fetch_item_parses_the_scene_metadata() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(httpx.codes.OK, json=ITEM_DOCUMENT)

    with _client(handler) as client:
        item = fetch_item(ITEM_URL, client=client)

    assert item.id == "scene"
    assert item.location == "Singrauli, Madhya Pradesh, India"
    assert item.asset("ortho_visual").filename == "visual.tif"


def test_fetch_item_requests_the_url_it_was_given() -> None:
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(httpx.codes.OK, json=ITEM_DOCUMENT)

    with _client(handler) as client:
        fetch_item(ITEM_URL, client=client)

    assert requested == [ITEM_URL]


def test_fetch_item_reports_a_missing_item() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(httpx.codes.NOT_FOUND)

    with _client(handler) as client, pytest.raises(ItemFetchError, match=ITEM_URL):
        fetch_item(ITEM_URL, client=client)


def test_fetch_item_reports_an_unreachable_catalog() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(UNREACHABLE)

    with _client(handler) as client, pytest.raises(ItemFetchError, match=ITEM_URL):
        fetch_item(ITEM_URL, client=client)
