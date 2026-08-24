"""Tests for access to the USGS spectral library."""

from collections.abc import Callable
from typing import Any

import httpx
import pytest

from src.core.exceptions import ArchiveSizeError
from src.spectra.splib07 import archive_size

URL = "https://example.test/item/abc?format=json"
ARCHIVE = "usgs_splib07.zip"
SIZE = 5479324354


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """Build a client that answers from `handler` instead of the network."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def _serve(payload: Any, status: int = httpx.codes.OK) -> Callable[..., httpx.Response]:
    """Answer every request with the same JSON document."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json=payload)

    return handler


def _record(*files: dict[str, Any]) -> dict[str, Any]:
    """A catalog record listing the given files."""
    return {"title": "USGS Spectral Library", "files": list(files)}


def test_archive_size_reads_the_size_of_the_named_archive() -> None:
    record = _record(
        {"name": "thumbnail.jpg", "size": 4900000},
        {"name": ARCHIVE, "size": SIZE},
    )

    with _client(_serve(record)) as client:
        assert archive_size(URL, ARCHIVE, client=client) == SIZE


def test_archive_size_rejects_a_record_without_that_archive() -> None:
    with (
        _client(_serve(_record({"name": "other.zip", "size": 1}))) as client,
        pytest.raises(ArchiveSizeError),
    ):
        archive_size(URL, ARCHIVE, client=client)


def test_archive_size_rejects_an_archive_with_no_size() -> None:
    record = _record({"name": ARCHIVE, "size": None})

    with _client(_serve(record)) as client, pytest.raises(ArchiveSizeError):
        archive_size(URL, ARCHIVE, client=client)


def test_archive_size_rejects_a_record_that_cannot_be_fetched() -> None:
    handler = _serve({}, status=httpx.codes.NOT_FOUND)

    with _client(handler) as client, pytest.raises(ArchiveSizeError):
        archive_size(URL, ARCHIVE, client=client)


def test_archive_size_rejects_a_record_that_is_not_a_catalog_record() -> None:
    with (
        _client(_serve({"unexpected": "shape"})) as client,
        pytest.raises(ArchiveSizeError),
    ):
        archive_size(URL, ARCHIVE, client=client)
