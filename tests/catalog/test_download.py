"""Tests for asset downloads."""

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from src.catalog.download import download_asset, download_assets
from src.catalog.models import Asset, Item
from src.core.exceptions import DownloadError, IncompleteDownloadError

CONTENT = b"hyperspectral-cube-bytes"
FILENAME = "scene_ortho_visual.tif"
CHUNK_SIZE = 8


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
    """Build a client that answers from `handler` instead of the network."""
    return httpx.Client(transport=httpx.MockTransport(handler))


def _serve(content: bytes) -> Callable[[httpx.Request], httpx.Response]:
    """Answer every request with the whole content."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(httpx.codes.OK, content=content)

    return handler


def _asset(filename: str = FILENAME) -> Asset:
    return Asset(href=f"https://example.test/{filename}")


def test_download_asset_writes_the_file(tmp_path: Path) -> None:
    with _client(_serve(CONTENT)) as client:
        path = download_asset(_asset(), tmp_path, client=client, chunk_size=CHUNK_SIZE)

    assert path == tmp_path / FILENAME
    assert path.read_bytes() == CONTENT


def test_download_asset_leaves_no_partial_file_behind(tmp_path: Path) -> None:
    with _client(_serve(CONTENT)) as client:
        download_asset(_asset(), tmp_path, client=client, chunk_size=CHUNK_SIZE)

    assert list(tmp_path.glob("*.part")) == []


def test_download_asset_creates_the_destination_directory(tmp_path: Path) -> None:
    destination_dir = tmp_path / "data" / "scene"

    with _client(_serve(CONTENT)) as client:
        download_asset(
            _asset(),
            destination_dir,
            client=client,
            chunk_size=CHUNK_SIZE,
        )

    assert (destination_dir / FILENAME).exists()


def test_download_asset_keeps_a_file_that_is_already_there(tmp_path: Path) -> None:
    (tmp_path / FILENAME).write_bytes(b"downloaded earlier")

    def handler(request: httpx.Request) -> httpx.Response:
        pytest.fail("an asset that is already on disk was requested again")

    with _client(handler) as client:
        path = download_asset(_asset(), tmp_path, client=client, chunk_size=CHUNK_SIZE)

    assert path.read_bytes() == b"downloaded earlier"


def test_download_asset_replaces_the_file_when_asked_to_overwrite(
    tmp_path: Path,
) -> None:
    (tmp_path / FILENAME).write_bytes(b"downloaded earlier")

    with _client(_serve(CONTENT)) as client:
        path = download_asset(
            _asset(),
            tmp_path,
            client=client,
            chunk_size=CHUNK_SIZE,
            overwrite=True,
        )

    assert path.read_bytes() == CONTENT


def test_download_asset_resumes_from_an_interrupted_attempt(tmp_path: Path) -> None:
    (tmp_path / f"{FILENAME}.part").write_bytes(CONTENT[:10])
    requested_ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_ranges.append(request.headers.get("Range"))
        return httpx.Response(httpx.codes.PARTIAL_CONTENT, content=CONTENT[10:])

    with _client(handler) as client:
        path = download_asset(_asset(), tmp_path, client=client, chunk_size=CHUNK_SIZE)

    assert requested_ranges == ["bytes=10-"]
    assert path.read_bytes() == CONTENT


def test_download_asset_starts_over_when_the_server_refuses_to_resume(
    tmp_path: Path,
) -> None:
    (tmp_path / f"{FILENAME}.part").write_bytes(b"stale bytes")

    with _client(_serve(CONTENT)) as client:
        path = download_asset(_asset(), tmp_path, client=client, chunk_size=CHUNK_SIZE)

    assert path.read_bytes() == CONTENT


def test_download_asset_discards_a_partial_file_when_overwriting(
    tmp_path: Path,
) -> None:
    (tmp_path / f"{FILENAME}.part").write_bytes(CONTENT[:10])
    requested_ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_ranges.append(request.headers.get("Range"))
        return httpx.Response(httpx.codes.OK, content=CONTENT)

    with _client(handler) as client:
        path = download_asset(
            _asset(),
            tmp_path,
            client=client,
            chunk_size=CHUNK_SIZE,
            overwrite=True,
        )

    assert requested_ranges == [None]
    assert path.read_bytes() == CONTENT


def test_download_asset_reports_a_failed_request(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(httpx.codes.INTERNAL_SERVER_ERROR)

    with (
        _client(handler) as client,
        pytest.raises(DownloadError, match=FILENAME),
    ):
        download_asset(_asset(), tmp_path, client=client, chunk_size=CHUNK_SIZE)

    assert not (tmp_path / FILENAME).exists()


def test_download_asset_starts_over_when_the_partial_file_is_unusable(
    tmp_path: Path,
) -> None:
    (tmp_path / f"{FILENAME}.part").write_bytes(CONTENT)
    requested_ranges: list[str | None] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_ranges.append(request.headers.get("Range"))
        if request.headers.get("Range"):
            return httpx.Response(httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE)
        return httpx.Response(httpx.codes.OK, content=CONTENT)

    with _client(handler) as client:
        path = download_asset(_asset(), tmp_path, client=client, chunk_size=CHUNK_SIZE)

    assert requested_ranges == [f"bytes={len(CONTENT)}-", None]
    assert path.read_bytes() == CONTENT


def test_download_asset_reports_a_response_that_ends_early(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            httpx.codes.OK,
            headers={"Content-Length": str(len(CONTENT))},
            content=iter([CONTENT[:5]]),
        )

    with (
        _client(handler) as client,
        pytest.raises(IncompleteDownloadError, match="5 of 24 bytes"),
    ):
        download_asset(_asset(), tmp_path, client=client, chunk_size=CHUNK_SIZE)

    assert not (tmp_path / FILENAME).exists()


def test_download_asset_keeps_what_arrived_so_it_can_resume(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            httpx.codes.OK,
            headers={"Content-Length": str(len(CONTENT))},
            content=iter([CONTENT[:5]]),
        )

    with _client(handler) as client, pytest.raises(IncompleteDownloadError):
        download_asset(_asset(), tmp_path, client=client, chunk_size=CHUNK_SIZE)

    assert (tmp_path / f"{FILENAME}.part").read_bytes() == CONTENT[:5]


def test_download_asset_accepts_a_response_without_an_announced_size(
    tmp_path: Path,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(httpx.codes.OK, content=iter([CONTENT]))

    with _client(handler) as client:
        path = download_asset(_asset(), tmp_path, client=client, chunk_size=CHUNK_SIZE)

    assert path.read_bytes() == CONTENT


def test_download_assets_groups_the_files_by_scene(tmp_path: Path) -> None:
    item = Item.model_validate(
        {
            "id": "scene-1",
            "assets": {
                "ortho_visual": {"href": "https://example.test/visual.tif"},
                "thumbnail": {"href": "https://example.test/thumb.png"},
            },
        },
    )

    with _client(_serve(CONTENT)) as client:
        paths = download_assets(
            item,
            ["ortho_visual", "thumbnail"],
            destination_dir=tmp_path,
            client=client,
            chunk_size=CHUNK_SIZE,
        )

    assert paths == {
        "ortho_visual": tmp_path / "scene-1" / "visual.tif",
        "thumbnail": tmp_path / "scene-1" / "thumb.png",
    }
    assert all(path.read_bytes() == CONTENT for path in paths.values())
