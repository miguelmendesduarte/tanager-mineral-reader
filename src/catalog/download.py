"""Downloading of STAC assets.

The hyperspectral cubes are hundreds of megabytes each, so downloads stream to
disk in chunks and resume from a partial file rather than starting over.
"""

from collections.abc import Iterable
from pathlib import Path

import httpx
from loguru import logger

from ..core.exceptions import DownloadError, IncompleteDownloadError
from .models import Asset, Item

PARTIAL_SUFFIX = ".part"
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * BYTES_PER_KB
_LOG_EVERY_BYTES = 100 * BYTES_PER_MB


def download_assets(
    item: Item,
    names: Iterable[str],
    *,
    destination_dir: Path,
    client: httpx.Client,
    chunk_size: int,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download several assets of an item into a directory of its own.

    Args:
        item: Item the assets belong to.
        names: Catalog keys of the assets to download.
        destination_dir: Root data directory; a subdirectory named after the
            item is created inside it.
        client: HTTP client used for the requests.
        chunk_size: Number of bytes held in memory at a time.
        overwrite: Download again even if a file is already present.

    Returns:
        dict[str, Path]: Local path of each downloaded asset, keyed by its
            catalog key.

    Raises:
        AssetNotFoundError: If the item does not expose one of the assets.
        DownloadError: If an asset cannot be retrieved.
    """
    item_dir = destination_dir / item.id
    return {
        name: download_asset(
            item.asset(name),
            item_dir,
            client=client,
            chunk_size=chunk_size,
            overwrite=overwrite,
        )
        for name in names
    }


def download_asset(
    asset: Asset,
    destination_dir: Path,
    *,
    client: httpx.Client,
    chunk_size: int,
    overwrite: bool = False,
) -> Path:
    """Download a single asset, resuming a previous attempt when possible.

    Args:
        asset: Asset to download.
        destination_dir: Directory the file is written to; created if missing.
        client: HTTP client used for the request.
        chunk_size: Number of bytes held in memory at a time.
        overwrite: Download again even if the file is already present.

    Returns:
        Path: Location of the downloaded file.

    Raises:
        DownloadError: If the asset cannot be retrieved, or if the response ends
            before the whole file has arrived.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / asset.filename

    if destination.exists() and not overwrite:
        logger.info("Skipping {}, already downloaded", asset.filename)
        return destination

    partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
    if overwrite:
        partial.unlink(missing_ok=True)

    try:
        _stream_to_file(asset.href, partial, client=client, chunk_size=chunk_size)
    except httpx.HTTPError as error:
        raise DownloadError(asset.href) from error

    partial.replace(destination)
    size = _human_readable(destination.stat().st_size)
    logger.info("Downloaded {} ({})", asset.filename, size)
    return destination


def _human_readable(size_in_bytes: int) -> str:
    """Render a byte count in the largest unit that keeps it above one."""
    if size_in_bytes >= BYTES_PER_MB:
        return f"{size_in_bytes / BYTES_PER_MB:.1f} MB"
    if size_in_bytes >= BYTES_PER_KB:
        return f"{size_in_bytes / BYTES_PER_KB:.1f} KB"
    return f"{size_in_bytes} B"


def _stream_to_file(
    url: str,
    destination: Path,
    *,
    client: httpx.Client,
    chunk_size: int,
) -> None:
    """Stream a URL into a file, resuming from it when it already holds bytes.

    Raises:
        IncompleteDownloadError: If the response ends early. The partial file is
            left in place so that the next attempt can resume from it.
    """
    if _fetch_into(url, destination, client=client, chunk_size=chunk_size):
        return

    logger.warning("The partial file for {} is unusable, downloading it again", url)
    destination.unlink(missing_ok=True)
    _fetch_into(url, destination, client=client, chunk_size=chunk_size)


def _fetch_into(
    url: str,
    destination: Path,
    *,
    client: httpx.Client,
    chunk_size: int,
) -> bool:
    """Write the response body to a file, appending when resuming.

    Returns:
        bool: False when the server rejected the resume request because the
            partial file is already as large as the asset, in which case nothing
            was written and the file has to be fetched from the start.

    Raises:
        IncompleteDownloadError: If the response ends before the whole asset has
            arrived.
    """
    downloaded = destination.stat().st_size if destination.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

    with client.stream("GET", url, headers=headers) as response:
        if response.status_code == httpx.codes.REQUESTED_RANGE_NOT_SATISFIABLE:
            return False

        response.raise_for_status()
        resuming = response.status_code == httpx.codes.PARTIAL_CONTENT

        if downloaded and not resuming:
            logger.warning("Resume was refused, downloading {} from the start", url)
            downloaded = 0

        expected = _announced_size(response, already_on_disk=downloaded)

        with destination.open("ab" if resuming else "wb") as handle:
            next_milestone = downloaded + _LOG_EVERY_BYTES
            for chunk in response.iter_bytes(chunk_size):
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_milestone:
                    logger.debug("{:.0f} MB written", downloaded / BYTES_PER_MB)
                    next_milestone += _LOG_EVERY_BYTES

    if expected is not None and downloaded != expected:
        raise IncompleteDownloadError(url, expected, downloaded)

    return True


def _announced_size(response: httpx.Response, already_on_disk: int) -> int | None:
    """Total size of the asset according to the response headers.

    Returns None when the size cannot be trusted: either the server does not
    announce one, or the body is encoded and the header describes the encoded
    length rather than the bytes we write to disk.
    """
    length = response.headers.get("Content-Length")
    if length is None or response.headers.get("Content-Encoding"):
        return None
    return already_on_disk + int(length)
