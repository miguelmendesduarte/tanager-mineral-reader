"""Downloading of STAC assets.

The hyperspectral cubes are hundreds of megabytes each, so downloads stream to
disk in chunks and resume from a partial file rather than starting over.
"""

import time
from collections.abc import Iterable
from pathlib import Path

import httpx
from loguru import logger

from ..core.exceptions import DownloadError, IncompleteDownloadError
from .models import Asset, Item

PARTIAL_SUFFIX = ".part"
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * BYTES_PER_KB
BYTES_PER_GB = 1024 * BYTES_PER_MB
_LOG_EVERY_BYTES = 100 * BYTES_PER_MB
_SECONDS_PER_MINUTE = 60


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
    filename: str | None = None,
    expected_size: int | None = None,
) -> Path:
    """Download a single asset, resuming a previous attempt when possible.

    Args:
        asset: Asset to download.
        destination_dir: Directory the file is written to; created if missing.
        client: HTTP client used for the request.
        chunk_size: Number of bytes held in memory at a time.
        overwrite: Download again even if the file is already present.
        filename: Name to save the file under. Defaults to the name the catalog
            publishes, which only works for hrefs that carry it in their path.
        expected_size: Size of the asset in bytes, for servers that announce
            none of their own. Only consulted when the response is silent.

    Returns:
        Path: Location of the downloaded file.

    Raises:
        DownloadError: If the asset cannot be retrieved, or if the response ends
            before the whole file has arrived.
    """
    destination_dir.mkdir(parents=True, exist_ok=True)
    name = filename or asset.filename
    destination = destination_dir / name

    if destination.exists() and not overwrite:
        logger.info("Skipping {}, already downloaded", name)
        return destination

    partial = destination.with_name(destination.name + PARTIAL_SUFFIX)
    if overwrite:
        partial.unlink(missing_ok=True)

    try:
        _stream_to_file(
            asset.href,
            partial,
            client=client,
            chunk_size=chunk_size,
            expected_size=expected_size,
        )
    except httpx.HTTPError as error:
        raise DownloadError(asset.href) from error

    partial.replace(destination)
    size = _human_readable(destination.stat().st_size)
    logger.info("Downloaded {} ({})", name, size)
    return destination


def _human_readable(size_in_bytes: int) -> str:
    """Render a byte count in the largest unit that keeps it above one."""
    if size_in_bytes >= BYTES_PER_GB:
        return f"{size_in_bytes / BYTES_PER_GB:.1f} GB"
    if size_in_bytes >= BYTES_PER_MB:
        return f"{size_in_bytes / BYTES_PER_MB:.1f} MB"
    if size_in_bytes >= BYTES_PER_KB:
        return f"{size_in_bytes / BYTES_PER_KB:.1f} KB"
    return f"{size_in_bytes} B"


def _remaining_time(seconds: float) -> str:
    """Render a wait in minutes and seconds."""
    minutes, whole_seconds = divmod(int(seconds), _SECONDS_PER_MINUTE)
    return f"{minutes}m{whole_seconds:02d}s" if minutes else f"{whole_seconds}s"


def _progress(downloaded: int, expected: int | None, rate: float) -> str:
    """Describe how far a download has got, and how long is left.

    Args:
        downloaded: Bytes written so far, including any resumed from disk.
        expected: Total size of the asset, when the server announces one.
        rate: Bytes per second over this attempt.

    Returns:
        str: A line for the log. Without an announced size there is nothing to
            count towards, so it reports only what has arrived.
    """
    written = _human_readable(downloaded)
    speed = f"{rate / BYTES_PER_MB:.1f} MB/s"

    if expected is None:
        return f"{written} written, {speed}"

    percent = 100 * downloaded / expected
    left = _remaining_time((expected - downloaded) / rate) if rate else "unknown"
    return f"{written} of {_human_readable(expected)} ({percent:.0f}%), {speed}, {left} left"


def _stream_to_file(
    url: str,
    destination: Path,
    *,
    client: httpx.Client,
    chunk_size: int,
    expected_size: int | None = None,
) -> None:
    """Stream a URL into a file, resuming from it when it already holds bytes.

    Raises:
        IncompleteDownloadError: If the response ends early. The partial file is
            left in place so that the next attempt can resume from it.
    """
    if _fetch_into(
        url,
        destination,
        client=client,
        chunk_size=chunk_size,
        expected_size=expected_size,
    ):
        return

    logger.warning("The partial file for {} is unusable, downloading it again", url)
    destination.unlink(missing_ok=True)
    _fetch_into(
        url,
        destination,
        client=client,
        chunk_size=chunk_size,
        expected_size=expected_size,
    )


def _fetch_into(
    url: str,
    destination: Path,
    *,
    client: httpx.Client,
    chunk_size: int,
    expected_size: int | None = None,
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

        announced = _announced_size(response, already_on_disk=downloaded)
        expected = expected_size if announced is None else announced

        with destination.open("ab" if resuming else "wb") as handle:
            started_at = time.monotonic()
            started_from = downloaded
            next_milestone = downloaded + _LOG_EVERY_BYTES
            for chunk in response.iter_bytes(chunk_size):
                handle.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_milestone:
                    elapsed = time.monotonic() - started_at
                    rate = (downloaded - started_from) / elapsed if elapsed else 0.0
                    logger.info(_progress(downloaded, expected, rate))
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
