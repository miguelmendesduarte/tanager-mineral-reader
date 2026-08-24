"""Access to the USGS Spectral Library Version 7.

The library is published as one archive rather than as individual spectra, so
it is downloaded whole and read from disk afterwards.
"""

from pathlib import Path

import httpx
from loguru import logger

from ..catalog.download import download_asset
from ..catalog.models import Asset
from ..core.exceptions import ArchiveSizeError


def fetch_archive(
    url: str,
    destination_dir: Path,
    *,
    archive_name: str,
    client: httpx.Client,
    chunk_size: int,
    expected_size: int | None = None,
    overwrite: bool = False,
) -> Path:
    """Download the spectral library archive.

    The archive is about 5.5 GB. It is kept rather than deleted after being
    read, so that a second run needs no network at all.

    Args:
        url: URL the archive is published at.
        destination_dir: Directory the archive is written to.
        archive_name: Name the archive is saved under.
        client: HTTP client used for the request.
        chunk_size: Number of bytes held in memory at a time.
        expected_size: Size of the archive in bytes, when it is known from the
            catalog record. The server announces no size of its own, so without
            this there is no progress to report and no way to notice a download
            that ended early.
        overwrite: Download again even if the archive is already present.

    Returns:
        Path: Location of the archive.

    Raises:
        DownloadError: If the archive cannot be retrieved.
    """
    return download_asset(
        Asset(href=url),
        destination_dir,
        client=client,
        chunk_size=chunk_size,
        overwrite=overwrite,
        filename=archive_name,
        expected_size=expected_size,
    )


def archive_size(url: str, name: str, *, client: httpx.Client) -> int:
    """Look up the size of the archive in the catalog record.

    Args:
        url: URL of the catalog record, as JSON.
        name: Name of the archive within the record.
        client: HTTP client used for the request.

    Returns:
        int: Size of the archive in bytes.

    Raises:
        ArchiveSizeError: If the record cannot be read, or carries no size for
            an archive of that name.
    """
    try:
        response = client.get(url)
        response.raise_for_status()
        files = response.json()["files"]
    except (httpx.HTTPError, KeyError, ValueError) as error:
        raise ArchiveSizeError(url, name) from error

    for file in files:
        size = file.get("size")
        if file.get("name") == name and size is not None:
            logger.info("The catalog record reports {} is {} bytes", name, size)
            return int(size)

    raise ArchiveSizeError(url, name)
