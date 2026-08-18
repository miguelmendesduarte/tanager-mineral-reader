"""Exception hierarchy for the project.

Every error raised by our own code inherits from `TanagerError`, so callers can
catch the whole family without reaching for bare `Exception`.
"""

from collections.abc import Iterable


class TanagerError(Exception):
    """Base class for every error raised by this project."""


class CatalogError(TanagerError):
    """Base class for failures while reading from the STAC catalog."""


class ItemFetchError(CatalogError):
    """Raised when a STAC item cannot be retrieved."""

    def __init__(self, url: str) -> None:
        super().__init__(f"Could not fetch the STAC item at {url}.")


class AssetNotFoundError(CatalogError):
    """Raised when an item does not expose the requested asset."""

    def __init__(self, name: str, available: Iterable[str]) -> None:
        options = ", ".join(sorted(available)) or "none"
        super().__init__(f"Item has no asset named {name!r}. Available: {options}.")


class DownloadError(CatalogError):
    """Raised when an asset cannot be downloaded."""

    def __init__(self, url: str, detail: str = "") -> None:
        super().__init__(f"Could not download the asset at {url}. {detail}".strip())


class IncompleteDownloadError(DownloadError):
    """Raised when the response ends before the whole asset has arrived."""

    def __init__(self, url: str, expected: int, received: int) -> None:
        super().__init__(url, f"Received {received} of {expected} bytes.")
