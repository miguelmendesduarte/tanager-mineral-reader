"""Exception hierarchy for the project.

Every error raised by our own code inherits from `TanagerError`, so callers can
catch the whole family without reaching for bare `Exception`.
"""

from collections.abc import Iterable
from pathlib import Path


class TanagerError(Exception):
    """Base class for every error raised by this project."""


class ConfigError(TanagerError, ValueError):
    """Base class for settings that cannot be used as given.

    Also a `ValueError` so that pydantic reports these as validation failures
    against the setting they came from, rather than as a crash.
    """


class NoMineralGroupsError(ConfigError):
    """Raised when no mineral group is configured to match against."""

    def __init__(self) -> None:
        super().__init__("At least one mineral group is needed.")


class EmptyMineralGroupError(ConfigError):
    """Raised when a mineral group names no reference spectra."""

    def __init__(self, group: str) -> None:
        super().__init__(f"Mineral group {group!r} names no species.")


class RepeatedSpeciesError(ConfigError):
    """Raised when a reference spectrum is filed under more than one group."""

    def __init__(self, names: Iterable[str]) -> None:
        listed = ", ".join(sorted(names))
        super().__init__(
            f"{listed} appears in more than one mineral group, so there is no "
            "single answer to what it is reported as."
        )


class EmptyMatchRangeError(ConfigError):
    """Raised when the wavelength range to match in ends before it starts."""

    def __init__(self, low: float, high: float) -> None:
        super().__init__(f"The match range {low} to {high} nm holds no wavelengths.")


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


class SpectraError(TanagerError):
    """Base class for failures while working with reference spectra."""


class ArchiveSizeError(SpectraError):
    """Raised when the catalog record does not give the size of an archive."""

    def __init__(self, url: str, name: str) -> None:
        super().__init__(f"The record at {url} carries no size for {name!r}.")


class SpectrumNotFoundError(SpectraError):
    """Raised when the archive holds no spectrum under that name."""

    def __init__(self, name: str, archive: Path) -> None:
        super().__init__(f"{archive} holds no spectrum named {name!r}.")


class UnknownInstrumentError(SpectraError):
    """Raised when a spectrum was recorded on an instrument we cannot place."""

    def __init__(self, name: str, code: str, known: Iterable[str]) -> None:
        options = ", ".join(sorted(known)) or "none"
        super().__init__(
            f"{name!r} was recorded on {code!r}, which has no wavelength grid "
            f"here. Known instruments: {options}."
        )


class LibraryTooCoarseError(SpectraError):
    """Raised when a library spectrum cannot be averaged onto a sensor's bands."""

    def __init__(self, name: str, spacing: float, width: float) -> None:
        super().__init__(
            f"{name!r} is sampled every {spacing:.1f} nm, wider than the "
            f"narrowest band it would be averaged onto ({width:.1f} nm). "
            "Averaging it would be an interpolation."
        )


class SpectrumLengthError(SpectraError):
    """Raised when a spectrum and its wavelength grid disagree in length."""

    def __init__(self, name: str, values: int, wavelengths: int) -> None:
        super().__init__(
            f"{name!r} holds {values} values against {wavelengths} wavelengths."
        )


class CubeError(TanagerError):
    """Base class for failures while reading a hyperspectral cube."""


class UnknownCubeError(CubeError):
    """Raised when a file holds no cube this project knows how to read."""

    def __init__(self, path: Path, available: Iterable[str]) -> None:
        options = ", ".join(sorted(available)) or "none"
        super().__init__(f"{path} holds no known cube. Found: {options}.")


class NoBandsInRangeError(CubeError):
    """Raised when no usable band falls inside the requested wavelengths."""

    def __init__(self, low: float, high: float) -> None:
        super().__init__(f"No usable band between {low} nm and {high} nm.")


class GridMetadataError(CubeError):
    """Raised when the grid metadata does not describe the scene corners."""

    def __init__(self, name: str) -> None:
        super().__init__(f"The grid metadata has no {name} entry.")


class LayerNotFoundError(CubeError):
    """Raised when a file does not carry the requested per pixel layer."""

    def __init__(self, name: str, available: Iterable[str]) -> None:
        options = ", ".join(sorted(available)) or "none"
        super().__init__(f"No layer named {name!r}. Available: {options}.")
