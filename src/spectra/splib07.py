"""Access to the USGS Spectral Library Version 7.

The library is published as one archive rather than as individual spectra, so
it is downloaded whole and read from disk afterwards. Spectra are read straight
out of the zip, which holds 178,818 members and needs none of them unpacked.

Each spectrum is an ASCII file: one header line, then one value per line. The
values carry no wavelengths of their own — those live in a separate file per
instrument, in the same order — and a deleted point is written as -1.23e34.

Spectra are filed under chapters: minerals, vegetation, soils, coatings and
more. All of them are reachable here. Minerals are what gets mapped, but a
pixel of dry grass has to be recognised as dry grass rather than handed the
name of whichever mineral it least resembles, and that means holding its
spectrum too.
"""

import re
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
from loguru import logger
from numpy.typing import NDArray

from ..catalog.download import download_asset
from ..catalog.models import Asset
from ..core.exceptions import (
    ArchiveSizeError,
    SpectrumLengthError,
    SpectrumNotFoundError,
    UnknownInstrumentError,
)

ROOT = "ASCIIdata/ASCIIdata_splib07b"
DELETED = -1.0
MICRONS_TO_NM = 1000.0

WAVELENGTH_FILES = {
    "ASD": f"{ROOT}/splib07b_Wavelengths_ASDFR_0.35-2.5microns_2151ch.txt",
    "BECK": f"{ROOT}/splib07b_Wavelengths_BECK_Beckman_interp._3961_ch.txt",
    "AVIRIS": f"{ROOT}/splib07b_Wavelengths_AVIRIS_1996_interp_to_2203ch.txt",
}
BANDPASS_FILES = {
    "ASDFR": f"{ROOT}/splib07b_Bandpass_(FWHM)_ASDFR_StandardResolution.txt",
    "ASDHR": f"{ROOT}/splib07b_Bandpass_(FWHM)_ASDHR_High-Resolution.txt",
    "ASDNG": f"{ROOT}/splib07b_Bandpass_(FWHM)_ASDNG_High-Res_NextGen.txt",
    "BECK": f"{ROOT}/splib07b_Bandpass_(FWHM)_BECK_Beckman_in_microns.txt",
    "AVIRIS": f"{ROOT}/splib07b_Bandpass_(FWHM)_AVIRIS_1996_in_microns.txt",
}
# Every ASD variant shares one wavelength grid and differs only in bandpass.
WAVELENGTH_GRIDS = {"ASDFR": "ASD", "ASDHR": "ASD", "ASDNG": "ASD"}


@dataclass(frozen=True)
class Spectrum:
    """One laboratory spectrum, on the grid of the instrument that recorded it.

    Attributes:
        name: File stem in the archive, without the `splib07b_` prefix.
        instrument: Code of the instrument, e.g. `ASDNG`.
        wavelengths: Wavelength of each channel, in nanometres.
        reflectance: Reflectance of each channel; deleted points are NaN.
        widths: Full width at half maximum of each channel, in nanometres.
    """

    name: str
    instrument: str
    wavelengths: NDArray[np.float64]
    reflectance: NDArray[np.float64]
    widths: NDArray[np.float64]

    def __len__(self) -> int:
        """Number of channels."""
        return int(self.wavelengths.size)


def read_spectrum(archive: Path, name: str) -> Spectrum:
    """Read one mineral spectrum out of the archive.

    Reading several is much cheaper through `read_spectra`, which opens the
    archive once rather than once per spectrum.

    Args:
        archive: Path to `usgs_splib07.zip`.
        name: File stem without the `splib07b_` prefix or the extension, e.g.
            `Calcite_WS272_ASDNGa_AREF`.

    Returns:
        Spectrum: The spectrum, its wavelengths and its channel widths.

    Raises:
        SpectrumNotFoundError: If the archive holds no such spectrum.
        UnknownInstrumentError: If its instrument has no wavelength grid here.
        SpectrumLengthError: If it disagrees in length with that grid.
    """
    return read_spectra(archive, [name])[0]


def read_spectra(archive: Path, names: Iterable[str]) -> list[Spectrum]:
    """Read several mineral spectra out of the archive in one pass.

    The archive holds 178,818 members, so its index takes over a second to
    read. Opening it once for the whole batch, and reading each instrument's
    wavelength grid only the first time it is needed, turns a second per
    spectrum into a second overall.

    Args:
        archive: Path to `usgs_splib07.zip`.
        names: File stems, as for `read_spectrum`.

    Returns:
        list[Spectrum]: One spectrum per name, in the order given.

    Raises:
        SpectrumNotFoundError: If the archive holds no such spectrum.
        UnknownInstrumentError: If an instrument has no wavelength grid here.
        SpectrumLengthError: If a spectrum disagrees in length with its grid.
    """
    grids: dict[str, tuple[NDArray[np.float64], NDArray[np.float64]]] = {}
    spectra = []

    with zipfile.ZipFile(archive) as library:
        members = _members(library)
        for name in names:
            member = members.get(name)
            if member is None:
                raise SpectrumNotFoundError(name, archive)
            reflectance = _column(library, member)

            instrument = _instrument(name)
            if instrument not in grids:
                grids[instrument] = _grid(library, name, instrument)
            wavelengths, widths = grids[instrument]

            if reflectance.size != wavelengths.size:
                raise SpectrumLengthError(name, reflectance.size, wavelengths.size)

            spectra.append(
                Spectrum(
                    name=name,
                    instrument=instrument,
                    wavelengths=wavelengths,
                    reflectance=reflectance,
                    widths=widths,
                )
            )

    return spectra


def _members(library: zipfile.ZipFile) -> dict[str, str]:
    """Where every spectrum lives in the archive, by name.

    Built once per archive rather than guessed at, because a spectrum's chapter
    is not derivable from its name, and the file prefix is not fixed either:
    some chapters of the oversampled release carry names from the measured one.
    """
    found = {}
    for member in library.namelist():
        if not member.startswith(f"{ROOT}/") or not member.endswith(".txt"):
            continue
        stem = member.rsplit("/", 1)[-1][: -len(".txt")]
        _, _, name = stem.partition("_")
        if name:
            found[name] = member
    return found


def _grid(
    library: zipfile.ZipFile,
    name: str,
    instrument: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wavelengths and channel widths of the instrument a spectrum was taken on.

    Raises:
        UnknownInstrumentError: If the instrument has no wavelength grid here.
    """
    grid = WAVELENGTH_GRIDS.get(instrument, instrument)
    if grid not in WAVELENGTH_FILES or instrument not in BANDPASS_FILES:
        raise UnknownInstrumentError(name, instrument, BANDPASS_FILES)

    return (
        _column(library, WAVELENGTH_FILES[grid]) * MICRONS_TO_NM,
        _column(library, BANDPASS_FILES[instrument]) * MICRONS_TO_NM,
    )


def _instrument(name: str) -> str:
    """Instrument a spectrum was recorded on, without its quality grade.

    The grade is a run of trailing letters on the instrument code, so
    `ASDNGb` and `NIC4bb` become `ASDNG` and `NIC4`.
    """
    parts = name.split("_")
    code = parts[-2] if len(parts) >= 2 else name
    return re.sub(r"[abc]+$", "", code)


def _column(library: zipfile.ZipFile, member: str) -> NDArray[np.float64]:
    """Read a one-value-per-line ASCII file, dropping its header line.

    Deleted points, which the library writes as -1.23e34, become NaN so that
    they cannot pass unnoticed into an average.
    """
    text = library.read(member).decode("utf-8", errors="replace")
    values = np.array([float(line) for line in text.splitlines()[1:]])
    return np.where(values < DELETED, np.nan, values)


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
