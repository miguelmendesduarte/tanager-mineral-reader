"""A tiny archive shaped like the real spectral library."""

import zipfile
from pathlib import Path

import pytest

from src.spectra.splib07 import BANDPASS_FILES, ROOT, WAVELENGTH_FILES

MINERALS = f"{ROOT}/ChapterM_Minerals"
VEGETATION = f"{ROOT}/ChapterV_Vegetation"

DELETED = "-1.2300000e+034"

# Four channels, 2.0 to 2.3 microns, so that a spectrum stays readable by eye.
ASD_WAVELENGTHS = [
    "2.0000000e+000",
    "2.1000000e+000",
    "2.2000000e+000",
    "2.3000000e+000",
]
ASD_BANDPASS = ["5.6000000e-003"] * 4
BECK_WAVELENGTHS = ["2.0000000e+000", "2.2000000e+000"]
BECK_BANDPASS = ["1.0000000e-002"] * 2

REFLECTANCE = ["8.0000000e-001", DELETED, "4.0000000e-001", "7.5000000e-001"]


def _member(name: str, values: list[str]) -> tuple[str, str]:
    """One archive member: a header line, then one value per line."""
    return name, "\n".join([f" splib07b Record=1: {name}", *values]) + "\n"


def write_archive(path: Path, spectra: dict[str, list[str]] | None = None) -> Path:
    """Write an archive holding the grids and whichever spectra are asked for.

    Args:
        path: Where to write the zip.
        spectra: File stems mapped to their values. Defaults to one ASD Next-Gen
            spectrum, `Calcite_TEST1_ASDNGa_AREF`.

    Returns:
        Path: The archive.
    """
    if spectra is None:
        spectra = {"Calcite_TEST1_ASDNGa_AREF": REFLECTANCE}

    members = [
        _member(WAVELENGTH_FILES["ASD"], ASD_WAVELENGTHS),
        _member(WAVELENGTH_FILES["BECK"], BECK_WAVELENGTHS),
        _member(BANDPASS_FILES["ASDNG"], ASD_BANDPASS),
        _member(BANDPASS_FILES["BECK"], BECK_BANDPASS),
    ]
    members += [
        _member(f"{MINERALS}/splib07b_{stem}.txt", values)
        for stem, values in spectra.items()
    ]

    with zipfile.ZipFile(path, "w") as archive:
        for name, text in members:
            archive.writestr(name, text)
    return path


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """Path to an archive holding a single ASD Next-Gen calcite spectrum."""
    return write_archive(tmp_path / "usgs_splib07.zip")
