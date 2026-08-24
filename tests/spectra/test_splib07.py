"""Tests for access to the USGS spectral library."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest

from src.core.exceptions import (
    ArchiveSizeError,
    SpectrumLengthError,
    SpectrumNotFoundError,
    UnknownInstrumentError,
)
from src.spectra.splib07 import archive_size, read_spectrum
from tests.spectra.conftest import write_archive

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


def test_read_spectrum_returns_wavelengths_in_nanometres(archive: Path) -> None:
    spectrum = read_spectrum(archive, "Calcite_TEST1_ASDNGa_AREF")

    assert spectrum.wavelengths.tolist() == [2000.0, 2100.0, 2200.0, 2300.0]


def test_read_spectrum_turns_deleted_points_into_nan(archive: Path) -> None:
    spectrum = read_spectrum(archive, "Calcite_TEST1_ASDNGa_AREF")

    assert np.isnan(spectrum.reflectance[1])
    assert spectrum.reflectance[0] == pytest.approx(0.8)
    assert spectrum.reflectance[3] == pytest.approx(0.75)


def test_read_spectrum_reads_the_channel_widths_of_its_instrument(
    archive: Path,
) -> None:
    spectrum = read_spectrum(archive, "Calcite_TEST1_ASDNGa_AREF")

    assert spectrum.instrument == "ASDNG"
    assert spectrum.widths.tolist() == [5.6, 5.6, 5.6, 5.6]


def test_read_spectrum_pairs_each_instrument_with_its_own_grid(
    tmp_path: Path,
) -> None:
    """Beckman spectra are longer and blunter; they must not get the ASD grid."""
    path = write_archive(
        tmp_path / "library.zip",
        {"Alunite_TEST2_BECKb_AREF": ["6.0000000e-001", "3.0000000e-001"]},
    )

    spectrum = read_spectrum(path, "Alunite_TEST2_BECKb_AREF")

    assert spectrum.instrument == "BECK"
    assert spectrum.wavelengths.tolist() == [2000.0, 2200.0]
    assert spectrum.widths.tolist() == [10.0, 10.0]


def test_read_spectrum_rejects_a_name_that_is_not_in_the_archive(
    archive: Path,
) -> None:
    with pytest.raises(SpectrumNotFoundError):
        read_spectrum(archive, "Alunite_NOPE_ASDNGa_AREF")


def test_read_spectrum_rejects_an_instrument_with_no_grid(tmp_path: Path) -> None:
    path = write_archive(
        tmp_path / "library.zip",
        {"Calcite_TEST1_NIC4bb_RREF": ["1.0000000e-001"]},
    )

    with pytest.raises(UnknownInstrumentError):
        read_spectrum(path, "Calcite_TEST1_NIC4bb_RREF")


def test_read_spectrum_rejects_a_spectrum_that_does_not_fit_its_grid(
    tmp_path: Path,
) -> None:
    path = write_archive(
        tmp_path / "library.zip",
        {"Calcite_TEST1_ASDNGa_AREF": ["1.0000000e-001", "2.0000000e-001"]},
    )

    with pytest.raises(SpectrumLengthError):
        read_spectrum(path, "Calcite_TEST1_ASDNGa_AREF")
