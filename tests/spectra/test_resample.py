"""Tests for averaging library spectra onto a sensor's bands."""

import numpy as np
import pytest
from numpy.typing import NDArray

from src.core.exceptions import LibraryTooCoarseError
from src.readers.wavelengths import Wavelengths
from src.spectra.resample import FWHM_TO_SIGMA, blunter_bands, convolve
from src.spectra.splib07 import Spectrum


def _spectrum(
    wavelengths: NDArray[np.float64],
    reflectance: NDArray[np.float64],
    width: float = 1.0,
) -> Spectrum:
    """A library spectrum on an arbitrary grid."""
    return Spectrum(
        name="Test_SAMPLE_ASDNGa_AREF",
        instrument="ASDNG",
        wavelengths=wavelengths,
        reflectance=reflectance,
        widths=np.full(wavelengths.size, width),
    )


def _bands(centres: list[float], width: float = 6.0) -> Wavelengths:
    """Sensor bands of a single width."""
    array = np.asarray(centres, dtype=np.float64)
    return Wavelengths(
        centres=array,
        widths=np.full(array.size, width),
        usable=np.ones(array.size, dtype=np.bool_),
    )


def test_a_flat_spectrum_averages_to_the_same_value() -> None:
    grid = np.arange(2000.0, 2400.0, 0.5)
    spectrum = _spectrum(grid, np.full(grid.size, 0.42))

    values = convolve(spectrum, _bands([2100.0, 2200.0, 2300.0]))

    assert values == pytest.approx(0.42)


def test_a_gaussian_absorption_shallows_by_the_expected_factor() -> None:
    """Two Gaussians convolve to a Gaussian of the summed variances.

    A dip of depth d and width sigma_l, averaged by a band response of width
    sigma_s, keeps its position and comes back with depth
    d * sigma_l / sqrt(sigma_l^2 + sigma_s^2).
    """
    centre, depth, library_sigma = 2200.0, 0.5, 20.0
    band_fwhm = 30.0
    band_sigma = band_fwhm / FWHM_TO_SIGMA

    grid = np.arange(2000.0, 2400.0, 0.1)
    dip = depth * np.exp(-0.5 * ((grid - centre) / library_sigma) ** 2)
    spectrum = _spectrum(grid, 1.0 - dip)

    values = convolve(spectrum, _bands([centre], width=band_fwhm))

    expected = 1.0 - depth * library_sigma / np.hypot(library_sigma, band_sigma)
    assert values[0] == pytest.approx(expected, abs=1e-3)


def test_a_gaussian_absorption_keeps_its_position() -> None:
    centre, library_sigma = 2200.0, 20.0
    grid = np.arange(2000.0, 2400.0, 0.5)
    dip = 0.5 * np.exp(-0.5 * ((grid - centre) / library_sigma) ** 2)
    spectrum = _spectrum(grid, 1.0 - dip)

    bands = _bands([2160.0, 2180.0, 2200.0, 2220.0, 2240.0])
    values = convolve(spectrum, bands)

    assert bands.centres[int(np.argmin(values))] == centre


def test_bands_outside_the_library_are_not_invented() -> None:
    grid = np.arange(2100.0, 2300.0, 0.5)
    spectrum = _spectrum(grid, np.full(grid.size, 0.3))

    values = convolve(spectrum, _bands([1500.0, 2200.0, 2450.0]))

    assert np.isnan(values[0])
    assert values[1] == pytest.approx(0.3)
    assert np.isnan(values[2])


def test_deleted_points_are_left_out_of_the_average() -> None:
    grid = np.arange(2000.0, 2400.0, 0.5)
    reflectance = np.full(grid.size, 0.6)
    reflectance[np.abs(grid - 2200.0) < 1.0] = np.nan

    values = convolve(_spectrum(grid, reflectance), _bands([2200.0]))

    assert values[0] == pytest.approx(0.6)


def test_a_thinly_sampled_library_is_refused() -> None:
    """Averaging needs several library channels per band, not one."""
    grid = np.arange(2000.0, 2400.0, 10.0)
    spectrum = _spectrum(grid, np.full(grid.size, 0.5), width=10.0)

    with pytest.raises(LibraryTooCoarseError):
        convolve(spectrum, _bands([2200.0], width=6.0))


def test_blunter_bands_counts_only_inside_the_range_that_matters() -> None:
    """The library is blunt in the visible and sharp in the shortwave here."""
    grid = np.arange(400.0, 2400.0, 1.0)
    widths = np.where(grid < 1000.0, 9.0, 5.0)
    spectrum = Spectrum(
        name="Test_SAMPLE_ASDNGa_AREF",
        instrument="ASDNG",
        wavelengths=grid,
        reflectance=np.full(grid.size, 0.5),
        widths=widths,
    )
    bands = _bands([500.0, 600.0, 2200.0, 2300.0], width=6.0)

    assert blunter_bands(spectrum, bands, 2100.0, 2400.0) == (0, 2)
    assert blunter_bands(spectrum, bands, 400.0, 2400.0) == (2, 4)
