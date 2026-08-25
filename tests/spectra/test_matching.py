"""Tests for deciding which mineral a pixel looks most like."""

import numpy as np
import pytest

from src.core.exceptions import TooFewReferencesError
from src.spectra.matching import match

WAVELENGTHS = np.linspace(2080.0, 2490.0, 83)


def _dip(centre: float, depth: float, width: float = 25.0) -> np.ndarray:
    """A spectrum with one absorption in it, on a flat continuum."""
    return 1.0 - depth * np.exp(-0.5 * ((WAVELENGTHS - centre) / width) ** 2)


def _references() -> np.ndarray:
    """Three references, absorbing at 2170, 2210 and 2340 nm."""
    return np.stack([_dip(2170.0, 0.5), _dip(2210.0, 0.4), _dip(2340.0, 0.3)])


def test_a_pixel_is_matched_to_the_reference_it_copies() -> None:
    pixels = np.stack([_dip(2170.0, 0.5), _dip(2210.0, 0.4), _dip(2340.0, 0.3)])

    result = match(pixels, _references(), WAVELENGTHS)

    assert result.best.tolist() == [0, 1, 2]
    assert result.best_angle == pytest.approx(0.0, abs=1e-6)


def test_brightness_and_abundance_do_not_change_the_match() -> None:
    """A third of a pixel of alunite is still alunite, only shallower.

    Not quite exactly: a shallower absorption moves where the continuum hull
    touches the spectrum, so dilution is very slightly non-proportional after
    the continuum is removed. The residual angle is a thousandth of what
    separates the two most alike minerals, so it decides nothing.
    """
    pure = _dip(2170.0, 0.5)
    diluted = 1.0 - (1.0 - pure) / 3.0

    result = match(np.stack([0.6 * pure, 0.15 * diluted]), _references(), WAVELENGTHS)

    assert result.best.tolist() == [0, 0]
    assert result.best_angle[0] == pytest.approx(0.0, abs=1e-6)
    assert result.best_angle[1] < 0.01


def test_depth_reports_how_much_there_was_to_decide_on() -> None:
    result = match(
        np.stack([_dip(2170.0, 0.5), _dip(2170.0, 0.05)]),
        _references(),
        WAVELENGTHS,
    )

    assert result.depth[0] == pytest.approx(0.5, abs=0.01)
    assert result.depth[1] == pytest.approx(0.05, abs=0.01)


def test_the_runner_up_shows_how_close_the_call_was() -> None:
    """A pixel between two references should not look like a clear winner."""
    clear = match(np.stack([_dip(2340.0, 0.3)]), _references(), WAVELENGTHS)
    ambiguous = match(np.stack([_dip(2190.0, 0.45)]), _references(), WAVELENGTHS)

    assert clear.margin[0] > ambiguous.margin[0]


def test_a_flat_spectrum_has_no_direction_to_compare() -> None:
    """Nothing absorbs, so there is no shape, and the angle is undefined."""
    result = match(
        np.stack([np.full(WAVELENGTHS.size, 0.3)]), _references(), WAVELENGTHS
    )

    assert result.depth[0] == pytest.approx(0.0)
    assert np.isnan(result.best_angle[0])


def test_matching_needs_a_runner_up_to_be_possible() -> None:
    with pytest.raises(TooFewReferencesError):
        match(np.stack([_dip(2170.0, 0.5)]), _references()[:1], WAVELENGTHS)


def test_noise_does_not_move_a_deep_match() -> None:
    generator = np.random.default_rng(20260824)
    noisy = _dip(2210.0, 0.4) + generator.normal(0.0, 0.0026, WAVELENGTHS.size)

    result = match(np.stack([noisy]), _references(), WAVELENGTHS)

    assert result.best[0] == 1
