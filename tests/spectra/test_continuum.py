"""Tests for taking the envelope off a spectrum."""

import numpy as np
import pytest

from src.spectra.continuum import deepest_feature, remove


def test_a_straight_line_is_entirely_its_own_envelope() -> None:
    """Any sloped, featureless spectrum removes to a flat 1.0."""
    wavelengths = np.linspace(2000.0, 2400.0, 50)
    values = 0.2 + 0.0005 * (wavelengths - 2000.0)

    assert remove(wavelengths, values) == pytest.approx(1.0)


def test_brightness_divides_out() -> None:
    """The same mineral in sun and in shade removes to the same shape."""
    wavelengths = np.linspace(2000.0, 2400.0, 200)
    shape = 1.0 - 0.4 * np.exp(-0.5 * ((wavelengths - 2200.0) / 30.0) ** 2)

    bright = remove(wavelengths, 0.6 * shape)
    dim = remove(wavelengths, 0.15 * shape)

    assert bright == pytest.approx(dim)


def test_a_known_dip_comes_back_at_its_own_depth() -> None:
    wavelengths = np.linspace(2000.0, 2400.0, 401)
    values = 0.5 * (1.0 - 0.3 * np.exp(-0.5 * ((wavelengths - 2200.0) / 25.0) ** 2))

    position, depth = deepest_feature(wavelengths, values, 2000.0, 2400.0)

    assert position == pytest.approx(2200.0, abs=1.0)
    assert depth == pytest.approx(0.3, abs=0.01)


def test_a_roll_off_at_the_edge_is_not_read_as_an_absorption() -> None:
    """The hull follows a decline down rather than bridging over it.

    Detectors fall away steeply at the ends of their range. A continuum drawn
    between fixed endpoints would report that fall as the deepest feature in
    the spectrum; the hull lies along it, so only the real dip survives.
    """
    wavelengths = np.linspace(2000.0, 2500.0, 501)
    dip = 0.3 * np.exp(-0.5 * ((wavelengths - 2200.0) / 25.0) ** 2)
    cliff = 0.8 * np.clip((wavelengths - 2450.0) / 50.0, 0.0, None)
    values = 0.5 * (1.0 - dip - cliff)

    position, depth = deepest_feature(wavelengths, values, 2000.0, 2500.0)

    assert position == pytest.approx(2200.0, abs=1.0)
    assert depth == pytest.approx(0.3, abs=0.02)


def test_two_dips_report_the_deeper_one() -> None:
    wavelengths = np.linspace(2000.0, 2400.0, 401)
    shallow = 0.1 * np.exp(-0.5 * ((wavelengths - 2160.0) / 20.0) ** 2)
    deep = 0.35 * np.exp(-0.5 * ((wavelengths - 2260.0) / 20.0) ** 2)

    position, depth = deepest_feature(wavelengths, 1.0 - shallow - deep, 2000.0, 2400.0)

    assert position == pytest.approx(2260.0, abs=2.0)
    assert depth == pytest.approx(0.35, abs=0.02)


def test_missing_points_take_no_part() -> None:
    wavelengths = np.linspace(2000.0, 2400.0, 50)
    values = np.full(wavelengths.size, 0.4)
    values[10] = np.nan

    removed = remove(wavelengths, values)

    assert np.isnan(removed[10])
    assert removed[np.isfinite(removed)] == pytest.approx(1.0)


def test_a_range_holding_nothing_gives_nothing() -> None:
    wavelengths = np.linspace(2000.0, 2400.0, 50)
    values = np.full(wavelengths.size, 0.4)

    position, depth = deepest_feature(wavelengths, values, 3000.0, 3100.0)

    assert np.isnan(position)
    assert np.isnan(depth)
