"""Tests for the spectral axis."""

import numpy as np
import pytest

from src.core.exceptions import NoBandsInRangeError
from src.readers.wavelengths import Wavelengths


def _wavelengths(usable: list[bool] | None = None) -> Wavelengths:
    centres = np.array([400.0, 900.0, 1400.0, 1900.0, 2200.0, 2400.0])
    flags = (
        np.ones(centres.size, dtype=np.bool_) if usable is None else np.array(usable)
    )
    return Wavelengths(
        centres=centres,
        widths=np.full(centres.size, 5.0),
        usable=flags,
    )


def test_wavelengths_count_their_bands() -> None:
    assert len(_wavelengths()) == 6


def test_wavelengths_select_the_bands_inside_a_range() -> None:
    selected = _wavelengths().indices_between(2100, 2450)

    assert selected.tolist() == [4, 5]


def test_wavelengths_include_the_bounds_of_the_range() -> None:
    selected = _wavelengths().indices_between(400, 900)

    assert selected.tolist() == [0, 1]


def test_wavelengths_skip_the_bands_the_file_flags_as_unusable() -> None:
    usable = [True, True, False, False, True, True]

    selected = _wavelengths(usable).indices_between(1300, 2450)

    assert selected.tolist() == [4, 5]


def test_wavelengths_can_keep_the_unusable_bands_on_request() -> None:
    usable = [True, True, False, False, True, True]

    selected = _wavelengths(usable).indices_between(1300, 2450, usable_only=False)

    assert selected.tolist() == [2, 3, 4, 5]


def test_wavelengths_report_a_range_that_holds_no_band() -> None:
    with pytest.raises(NoBandsInRangeError, match="3000"):
        _wavelengths().indices_between(2600, 3000)


def test_wavelengths_report_a_range_where_every_band_is_unusable() -> None:
    usable = [True, True, False, False, True, True]

    with pytest.raises(NoBandsInRangeError):
        _wavelengths(usable).indices_between(1350, 1950)
