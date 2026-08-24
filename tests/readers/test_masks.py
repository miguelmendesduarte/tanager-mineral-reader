"""Tests for deciding which pixels could be bare rock."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.core.exceptions import NoBandsInRangeError
from src.readers.cube import Cube
from src.readers.masks import greenness, rock

from .conftest import COLUMNS, ROWS, write_spectral_cube

RED = (660.0, 680.0)
NEAR_INFRARED = (850.0, 870.0)
SHORTWAVE = (2190.0, 2210.0)


def _cube(tmp_path: Path, **kwargs: Any) -> Cube:
    """A cube whose pixels can be given their own spectra."""
    return Cube(write_spectral_cube(tmp_path / "scene.h5", **kwargs))


def test_greenness_separates_leaves_from_rock(tmp_path: Path) -> None:
    """Leaves reflect infrared far more than red; rock reflects both alike."""
    with _cube(tmp_path, red=[0.05, 0.25], near_infrared=[0.45, 0.27]) as cube:
        values = greenness(cube, RED, NEAR_INFRARED)

    assert values[0, 0] == pytest.approx(0.8)
    assert values[0, 1] == pytest.approx(0.038, abs=0.01)


def test_greenness_is_undefined_where_both_bands_are_zero(tmp_path: Path) -> None:
    with _cube(tmp_path, red=[0.0, 0.2], near_infrared=[0.0, 0.4]) as cube:
        values = greenness(cube, RED, NEAR_INFRARED)

    assert np.isnan(values[0, 0])
    assert np.isfinite(values[0, 1])


def test_rock_drops_the_green_pixels(tmp_path: Path) -> None:
    with _cube(tmp_path, red=[0.05, 0.25], near_infrared=[0.45, 0.27]) as cube:
        mask = rock(
            cube,
            vegetation_ndvi=0.2,
            dark_reflectance=0.05,
            red=RED,
            near_infrared=NEAR_INFRARED,
            shortwave=SHORTWAVE,
        )

    assert not mask[0, 0]
    assert mask[0, 1]


def test_rock_drops_the_dark_pixels(tmp_path: Path) -> None:
    """Shadow and water reflect too little for an absorption to be measurable."""
    with _cube(tmp_path, shortwave=[0.01, 0.30]) as cube:
        mask = rock(
            cube,
            vegetation_ndvi=0.2,
            dark_reflectance=0.05,
            red=RED,
            near_infrared=NEAR_INFRARED,
            shortwave=SHORTWAVE,
        )

    assert not mask[0, 0]
    assert mask[0, 1]


def test_rock_keeps_nothing_the_file_flags_as_unusable(tmp_path: Path) -> None:
    clouded = np.zeros((ROWS, COLUMNS), dtype=np.float32)
    clouded[0, 1] = 1.0

    with _cube(tmp_path, masks={"beta_cloud_mask": clouded}) as cube:
        mask = rock(
            cube,
            vegetation_ndvi=0.2,
            dark_reflectance=0.05,
            red=RED,
            near_infrared=NEAR_INFRARED,
            shortwave=SHORTWAVE,
        )

    assert not mask[0, 1]
    assert mask[0, 0]


def test_rock_needs_bands_in_every_range(tmp_path: Path) -> None:
    with _cube(tmp_path) as cube, pytest.raises(NoBandsInRangeError):
        rock(
            cube,
            vegetation_ndvi=0.2,
            dark_reflectance=0.05,
            red=RED,
            near_infrared=NEAR_INFRARED,
            shortwave=(3000.0, 3100.0),
        )
