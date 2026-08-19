"""Tests for the cube reader."""

from pathlib import Path

import numpy as np
import pytest

from src.core.exceptions import (
    LayerNotFoundError,
    NoBandsInRangeError,
    UnknownCubeError,
)
from src.readers.cube import Cube

from .conftest import COLUMNS, EPSG, ROWS, write_cube


def test_cube_finds_the_reflectance_product(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        assert cube.name == "surface_reflectance"


def test_cube_finds_the_radiance_product(tmp_path: Path) -> None:
    path = write_cube(tmp_path / "radiance.h5", name="toa_radiance")

    with Cube(path) as cube:
        assert cube.name == "toa_radiance"


def test_cube_lists_what_it_found_when_the_product_is_unknown(tmp_path: Path) -> None:
    path = write_cube(tmp_path / "other.h5", name="brightness_temperature")

    with pytest.raises(UnknownCubeError, match="brightness_temperature"):
        Cube(path)


def test_cube_reports_its_shape(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        assert cube.shape == (6, ROWS, COLUMNS)


def test_cube_reads_its_spectral_axis(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        wavelengths = cube.wavelengths

    assert len(wavelengths) == 6
    assert wavelengths.centres[0] == pytest.approx(400.0)
    assert wavelengths.widths[0] == pytest.approx(5.0)


def test_cube_treats_every_band_as_usable_when_the_file_is_silent(
    cube_path: Path,
) -> None:
    with Cube(cube_path) as cube:
        assert cube.wavelengths.usable.all()


def test_cube_honours_the_usable_band_flags_of_the_file(tmp_path: Path) -> None:
    path = write_cube(tmp_path / "flagged.h5", usable=[1, 1, 0, 0, 1, 1])

    with Cube(path) as cube:
        assert cube.wavelengths.usable.tolist() == [
            True,
            True,
            False,
            False,
            True,
            True,
        ]


def test_cube_selects_bands_by_wavelength(tmp_path: Path) -> None:
    path = write_cube(tmp_path / "flagged.h5", usable=[1, 1, 0, 0, 1, 1])

    with Cube(path) as cube:
        assert cube.bands_between(1000, 2500).tolist() == [4, 5]


def test_cube_reports_a_wavelength_range_it_cannot_serve(cube_path: Path) -> None:
    with Cube(cube_path) as cube, pytest.raises(NoBandsInRangeError):
        cube.bands_between(2600, 3000)


def test_cube_places_the_scene_on_the_map(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        grid = cube.grid

    assert grid.epsg == EPSG
    assert grid.shape == (ROWS, COLUMNS)
    assert grid.pixel_size == (30.0, 30.0)


def test_cube_reads_whole_bands(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        values = cube.read_bands(np.array([2, 4]))

    assert values.shape == (2, ROWS, COLUMNS)
    assert values[0, 1, 3] == pytest.approx(213.0)
    assert values[1, 1, 3] == pytest.approx(413.0)


def test_cube_reads_bands_in_order_and_only_once(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        values = cube.read_bands(np.array([4, 2, 4]))

    assert values.shape == (2, ROWS, COLUMNS)
    assert values[0, 0, 1] == pytest.approx(201.0)


def test_cube_turns_fill_values_into_nan(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        values = cube.read_bands(np.array([0]))

    assert np.isnan(values[0, 0, 0])
    assert values[0, 0, 1] == pytest.approx(1.0)


def test_cube_reads_a_window(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        values = cube.read_window(slice(1, 3), slice(2, 4), bands=np.array([1]))

    assert values.shape == (1, 2, 2)
    assert values[0, 0, 0] == pytest.approx(112.0)


def test_cube_reads_every_band_of_a_window_by_default(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        values = cube.read_window(slice(0, 2), slice(0, 2))

    assert values.shape == (6, 2, 2)


def test_cube_reads_the_spectrum_of_one_pixel(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        spectrum = cube.read_spectrum(1, 2)

    assert spectrum.shape == (6,)
    assert spectrum.tolist() == [12.0, 112.0, 212.0, 312.0, 412.0, 512.0]


def test_cube_lists_the_layers_that_sit_beside_the_cube(tmp_path: Path) -> None:
    path = write_cube(
        tmp_path / "layered.h5",
        masks={
            "sun_zenith": np.full((ROWS, COLUMNS), 55.8),
            "nodata_pixels": np.zeros((ROWS, COLUMNS)),
        },
    )

    with Cube(path) as cube:
        assert cube.layers == ("nodata_pixels", "sun_zenith")


def test_cube_reads_a_layer(tmp_path: Path) -> None:
    geometry = np.arange(ROWS * COLUMNS).reshape(ROWS, COLUMNS)
    path = write_cube(tmp_path / "layered.h5", masks={"sun_zenith": geometry})

    with Cube(path) as cube:
        values = cube.read_layer("sun_zenith")

    assert values.shape == (ROWS, COLUMNS)
    assert values[1, 2] == pytest.approx(7.0)


def test_cube_turns_the_fill_values_of_a_layer_into_nan(tmp_path: Path) -> None:
    geometry = np.full((ROWS, COLUMNS), 55.8)
    geometry[0, 0] = -9999.0
    path = write_cube(
        tmp_path / "layered.h5",
        masks={"sun_zenith": geometry},
        layer_fill=-9999.0,
    )

    with Cube(path) as cube:
        values = cube.read_layer("sun_zenith")

    assert np.isnan(values[0, 0])
    assert values[1, 1] == pytest.approx(55.8)


def test_cube_lists_the_alternatives_when_a_layer_is_missing(tmp_path: Path) -> None:
    path = write_cube(
        tmp_path / "layered.h5",
        masks={"sun_zenith": np.zeros((ROWS, COLUMNS))},
    )

    with Cube(path) as cube, pytest.raises(LayerNotFoundError, match="sun_zenith"):
        cube.read_layer("column_water_vapour")


def test_cube_treats_every_pixel_as_valid_when_there_are_no_masks(
    cube_path: Path,
) -> None:
    with Cube(cube_path) as cube:
        assert cube.valid_mask().all()


def test_cube_excludes_pixels_outside_the_strip_and_under_cloud(
    tmp_path: Path,
) -> None:
    nodata = np.zeros((ROWS, COLUMNS))
    nodata[0, 0] = 1
    cloud = np.zeros((ROWS, COLUMNS))
    cloud[3, 4] = 1
    path = write_cube(
        tmp_path / "masked.h5",
        masks={"nodata_pixels": nodata, "beta_cloud_mask": cloud},
    )

    with Cube(path) as cube:
        valid = cube.valid_mask()

    assert valid.shape == (ROWS, COLUMNS)
    assert not valid[0, 0]
    assert not valid[3, 4]
    assert valid.sum() == ROWS * COLUMNS - 2


def test_cube_closes_its_file_on_exit(cube_path: Path) -> None:
    with Cube(cube_path) as cube:
        pass

    assert not cube._file
