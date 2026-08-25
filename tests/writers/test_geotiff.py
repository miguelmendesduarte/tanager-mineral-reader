"""Tests for writing a mineral map as a GeoTIFF."""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from src.readers.grid import Grid
from src.spectra.mapping import UNCLASSIFIED, Mapped
from src.spectra.rejection import NoiseFloor, Resolution
from src.writers.geotiff import BANDS, write_map

GROUPS = ("alunite", "kaolinite_group")
EPSG = 32611
UPPER_LEFT = (475290.0, 4167150.0)
PIXEL = 30.0


def _mapped(labels: list[list[int]]) -> Mapped:
    """A two by two map, placed on a real grid."""
    grid = np.asarray(labels, dtype=np.intp)
    rows, columns = grid.shape
    return Mapped(
        labels=grid,
        groups=GROUPS,
        angle=np.full(grid.shape, 21.5),
        depth=np.full(grid.shape, 0.12),
        margin=np.full(grid.shape, 6.0),
        second=np.full(grid.shape, UNCLASSIFIED),
        resolved=np.ones(grid.shape, dtype=np.bool_),
        floor=NoiseFloor(
            depth=0.0625, angle=38.8, noise=0.00224, brightness=0.21, quantile=0.99
        ),
        resolution=Resolution(
            depths=np.array([0.05, 0.3]),
            jitter=np.array([13.1, 2.7]),
            noise=0.00224,
            quantile=0.95,
        ),
        grid=Grid(
            epsg=EPSG,
            upper_left=UPPER_LEFT,
            lower_right=(UPPER_LEFT[0] + columns * PIXEL, UPPER_LEFT[1] - rows * PIXEL),
            shape=(rows, columns),
        ),
    )


def test_the_map_is_written_where_the_scene_is(tmp_path: Path) -> None:
    """Without this the file is a picture, not a map."""
    mapped = _mapped([[0, 1], [1, UNCLASSIFIED]])

    with rasterio.open(write_map(mapped, mapped.grid, tmp_path / "map.tif")) as raster:
        assert raster.crs.to_epsg() == EPSG
        assert raster.bounds.left == UPPER_LEFT[0]
        assert raster.bounds.top == UPPER_LEFT[1]
        assert raster.res == (PIXEL, PIXEL)


def test_every_layer_behind_the_decision_travels_with_it(tmp_path: Path) -> None:
    mapped = _mapped([[0, 1]])

    with rasterio.open(write_map(mapped, mapped.grid, tmp_path / "map.tif")) as raster:
        assert raster.descriptions == BANDS
        assert raster.read(1)[0].tolist() == [0.0, 1.0]
        assert raster.read(3)[0] == pytest.approx([0.12, 0.12])


def test_the_class_numbers_are_readable_without_this_code(tmp_path: Path) -> None:
    """A bare integer raster is unusable by anyone who did not write it."""
    mapped = _mapped([[0, 1]])

    with rasterio.open(write_map(mapped, mapped.grid, tmp_path / "map.tif")) as raster:
        classes = raster.tags()["classes"]

    assert "0=alunite" in classes
    assert "1=kaolinite_group" in classes
    assert f"{UNCLASSIFIED}=unclassified" in classes


def test_the_thresholds_the_map_was_made_with_are_recorded(tmp_path: Path) -> None:
    mapped = _mapped([[0]])

    with rasterio.open(write_map(mapped, mapped.grid, tmp_path / "map.tif")) as raster:
        tags = raster.tags()

    assert tags["depth_floor"] == "0.0625"
    assert tags["angle_floor"] == "38.80"
    assert "Planet Labs PBC" in tags["attribution"]


def test_the_directory_is_made_if_it_is_not_there(tmp_path: Path) -> None:
    mapped = _mapped([[0]])

    written = write_map(mapped, mapped.grid, tmp_path / "deep" / "map.tif")

    assert written.exists()
